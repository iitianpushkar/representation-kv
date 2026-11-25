"""Qwen2.5 attention replacement and single-sequence greedy decoding.

This module targets the Qwen2Attention interface in Transformers 4.57.1.
It deliberately does not support padding, batching, beam search or sliding
window attention. Those cases fail early rather than silently mis-mask keys.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
import time

import torch
import torch.nn.functional as F
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from .cache import RetainedCache


@dataclass
class GenerationResult:
    token_ids: list[int]
    prefill_seconds: float
    decode_seconds: float
    kv_bytes_prefill: int
    kv_bytes_final: int
    metadata_bytes_final: int
    layer_counts_prefill: list[int]
    layer_counts_final: list[int]
    peak_gpu_bytes: int | None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cache_bytes(cache: DynamicCache) -> int:
    return sum(
        t.numel() * t.element_size()
        for layer in cache.layers
        for t in (layer.keys, layer.values)
        if t is not None
    )


class CompressedQwen:
    def __init__(self, model: torch.nn.Module, ratio: float):
        if transformers.__version__ != "4.57.1":
            raise RuntimeError("This attention patch requires transformers==4.57.1")
        if model.config.model_type != "qwen2":
            raise ValueError("Only Qwen2/Qwen2.5 models are supported")
        if getattr(model.config, "use_sliding_window", False):
            raise ValueError("Sliding-window attention is not supported")
        self.model = model
        self.cache = RetainedCache(model.config.num_hidden_layers, ratio)
        self.phase = "prefill"

    def _attention(
        self, module, hidden_states, position_embeddings, attention_mask,
        past_key_values=None, cache_position=None, **kwargs,
    ):
        if hidden_states.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported")
        if attention_mask is not None:
            raise ValueError("Pass the explicit no-padding mask mapping")
        sequence_length = hidden_states.shape[1]
        shape = (1, sequence_length, -1, module.head_dim)
        query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)

        layer = module.layer_idx
        if self.phase == "prefill":
            all_key, all_value = key, value
            causal = True
        else:
            old = self.cache.layers[layer]
            assert old is not None and sequence_length == 1
            all_key = torch.cat((old.keys, key), dim=2)
            all_value = torch.cat((old.values, value), dim=2)
            causal = False  # The only query is the newest token.

        groups = module.num_key_value_groups
        output = F.scaled_dot_product_attention(
            query, repeat_kv(all_key, groups), repeat_kv(all_value, groups),
            is_causal=causal, dropout_p=0.0, scale=module.scaling,
        )
        output = output.transpose(1, 2).reshape(1, sequence_length, -1)
        output = module.o_proj(output)

        # Selection happens after attention, so current-token attention always
        # includes its own KV even if that KV is not persisted.
        if self.phase == "prefill":
            self.cache.store_prefill(layer, hidden_states, key, value)
        else:
            assert cache_position is not None and cache_position.numel() == 1
            self.cache.store_decode(
                layer, int(cache_position.item()), hidden_states, key, value
            )
        return output, None

    @contextmanager
    def _patched(self):
        modules = [layer.self_attn for layer in self.model.model.layers]
        originals = [module.forward for module in modules]
        try:
            for module in modules:
                module.forward = MethodType(self._attention, module)
            yield
        finally:
            for module, original in zip(modules, originals):
                module.forward = original

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, T]")
        if input_ids.shape[1] < 1 or max_new_tokens < 1:
            raise ValueError("Need a non-empty prompt and positive max_new_tokens")
        device = input_ids.device
        prompt_length = input_ids.shape[1]
        self.cache.start_prefill(prompt_length)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with self._patched():
            self.phase = "prefill"
            _sync(device)
            start = time.perf_counter()
            result = self.model(
                input_ids=input_ids,
                attention_mask={"full_attention": None},
                use_cache=False,
                logits_to_keep=1,
            )
            _sync(device)
            prefill_seconds = time.perf_counter() - start
            self.cache.finish_token_pass()
            kv_bytes_prefill = self.cache.kv_bytes()
            counts_prefill = self.cache.counts()

            tokens: list[int] = []
            next_id = result.logits[:, -1].argmax(dim=-1, keepdim=True)
            decode_seconds = 0.0
            for step in range(max_new_tokens):
                token = int(next_id.item())
                tokens.append(token)
                if token == eos_token_id or step + 1 == max_new_tokens:
                    break
                self.phase = "decode"
                self.cache.start_decode_token()
                position = prompt_length + step
                position_tensor = torch.tensor([position], device=device)
                _sync(device)
                start = time.perf_counter()
                result = self.model(
                    input_ids=next_id,
                    attention_mask={"full_attention": None},
                    cache_position=position_tensor,
                    position_ids=position_tensor.unsqueeze(0),
                    use_cache=False,
                    logits_to_keep=1,
                )
                _sync(device)
                decode_seconds += time.perf_counter() - start
                self.cache.finish_token_pass()
                next_id = result.logits[:, -1].argmax(dim=-1, keepdim=True)

        return GenerationResult(
            tokens, prefill_seconds, decode_seconds, kv_bytes_prefill,
            self.cache.kv_bytes(), self.cache.metadata_bytes(), counts_prefill,
            self.cache.counts(),
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        )


@torch.inference_mode()
def generate_full(model, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None):
    """Same greedy loop with the unmodified Qwen DynamicCache baseline."""
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    start = time.perf_counter()
    result = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    _sync(device)
    prefill_seconds = time.perf_counter() - start
    cache = result.past_key_values
    kv_bytes_prefill = _cache_bytes(cache)
    counts_prefill = [layer.keys.shape[2] for layer in cache.layers]
    next_id = result.logits[:, -1].argmax(dim=-1, keepdim=True)
    tokens: list[int] = []
    decode_seconds = 0.0
    for step in range(max_new_tokens):
        token = int(next_id.item())
        tokens.append(token)
        if token == eos_token_id or step + 1 == max_new_tokens:
            break
        _sync(device)
        start = time.perf_counter()
        result = model(input_ids=next_id, past_key_values=cache, use_cache=True, logits_to_keep=1)
        _sync(device)
        decode_seconds += time.perf_counter() - start
        cache = result.past_key_values
        next_id = result.logits[:, -1].argmax(dim=-1, keepdim=True)
    return GenerationResult(
        tokens, prefill_seconds, decode_seconds, kv_bytes_prefill,
        _cache_bytes(cache), 0, counts_prefill,
        [layer.keys.shape[2] for layer in cache.layers],
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    )
