#!/usr/bin/env python3
"""CPU smoke test using random Qwen2 weights; no model download required."""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from representation_kv.qwen import CompressedQwen, generate_full


def main():
    torch.manual_seed(123)
    config = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config).eval()
    model.config._attn_implementation = "sdpa"
    tokens = torch.randint(0, 128, (1, 9))
    baseline_cache = DynamicCache(config=config)
    with torch.inference_mode():
        baseline_prefill = model(tokens, past_key_values=baseline_cache,
                                 use_cache=True, logits_to_keep=1).logits
        compressor = CompressedQwen(model, 1.0)
        compressor.cache.start_prefill(tokens.shape[1])
        with compressor._patched():
            compressor.phase = "prefill"
            compressed_prefill = model(
                tokens, attention_mask={"full_attention": None},
                use_cache=False, logits_to_keep=1,
            ).logits
            compressor.cache.finish_token_pass()
            assert torch.allclose(baseline_prefill, compressed_prefill, atol=1e-5)
            next_token = baseline_prefill[:, -1].argmax(dim=-1, keepdim=True)
            baseline_decode = None
        baseline_decode = model(
            next_token, past_key_values=baseline_cache, use_cache=True,
            logits_to_keep=1,
        ).logits
        compressor.cache.start_decode_token()
        compressor.phase = "decode"
        position = torch.tensor([tokens.shape[1]])
        with compressor._patched():
            compressed_decode = model(
                next_token, attention_mask={"full_attention": None},
                cache_position=position, position_ids=position.unsqueeze(0),
                use_cache=False, logits_to_keep=1,
            ).logits
        assert torch.allclose(baseline_decode, compressed_decode, atol=1e-5)
    full = generate_full(model, tokens, 4, None)
    exact = CompressedQwen(model, 1.0).generate(tokens, 4, None)
    assert full.token_ids == exact.token_ids, (full.token_ids, exact.token_ids)
    assert exact.layer_counts_prefill == [9] * 4
    assert exact.layer_counts_final == [12] * 4
    half = CompressedQwen(model, 0.5).generate(tokens, 4, None)
    assert half.layer_counts_prefill == [9, 5, 5, 5], half.layer_counts_prefill
    assert half.layer_counts_final == [12, 6, 6, 6], half.layer_counts_final
    assert half.kv_bytes_prefill < full.kv_bytes_prefill
    print("PASS: full-KV prefill/decode logits, token match, layer counts, and online budget")


if __name__ == "__main__":
    main()
