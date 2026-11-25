"""Per-layer, position-preserving KV storage for batch-size-one inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch


def budget(ratio: float, token_count: int) -> int:
    if not 0 < ratio <= 1:
        raise ValueError("retention ratio must be in (0, 1]")
    return math.ceil(ratio * token_count)


def changed(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Cosine distance per token, computed in fp32 for stable ranking."""
    return 1 - torch.nn.functional.cosine_similarity(
        current.float(), reference.float(), dim=-1, eps=1e-8
    )


@dataclass
class LayerKV:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    scores: torch.Tensor

    @property
    def count(self) -> int:
        return self.positions.numel()

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.keys, self.values, self.positions, self.scores)
        )


class RetainedCache:
    """A separate compact cache per attention layer.

    Prefill sees every token before selecting top-K. Decode is online: only
    retained entries and the arriving token compete for the growing budget.
    Evicted entries cannot be restored when the budget later increases.
    """

    def __init__(self, layers: int, ratio: float):
        if not 0 < ratio <= 1:
            raise ValueError("retention ratio must be in (0, 1]")
        self.ratio = ratio
        self.layers: list[LayerKV | None] = [None] * layers
        self.reference: torch.Tensor | None = None
        self.reference_layer: torch.Tensor | None = None
        self.prompt_tokens = 0
        self.prefill_scores: list[torch.Tensor] = []
        self.prefill_selected: list[torch.Tensor] = []
        self.prefill_reference_layers: list[torch.Tensor] = []
        self.decode_scores: list[list[torch.Tensor]] = [[] for _ in range(layers)]
        self.decode_selected: list[list[bool]] = [[] for _ in range(layers)]
        self.decode_reference_layers: list[list[int]] = [[] for _ in range(layers)]
        self.decode_evicted_positions: list[list[int]] = [[] for _ in range(layers)]

    def start_prefill(self, token_count: int) -> None:
        self.prompt_tokens = token_count
        self.reference = None
        self.reference_layer = None
        self.layers = [None] * len(self.layers)
        self.prefill_scores = []
        self.prefill_selected = []
        self.prefill_reference_layers = []
        self.decode_scores = [[] for _ in self.layers]
        self.decode_selected = [[] for _ in self.layers]
        self.decode_reference_layers = [[] for _ in self.layers]
        self.decode_evicted_positions = [[] for _ in self.layers]

    def start_decode_token(self) -> None:
        self.reference = None
        self.reference_layer = None

    def finish_token_pass(self) -> None:
        # Historical tokens need no hidden-state references after their last layer.
        self.reference = None
        self.reference_layer = None

    def store_prefill(
        self, layer: int, hidden: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        # Shapes: hidden [1,T,D], K/V [1,Hkv,T,Dh]. No padding is supported.
        if layer == 0:
            self.reference = hidden.detach().clone()
            self.reference_layer = torch.zeros(
                hidden.shape[1], dtype=torch.int16, device=hidden.device
            )
            scores = torch.full((hidden.shape[1],), float("inf"), device=hidden.device)
            indices = torch.arange(hidden.shape[1], device=hidden.device)
            self.prefill_scores.append(torch.full_like(scores, float("nan")))
            self.prefill_reference_layers.append(torch.full_like(self.reference_layer, -1))
        else:
            assert self.reference is not None and self.reference_layer is not None
            scores = changed(hidden[0], self.reference[0])
            self.prefill_scores.append(scores.detach().clone())
            self.prefill_reference_layers.append(self.reference_layer.clone())
            k = budget(self.ratio, hidden.shape[1])
            indices = torch.topk(scores, k=k, sorted=False).indices.sort().values
            self.reference[:, indices] = hidden[:, indices]
            self.reference_layer[indices] = layer
        selected = torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        selected[indices] = True
        self.prefill_selected.append(selected)
        positions = indices.to(torch.long)
        self.layers[layer] = LayerKV(
            keys.index_select(2, indices).contiguous(),
            values.index_select(2, indices).contiguous(),
            positions,
            scores.index_select(0, indices).contiguous(),
        )

    def store_decode(
        self,
        layer: int,
        position: int,
        hidden: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> bool:
        entry = self.layers[layer]
        assert entry is not None
        if layer == 0:
            self.reference = hidden.detach().clone()
            self.reference_layer = torch.tensor([0], dtype=torch.int16, device=hidden.device)
            score = torch.tensor([float("inf")], device=hidden.device)
            trace_score = torch.tensor(float("nan"), device=hidden.device)
            previous_layer = -1
            keep = True
        else:
            assert self.reference is not None and self.reference_layer is not None
            previous_layer = int(self.reference_layer.item())
            score = changed(hidden[0], self.reference[0])
            trace_score = score[0].detach().clone()
            keep = False

        target = position + 1 if layer == 0 else budget(self.ratio, position + 1)
        evicted_position = -1
        if layer != 0:
            if entry.count < target:
                keep = True
            elif bool(score[0] > entry.scores.min()):
                keep = True
                lowest = int(entry.scores.argmin())
                evicted_position = int(entry.positions[lowest].item())
                mask = torch.ones(entry.count, dtype=torch.bool, device=hidden.device)
                mask[lowest] = False
                entry.keys = entry.keys[:, :, mask, :]
                entry.values = entry.values[:, :, mask, :]
                entry.positions = entry.positions[mask]
                entry.scores = entry.scores[mask]
        if keep:
            entry.keys = torch.cat((entry.keys, key), dim=2)
            entry.values = torch.cat((entry.values, value), dim=2)
            entry.positions = torch.cat(
                (entry.positions, torch.tensor([position], device=hidden.device))
            )
            entry.scores = torch.cat((entry.scores, score))
            if layer != 0:
                self.reference = hidden.detach().clone()
                self.reference_layer[0] = layer
        self.decode_scores[layer].append(trace_score)
        self.decode_selected[layer].append(keep)
        self.decode_reference_layers[layer].append(previous_layer)
        self.decode_evicted_positions[layer].append(evicted_position)
        return keep

    def save_trace(self, path: str | Path) -> None:
        """Save all token-layer scores and selection decisions as compressed NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        layers = len(self.layers)
        decoded = len(self.decode_scores[0])
        if any(len(row) != decoded for row in self.decode_scores):
            raise RuntimeError("Incomplete decode score trace")
        decode_scores = np.empty((layers, decoded), dtype=np.float32)
        for layer, values in enumerate(self.decode_scores):
            if values:
                decode_scores[layer] = torch.stack(values).float().cpu().numpy()
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                prefill_scores=torch.stack(self.prefill_scores).float().cpu().numpy(),
                prefill_selected=torch.stack(self.prefill_selected).cpu().numpy(),
                prefill_reference_layer=torch.stack(
                    self.prefill_reference_layers).cpu().numpy(),
                decode_scores=decode_scores,
                decode_selected=np.asarray(self.decode_selected, dtype=np.bool_),
                decode_reference_layer=np.asarray(
                    self.decode_reference_layers, dtype=np.int16),
                decode_evicted_position=np.asarray(
                    self.decode_evicted_positions, dtype=np.int64),
                decode_positions=np.arange(self.prompt_tokens,
                                           self.prompt_tokens + decoded, dtype=np.int64),
            )
        temporary.replace(path)

    def kv_bytes(self) -> int:
        return sum(
            (e.keys.numel() * e.keys.element_size() + e.values.numel() * e.values.element_size())
            for e in self.layers if e is not None
        )

    def metadata_bytes(self) -> int:
        return sum(
            (e.positions.numel() * e.positions.element_size() + e.scores.numel() * e.scores.element_size())
            for e in self.layers if e is not None
        )

    def counts(self) -> list[int]:
        return [e.count if e is not None else 0 for e in self.layers]
