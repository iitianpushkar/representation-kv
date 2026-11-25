"""Per-layer, position-preserving KV storage for batch-size-one inference."""

from __future__ import annotations

from dataclasses import dataclass
import math

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
        self.prompt_tokens = 0

    def start_prefill(self, token_count: int) -> None:
        self.prompt_tokens = token_count
        self.reference = None
        self.layers = [None] * len(self.layers)

    def start_decode_token(self) -> None:
        self.reference = None

    def finish_token_pass(self) -> None:
        # Historical tokens need no hidden-state references after their last layer.
        self.reference = None

    def store_prefill(
        self, layer: int, hidden: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        # Shapes: hidden [1,T,D], K/V [1,Hkv,T,Dh]. No padding is supported.
        if layer == 0:
            self.reference = hidden.detach().clone()
            scores = torch.full((hidden.shape[1],), float("inf"), device=hidden.device)
            indices = torch.arange(hidden.shape[1], device=hidden.device)
        else:
            assert self.reference is not None
            scores = changed(hidden[0], self.reference[0])
            k = budget(self.ratio, hidden.shape[1])
            indices = torch.topk(scores, k=k, sorted=False).indices.sort().values
            self.reference[:, indices] = hidden[:, indices]
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
            score = torch.tensor([float("inf")], device=hidden.device)
            keep = True
        else:
            assert self.reference is not None
            score = changed(hidden[0], self.reference[0])
            keep = False

        target = position + 1 if layer == 0 else budget(self.ratio, position + 1)
        if layer != 0:
            if entry.count < target:
                keep = True
            elif bool(score[0] > entry.scores.min()):
                keep = True
                lowest = int(entry.scores.argmin())
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
        return keep

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
