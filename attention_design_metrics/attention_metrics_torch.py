from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a [T, T] mask where row t can attend to columns <= t."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def make_sliding_window_mask(
    seq_len: int,
    window: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a causal local-attention mask.

    window includes the current token. For example, window=4 means token t can
    attend to t, t-1, t-2, and t-3.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    row = torch.arange(seq_len, device=device)[:, None]
    col = torch.arange(seq_len, device=device)[None, :]
    return (col <= row) & (col >= row - window + 1)


def qk_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute scaled QK logits for q/k shaped [B, H, T, D]."""
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [B, H, T, D]")
    return q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last axis with a [T, T] boolean attention mask."""
    mask = mask.to(device=logits.device, dtype=torch.bool)
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(0)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=-1)


def attention_probs(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked attention probabilities for q/k shaped [B, H, T, D]."""
    return masked_softmax(qk_logits(q, k), mask)


def _masked_values(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=x.device, dtype=torch.bool)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(0)
    return x.masked_select(mask.expand_as(x)).float()


def qk_logit_stats(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Robust statistics for masked QK logits.

    Prefer p99/p99.9 over max when comparing designs, because max can be a
    single numerical outlier.
    """
    logits = qk_logits(q, k)
    values = _masked_values(logits, mask)
    if values.numel() == 0:
        return {name: float("nan") for name in ["mean", "std", "p99", "p999", "max", "min"]}
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "p99": values.quantile(0.99).item(),
        "p999": values.quantile(0.999).item(),
        "max": values.max().item(),
        "min": values.min().item(),
    }


def normalized_attention_entropy(
    probs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return H(p) / log(N_visible), shaped like probs without the last axis."""
    mask = mask.to(device=probs.device, dtype=torch.bool)
    while mask.ndim < probs.ndim:
        mask = mask.unsqueeze(0)
    p = probs.masked_fill(~mask, 0.0).float()
    entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1)
    visible = mask.expand_as(probs).sum(dim=-1).float().clamp_min(1.0)
    denom = visible.log()
    return torch.where(visible > 1, entropy / denom.clamp_min(eps), torch.zeros_like(entropy))


def effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy effective rank: exp(-sum p_i log p_i), p_i=sigma_i/sum sigma."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    singular_values = torch.linalg.svdvals(matrix.float())
    weights = singular_values / singular_values.sum().clamp_min(eps)
    return torch.exp(-(weights * weights.clamp_min(eps).log()).sum())


def batch_effective_rank(matrices: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute effective rank for a batch of matrices shaped [..., M, N]."""
    flat = matrices.reshape(-1, matrices.shape[-2], matrices.shape[-1])
    ranks = [effective_rank(m, eps=eps) for m in flat]
    return torch.stack(ranks).reshape(matrices.shape[:-2])


def head_diversity(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    subtract_mask_baseline: bool = True,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Estimate diversity across attention heads.

    probs must be [B, H, T, T]. If subtract_mask_baseline=True, the uniform
    distribution induced by the mask is removed before comparing heads. This
    prevents every causal head from looking artificially similar.
    """
    if probs.ndim != 4:
        raise ValueError("probs must have shape [B, H, T, T]")
    x = probs.float()
    if subtract_mask_baseline:
        if mask is None:
            raise ValueError("mask is required when subtract_mask_baseline=True")
        mask = mask.to(device=probs.device, dtype=torch.bool)
        baseline = mask.float() / mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
        x = x - baseline[None, None, :, :]

    per_head = x.mean(dim=0).reshape(x.shape[1], -1)
    per_head = F.normalize(per_head, dim=-1, eps=eps)
    sim = per_head @ per_head.T
    off_diag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    mean_cosine = off_diag.mean().item() if off_diag.numel() else 1.0
    return {
        "mean_pairwise_cosine": mean_cosine,
        "diversity_1_minus_cosine": 1.0 - mean_cosine,
    }


def receptive_field_matrix(mask: torch.Tensor, layers: int) -> torch.Tensor:
    """Boolean [T, T] matrix: output row t can receive information from input col s."""
    if layers < 0:
        raise ValueError("layers must be >= 0")
    mask = mask.bool()
    seq_len = mask.shape[0]
    reach = torch.eye(seq_len, dtype=torch.bool, device=mask.device)
    adj = mask.float()
    for _ in range(layers):
        reach = (adj @ reach.float()) > 0
    return reach


def graph_diameter(mask: torch.Tensor, max_layers: Optional[int] = None) -> Dict[str, object]:
    """Shortest layer distance between causal input/output token pairs.

    Returns the largest finite distance over all causal pairs, plus the count
    of unreachable causal pairs within max_layers.
    """
    mask = mask.bool()
    seq_len = mask.shape[0]
    if max_layers is None:
        max_layers = seq_len

    dist = torch.full((seq_len, seq_len), float("inf"), device=mask.device)
    reach = torch.eye(seq_len, dtype=torch.bool, device=mask.device)
    dist[reach] = 0
    adj = mask.float()

    for layer in range(1, max_layers + 1):
        reach = (adj @ reach.float()) > 0
        newly_reached = reach & torch.isinf(dist)
        dist[newly_reached] = layer

    causal_pairs = make_causal_mask(seq_len, device=mask.device)
    causal_dist = dist[causal_pairs]
    finite = torch.isfinite(causal_dist)
    diameter = int(causal_dist[finite].max().item()) if finite.any() else math.inf
    return {
        "diameter": diameter,
        "unreachable_pairs": int((~finite).sum().item()),
        "distance_matrix": dist,
    }


def attention_rollout(attn: torch.Tensor, residual_weight: float = 0.5) -> torch.Tensor:
    """Compute attention rollout across layers.

    Accepts [L, H, T, T] or [L, T, T]. Heads are averaged. The residual path is
    modeled by blending each layer with the identity matrix.
    """
    if attn.ndim == 4:
        attn = attn.mean(dim=1)
    if attn.ndim != 3:
        raise ValueError("attn must have shape [L, H, T, T] or [L, T, T]")
    layers, seq_len, _ = attn.shape
    eye = torch.eye(seq_len, dtype=attn.dtype, device=attn.device)
    rollout = eye
    for layer in range(layers):
        mixed = residual_weight * eye + (1.0 - residual_weight) * attn[layer]
        mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = mixed @ rollout
    return rollout


def estimate_kv_cache_bytes(
    layers: int,
    batch_size: int,
    seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate decode KV cache memory."""
    return layers * batch_size * seq_len * 2 * n_kv_heads * head_dim * dtype_bytes


@dataclass
class BenchmarkResult:
    median_ms: float
    p95_ms: float
    tokens_per_sec: float


def benchmark_forward(
    fn,
    tokens: torch.Tensor,
    warmup: int = 5,
    iters: int = 20,
) -> BenchmarkResult:
    """Simple wall-clock benchmark for a forward function."""
    device = tokens.device
    for _ in range(warmup):
        fn(tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)

    t = torch.tensor(times)
    median_ms = t.median().item()
    p95_ms = t.quantile(0.95).item()
    tokens_per_sec = tokens.numel() / (median_ms / 1000.0)
    return BenchmarkResult(median_ms=median_ms, p95_ms=p95_ms, tokens_per_sec=tokens_per_sec)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class TinyCausalSelfAttention(nn.Module):
    """Small educational attention layer with full or sliding-window masks."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_kind: str = "full",
        window: Optional[int] = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if attn_kind not in {"full", "sliding"}:
            raise ValueError("attn_kind must be 'full' or 'sliding'")
        if attn_kind == "sliding" and window is None:
            raise ValueError("window is required for sliding attention")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_kind = attn_kind
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def make_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.attn_kind == "full":
            return make_causal_mask(seq_len, device=device)
        return make_sliding_window_mask(seq_len, self.window or 1, device=device)

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        mask = self.make_mask(seq_len, x.device)
        probs = attention_probs(q, k, mask)
        y = probs @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        y = self.out(y)
        if return_probs:
            return y, probs, mask
        return y


class TinyTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        attn_kind: str = "full",
        window: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = TinyCausalSelfAttention(d_model, n_heads, attn_kind=attn_kind, window=window)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.SiLU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        if return_probs:
            attn_out, probs, mask = self.attn(self.norm1(x), return_probs=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, probs, mask
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyKVModel(nn.Module):
    """Tiny model for key-value retrieval experiments.

    Input sequence: k1, v1, k2, v2, ..., query_marker, query_key.
    Target: the value token paired with query_key.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        attn_kind: str = "full",
        window: Optional[int] = None,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TinyTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    attn_kind=attn_kind,
                    window=window,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)[None, :]
        x = self.token_emb(tokens) + self.pos_emb(positions)
        all_probs = []
        mask = None
        for block in self.blocks:
            if return_attn:
                x, probs, mask = block(x, return_probs=True)
                all_probs.append(probs.detach())
            else:
                x = block(x)
        logits = self.head(self.norm(x[:, -1]))
        if return_attn:
            return logits, torch.stack(all_probs), mask
        return logits


def make_kv_retrieval_batch(
    batch_size: int,
    num_pairs: int,
    num_keys: int,
    num_values: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Create a synthetic key-value retrieval batch.

    Keys are sampled without replacement inside each example to avoid
    ambiguous duplicate keys.
    """
    if num_keys < num_pairs:
        raise ValueError("num_keys must be >= num_pairs")
    key_scores = torch.rand(batch_size, num_keys, device=device)
    keys = key_scores.topk(num_pairs, dim=-1).indices
    values = torch.randint(0, num_values, (batch_size, num_pairs), device=device) + num_keys
    query_index = torch.randint(0, num_pairs, (batch_size,), device=device)
    row = torch.arange(batch_size, device=device)
    query_key = keys[row, query_index]
    target = values[row, query_index]

    query_marker = num_keys + num_values
    tokens = torch.empty(batch_size, 2 * num_pairs + 2, dtype=torch.long, device=device)
    tokens[:, 0 : 2 * num_pairs : 2] = keys
    tokens[:, 1 : 2 * num_pairs : 2] = values
    tokens[:, -2] = query_marker
    tokens[:, -1] = query_key
    vocab_size = num_keys + num_values + 1
    return tokens, target, vocab_size


@torch.no_grad()
def retrieval_accuracy(model: nn.Module, tokens: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    pred = model(tokens).argmax(dim=-1)
    return (pred == targets).float().mean().item()
