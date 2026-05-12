"""Attention design metrics and tiny retrieval model for teaching.

This file supports `attention_metrics_tutorial.ipynb`. It focuses on small,
inspectable tensors rather than production kernels. The functions are designed
to answer questions such as:

- Can information flow from distant tokens to the current token?
- Are QK logits stable, or does softmax saturate?
- Do attention heads behave differently?
- How much KV cache does a design need at inference time?

Shape notation used below:
- B = batch size
- H = number of attention heads
- T = sequence length
- D = head dimension or hidden dimension, depending on context
- V = vocabulary size
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a full causal attention mask.

    Inputs:
        seq_len: Sequence length T.
        device: Optional device for the output tensor.

    Output:
        Boolean tensor shaped [T, T]. Row t is True for columns <= t.

    Example:
        >>> make_causal_mask(4).int()
        tensor([[1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1]], dtype=torch.int32)
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def make_sliding_window_mask(
    seq_len: int,
    window: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a causal local-attention mask.

    Inputs:
        seq_len: Sequence length T.
        window: Number of visible tokens per row, including the current token.
        device: Optional device for the output tensor.

    Output:
        Boolean tensor shaped [T, T].

    Example:
        window=2 means token t can attend to t and t-1 only.

        >>> make_sliding_window_mask(4, 2).int()
        tensor([[1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1]], dtype=torch.int32)
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    row = torch.arange(seq_len, device=device)[:, None]
    col = torch.arange(seq_len, device=device)[None, :]
    return (col <= row) & (col >= row - window + 1)


def qk_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute scaled QK attention logits.

    Inputs:
        q: Query tensor shaped [B, H, T, D].
        k: Key tensor shaped [B, H, T, D].

    Output:
        Float tensor shaped [B, H, T, T]. Entry [b, h, t, s] is the score for
        query token t attending to key token s.

    Example:
        >>> q = torch.randn(2, 4, 8, 16)
        >>> k = torch.randn(2, 4, 8, 16)
        >>> qk_logits(q, k).shape
        torch.Size([2, 4, 8, 8])
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [B, H, T, D]")
    return q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over visible attention positions only.

    Inputs:
        logits: Float tensor shaped [B, H, T, T] or any tensor ending in [T, T].
        mask: Boolean tensor shaped [T, T]. False entries are hidden.

    Output:
        Tensor with the same shape as logits. Hidden positions get probability
        close to 0, visible positions sum to 1 along the last axis.

    Example:
        >>> logits = torch.zeros(1, 1, 3, 3)
        >>> probs = masked_softmax(logits, make_causal_mask(3))
        >>> probs[0, 0, 0]
        tensor([1., 0., 0.])
    """
    mask = mask.to(device=logits.device, dtype=torch.bool)
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(0)
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=-1)


def attention_probs(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked attention probabilities from Q and K.

    Inputs:
        q: Query tensor shaped [B, H, T, D].
        k: Key tensor shaped [B, H, T, D].
        mask: Boolean attention mask shaped [T, T].

    Output:
        Probability tensor shaped [B, H, T, T].
    """
    return masked_softmax(qk_logits(q, k), mask)


def _masked_values(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return flattened values of x where mask is True.

    This helper broadcasts a [T, T] mask across leading dimensions such as
    batch and head. It is internal because teaching notebooks usually call
    `qk_logit_stats` instead.
    """
    mask = mask.to(device=x.device, dtype=torch.bool)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(0)
    return x.masked_select(mask.expand_as(x)).float()


def qk_logit_stats(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Robust statistics for masked QK logits.

    Inputs:
        q: Query tensor shaped [B, H, T, D].
        k: Key tensor shaped [B, H, T, D].
        mask: Boolean mask shaped [T, T].

    Output:
        Python dict with mean, std, p99, p999, max, and min over visible logits.

    Example:
        >>> q = torch.randn(2, 4, 8, 16)
        >>> k = torch.randn(2, 4, 8, 16)
        >>> stats = qk_logit_stats(q, k, make_causal_mask(8))
        >>> sorted(stats)
        ['max', 'mean', 'min', 'p99', 'p999', 'std']

    Teaching point:
        Very large logits make softmax too sharp and can destabilize training.
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
    """Return normalized attention entropy H(p) / log(N_visible).

    Inputs:
        probs: Attention probabilities shaped [B, H, T, T].
        mask: Boolean attention mask shaped [T, T].
        eps: Numerical stability constant.

    Output:
        Tensor shaped [B, H, T]. Each value is in roughly [0, 1].

    Example:
        >>> q = torch.randn(1, 2, 4, 8)
        >>> p = attention_probs(q, q, make_causal_mask(4))
        >>> normalized_attention_entropy(p, make_causal_mask(4)).shape
        torch.Size([1, 2, 4])

    Interpretation:
        1 means nearly uniform over visible tokens. 0 means nearly one-hot.
    """
    mask = mask.to(device=probs.device, dtype=torch.bool)
    while mask.ndim < probs.ndim:
        mask = mask.unsqueeze(0)
    p = probs.masked_fill(~mask, 0.0).float()
    entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1)
    visible = mask.expand_as(probs).sum(dim=-1).float().clamp_min(1.0)
    denom = visible.log()
    return torch.where(visible > 1, entropy / denom.clamp_min(eps), torch.zeros_like(entropy))


def effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy effective rank of one matrix.

    Inputs:
        matrix: 2D tensor shaped [M, N].
        eps: Numerical stability constant.

    Output:
        Scalar tensor. Higher means the matrix uses more independent
        directions; lower means it is closer to low-rank collapse.

    Formula:
        p_i = sigma_i / sum_j sigma_j
        r_eff = exp(-sum_i p_i log p_i)

    Example:
        >>> effective_rank(torch.eye(4)).round()
        tensor(4.)
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    singular_values = torch.linalg.svdvals(matrix.float())
    weights = singular_values / singular_values.sum().clamp_min(eps)
    return torch.exp(-(weights * weights.clamp_min(eps).log()).sum())


def batch_effective_rank(matrices: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute effective rank for a batch of matrices.

    Inputs:
        matrices: Tensor shaped [..., M, N].
        eps: Numerical stability constant.

    Output:
        Tensor shaped [...].

    Example:
        >>> x = torch.eye(4).repeat(3, 1, 1)
        >>> batch_effective_rank(x).shape
        torch.Size([3])
    """
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

    Inputs:
        probs: Attention probabilities shaped [B, H, T, T].
        mask: Boolean mask shaped [T, T], required when subtracting baseline.
        subtract_mask_baseline: If True, remove the uniform distribution
            induced by the mask before comparing heads.
        eps: Normalization stability constant.

    Output:
        Dict with:
            mean_pairwise_cosine: average off-diagonal head similarity.
            diversity_1_minus_cosine: 1 - similarity.

    Example:
        >>> q = torch.randn(2, 4, 8, 16)
        >>> mask = make_causal_mask(8)
        >>> probs = attention_probs(q, q, mask)
        >>> head_diversity(probs, mask).keys()
        dict_keys(['mean_pairwise_cosine', 'diversity_1_minus_cosine'])

    Teaching point:
        The causal mask makes all heads share a triangular support pattern.
        Subtracting the mask baseline avoids overestimating similarity.
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
    """Compute theoretical receptive field after several attention layers.

    Inputs:
        mask: Boolean attention mask shaped [T, T].
        layers: Number of stacked attention layers.

    Output:
        Boolean tensor shaped [T, T]. output row t can receive information
        from input column s if result[t, s] is True.

    Example:
        >>> mask = make_sliding_window_mask(8, window=2)
        >>> receptive_field_matrix(mask, layers=2)[-1].sum()
        tensor(3)

    Teaching point:
        This is theoretical connectivity, not learned attention mass. A token
        can be reachable in the graph but still receive almost no probability.
    """
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

    Inputs:
        mask: Boolean attention mask shaped [T, T].
        max_layers: Search depth. Defaults to T.

    Output:
        Dict with:
            diameter: Largest finite shortest path over causal token pairs.
            unreachable_pairs: Number of causal pairs not connected.
            distance_matrix: [T, T] matrix of shortest layer distances.

    Example:
        >>> graph_diameter(make_causal_mask(8))["diameter"]
        1
        >>> graph_diameter(make_sliding_window_mask(8, 2))["diameter"]
        7
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

    Inputs:
        attn: Attention probabilities shaped [L, H, T, T] or [L, T, T].
        residual_weight: How much identity path to mix into each layer.

    Output:
        Tensor shaped [T, T]. Row t estimates how much final token t depends
        on each input token.

    Example:
        >>> attn = torch.eye(4).repeat(2, 1, 1)
        >>> attention_rollout(attn).shape
        torch.Size([4, 4])

    Notes:
        Heads are averaged before rollout. This is a diagnostic approximation,
        not an exact causal attribution method.
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
    """Estimate decode-time KV cache memory in bytes.

    Inputs:
        layers: Number of Transformer layers.
        batch_size: Number of sequences served together.
        seq_len: Cached context length.
        n_kv_heads: Number of key/value heads. MHA uses n_heads, GQA uses fewer,
            MQA uses 1.
        head_dim: Dimension per KV head.
        dtype_bytes: Bytes per scalar, e.g. 2 for fp16/bf16.

    Output:
        Integer number of bytes.

    Example:
        >>> estimate_kv_cache_bytes(32, 1, 8192, 8, 128, 2) / 1024**3
        1.0
    """
    return layers * batch_size * seq_len * 2 * n_kv_heads * head_dim * dtype_bytes


@dataclass
class BenchmarkResult:
    """Forward benchmark result.

    Attributes:
        median_ms: Median forward latency in milliseconds.
        p95_ms: 95th percentile forward latency in milliseconds.
        tokens_per_sec: Input tokens processed per second.
    """

    median_ms: float
    p95_ms: float
    tokens_per_sec: float


def benchmark_forward(
    fn,
    tokens: torch.Tensor,
    warmup: int = 5,
    iters: int = 20,
) -> BenchmarkResult:
    """Simple wall-clock benchmark for a forward function.

    Inputs:
        fn: Callable that accepts `tokens` and runs one forward pass.
        tokens: Long tensor shaped [B, T].
        warmup: Number of unmeasured warmup iterations.
        iters: Number of measured iterations.

    Output:
        BenchmarkResult with latency and throughput.

    Example:
        >>> model = TinyKVModel(vocab_size=20, max_seq_len=6, d_model=32, n_heads=4, n_layers=1)
        >>> tokens = torch.randint(0, 20, (2, 6))
        >>> result = benchmark_forward(lambda x: model(x), tokens, warmup=1, iters=2)
        >>> result.tokens_per_sec > 0
        True
    """
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
    """RMSNorm for the tiny retrieval model.

    Forward input/output:
        x: [..., D] -> [..., D].
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize each token vector by RMS magnitude, then apply learned scale.
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class TinyCausalSelfAttention(nn.Module):
    """Small educational attention layer with full or sliding-window masks.

    Inputs:
        d_model: Hidden size D.
        n_heads: Number of attention heads H.
        attn_kind: "full" or "sliding".
        window: Local window size when attn_kind="sliding".

    Forward input:
        x: Float tensor shaped [B, T, D].
        return_probs: If True, also return attention probabilities and mask.

    Forward output:
        If return_probs=False:
            y: [B, T, D]
        If return_probs=True:
            y: [B, T, D]
            probs: [B, H, T, T]
            mask: [T, T]

    Example:
        >>> attn = TinyCausalSelfAttention(64, 4, attn_kind="sliding", window=8)
        >>> y, probs, mask = attn(torch.randn(2, 16, 64), return_probs=True)
        >>> y.shape, probs.shape, mask.shape
        (torch.Size([2, 16, 64]), torch.Size([2, 4, 16, 16]), torch.Size([16, 16]))
    """

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
        """Create the attention mask used by this layer.

        Output:
            Boolean tensor shaped [T, T].
        """
        if self.attn_kind == "full":
            return make_causal_mask(seq_len, device=device)
        return make_sliding_window_mask(seq_len, self.window or 1, device=device)

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        batch, seq_len, _ = x.shape

        # One projection produces Q, K, V. After view:
        # qkv is [B, T, 3, H, head_dim].
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Move heads before sequence: [B, T, H, D_head] -> [B, H, T, D_head].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        mask = self.make_mask(seq_len, x.device)
        probs = attention_probs(q, k, mask)

        # Attention output per head, then merge heads back to model dimension.
        y = probs @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        y = self.out(y)
        if return_probs:
            return y, probs, mask
        return y


class TinyTransformerBlock(nn.Module):
    """Tiny Pre-Norm Transformer block used by TinyKVModel.

    Forward input/output:
        x: [B, T, D] -> [B, T, D].

    Optional output:
        If return_probs=True, returns (x, probs, mask) so the notebook can
        inspect attention behavior.
    """

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
            # Residual update from attention.
            x = x + attn_out
            # Residual update from MLP.
            x = x + self.mlp(self.norm2(x))
            return x, probs, mask
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyKVModel(nn.Module):
    """Tiny model for key-value retrieval experiments.

    Input sequence: k1, v1, k2, v2, ..., query_marker, query_key.
    Target: the value token paired with query_key.

    Inputs:
        vocab_size: Total token vocabulary size V.
        max_seq_len: Maximum sequence length T.
        d_model: Hidden size D.
        n_heads: Attention heads H.
        n_layers: Number of Transformer blocks.
        attn_kind: "full" or "sliding".
        window: Local window size for sliding attention.

    Forward input:
        tokens: Long tensor shaped [B, T].
        return_attn: If True, return attention probabilities.

    Forward output:
        If return_attn=False:
            logits: [B, V], logits for the answer value token.
        If return_attn=True:
            logits: [B, V]
            all_probs: [L, B, H, T, T]
            mask: [T, T]

    Example:
        >>> tokens, targets, vocab = make_kv_retrieval_batch(4, 3, 10, 10)
        >>> model = TinyKVModel(vocab, tokens.shape[1], d_model=32, n_heads=4, n_layers=1)
        >>> model(tokens).shape
        torch.Size([4, 21])
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

        # Add token embedding and learned absolute position embedding.
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

        # The task asks for one answer token, so only the final sequence
        # position is classified.
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

    Inputs:
        batch_size: Number of examples B.
        num_pairs: Number of key/value pairs in each sequence.
        num_keys: Number of possible key tokens.
        num_values: Number of possible value tokens.
        device: Optional output device.

    Outputs:
        tokens: Long tensor shaped [B, 2 * num_pairs + 2].
            Sequence layout: k1, v1, k2, v2, ..., query_marker, query_key.
        target: Long tensor shaped [B]. The value paired with query_key.
        vocab_size: Integer vocabulary size needed by TinyKVModel.

    Example:
        >>> tokens, target, vocab = make_kv_retrieval_batch(2, 3, 10, 10)
        >>> tokens.shape, target.shape, vocab
        (torch.Size([2, 8]), torch.Size([2]), 21)

    Notes:
        Keys are sampled without replacement inside each example to avoid
        ambiguous duplicate keys.
    """
    if num_keys < num_pairs:
        raise ValueError("num_keys must be >= num_pairs")

    # Pick unique keys by taking top-k random scores per example.
    key_scores = torch.rand(batch_size, num_keys, device=device)
    keys = key_scores.topk(num_pairs, dim=-1).indices

    # Value token ids live after the key-token range: [num_keys, num_keys+num_values).
    values = torch.randint(0, num_values, (batch_size, num_pairs), device=device) + num_keys

    # Pick which pair will be queried in each example.
    query_index = torch.randint(0, num_pairs, (batch_size,), device=device)
    row = torch.arange(batch_size, device=device)
    query_key = keys[row, query_index]
    target = values[row, query_index]

    # Reserve one token id after all keys/values as a query marker.
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
    """Compute answer accuracy for key-value retrieval.

    Inputs:
        model: TinyKVModel-like model returning logits shaped [B, V].
        tokens: Long tensor shaped [B, T].
        targets: Long tensor shaped [B].

    Output:
        Python float in [0, 1].

    Example:
        >>> tokens, targets, vocab = make_kv_retrieval_batch(4, 3, 10, 10)
        >>> model = TinyKVModel(vocab, tokens.shape[1], d_model=32, n_heads=4, n_layers=1)
        >>> isinstance(retrieval_accuracy(model, tokens, targets), float)
        True
    """
    model.eval()
    pred = model(tokens).argmax(dim=-1)
    return (pred == targets).float().mean().item()
