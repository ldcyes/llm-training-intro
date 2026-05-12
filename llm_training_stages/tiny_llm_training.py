"""Tiny LLM training utilities used by the teaching notebook.

This file intentionally avoids external datasets and pretrained models. It is
small enough to debug line by line in Jupyter, while still showing the same
interfaces used by larger LLM training code:

- pre-train: input_ids -> next-token labels
- SFT: prompt tokens are masked with IGNORE_INDEX, assistant tokens are trained
- DPO: chosen/rejected responses are scored with policy and reference models

Shape notation used below:
- B = batch size
- T = sequence length
- V = vocabulary size
- D = hidden size / d_model
- H = number of attention heads
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


def set_seed(seed: int = 42) -> None:
    """Make random sampling more reproducible.

    Inputs:
        seed: Integer seed used by Python's `random` module and PyTorch.

    Outputs:
        None. This function changes global random-number-generator state.

    Example:
        >>> set_seed(7)
        >>> torch.randint(0, 10, (3,))
        tensor([5, 2, 1])
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharTokenizer:
    """A tiny character tokenizer for local educational experiments.

    Inputs:
        texts: Iterable of strings used to build the vocabulary.

    Outputs / attributes:
        stoi: Character/string token -> integer id.
        itos: Integer id -> character/string token.
        pad_id, bos_id, eos_id, unk_id: Special token ids.

    Example:
        >>> tok = CharTokenizer(["abc", "你好"])
        >>> ids = tok.encode("ab", add_bos=True, add_eos=True)
        >>> ids
        [1, 4, 5, 2]
        >>> tok.decode(ids)
        'ab'

    Notes:
        This is character-level, not BPE/SentencePiece. It is useful for
        teaching because every step is transparent, but real LLMs use subword
        tokenizers for better compression and generalization.
    """

    def __init__(self, texts: Iterable[str]):
        special = ["<pad>", "<bos>", "<eos>", "<unk>"]
        chars = sorted(set("".join(texts)))
        self.itos = special + [ch for ch in chars if ch not in special]
        self.stoi = {ch: idx for idx, ch in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        """Return number of tokens in the vocabulary, including special tokens."""
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert text to token ids.

        Inputs:
            text: Raw string.
            add_bos: Add beginning-of-sequence token at the front.
            add_eos: Add end-of-sequence token at the end.

        Output:
            A Python list of integer token ids.

        Example:
            >>> tok = CharTokenizer(["ab"])
            >>> tok.encode("ab", add_bos=True)
            [1, 4, 5]
        """
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Convert token ids back to text.

        Inputs:
            ids: Sequence of integer token ids.

        Output:
            Decoded string. `<pad>`, `<bos>`, and `<eos>` are skipped.
            `<unk>` is not skipped, so unknown characters remain visible.
        """
        skip = {self.pad_id, self.bos_id, self.eos_id}
        return "".join(self.itos[int(i)] for i in ids if int(i) not in skip)


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm variant used by many modern LLMs.

    Inputs:
        dim: Hidden dimension D.
        eps: Small constant to avoid division by zero.

    Forward input:
        x: Tensor shaped [..., D].

    Forward output:
        Tensor with the same shape as x.

    Example:
        >>> norm = RMSNorm(128)
        >>> y = norm(torch.randn(2, 16, 128))
        >>> y.shape
        torch.Size([2, 16, 128])
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize each token vector by its root-mean-square magnitude.
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class CausalSelfAttention(nn.Module):
    """Minimal multi-head masked self-attention.

    Inputs:
        d_model: Hidden size D.
        n_heads: Number of attention heads H. D must be divisible by H.
        dropout: Dropout probability applied to attention probabilities.

    Forward input:
        x: Float tensor shaped [B, T, D].

    Forward output:
        Tensor shaped [B, T, D].

    Example:
        >>> attn = CausalSelfAttention(d_model=128, n_heads=4)
        >>> y = attn(torch.randn(2, 32, 128))
        >>> y.shape
        torch.Size([2, 32, 128])

    Teaching point:
        The causal mask prevents token t from seeing tokens after t. This is
        what makes next-token prediction valid.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        # Project one [B, T, D] tensor into Q, K, V at once:
        # qkv shape after view: [B, T, 3, H, head_dim].
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Move heads before sequence length:
        # [B, T, H, head_dim] -> [B, H, T, head_dim].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scores[b, h, t, s] is how much token t wants to read token s.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Lower-triangular causal mask: row t can attend only columns <= t.
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # Weighted sum of values, then merge heads back to [B, T, D].
        y = probs @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out(y)


class MLP(nn.Module):
    """Feed-forward network inside a Transformer block.

    Inputs:
        d_model: Hidden size D.
        expansion: Hidden expansion ratio. expansion=4 gives hidden size 4D.
        dropout: Dropout after the second linear projection.

    Forward input/output:
        x: [B, T, D] -> [B, T, D].

    Example:
        >>> mlp = MLP(d_model=128)
        >>> mlp(torch.randn(2, 16, 128)).shape
        torch.Size([2, 16, 128])
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = expansion * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One decoder-only Transformer block.

    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))

    Forward input/output:
        x: [B, T, D] -> [B, T, D].

    Teaching point:
        This is "Pre-Norm" style. Norm happens before each sublayer, which is
        easier to train in deep LLMs than the original Post-Norm layout.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path keeps the old representation and adds the attention update.
        x = x + self.attn(self.norm1(x))

        # Second residual path adds the nonlinear MLP update.
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    """Small decoder-only GPT used for pre-train, SFT, and DPO demos.

    Inputs:
        vocab_size: Number of tokenizer ids V.
        max_seq_len: Maximum supported sequence length T.
        d_model: Hidden size D.
        n_heads: Attention heads H.
        n_layers: Number of Transformer blocks.
        dropout: Dropout probability.

    Forward input:
        input_ids: Long tensor shaped [B, T].
        labels: Optional long tensor shaped [B, T]. Positions set to
            IGNORE_INDEX are ignored by the loss.

    Forward output:
        logits: Float tensor shaped [B, T, V].
        loss: Scalar tensor if labels is provided, otherwise None.

    Example:
        >>> model = TinyGPT(vocab_size=100, max_seq_len=32, d_model=64, n_heads=4, n_layers=1)
        >>> input_ids = torch.randint(0, 100, (2, 32))
        >>> logits, loss = model(input_ids, labels=input_ids)
        >>> logits.shape
        torch.Size([2, 32, 100])
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

        # Weight tying: input embedding and output lm_head share parameters.
        # This is common in language models and reduces parameter count.
        self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize linear and embedding weights with a small normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        # positions shape is [1, T]; broadcasting adds the same position ids
        # to every item in the batch.
        positions = torch.arange(seq_len, device=input_ids.device)[None, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = causal_lm_loss(logits, labels)
        return logits, loss


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross entropy for causal language modeling.

    Inputs:
        logits: Float tensor shaped [B, T, V].
        labels: Long tensor shaped [B, T]. Use IGNORE_INDEX for positions that
            should not contribute to the loss.

    Output:
        Scalar cross-entropy loss.

    Example:
        >>> logits = torch.randn(2, 5, 10)
        >>> labels = torch.randint(0, 10, (2, 5))
        >>> loss = causal_lm_loss(logits, labels)
        >>> loss.ndim
        0

    Teaching point:
        The model at position t predicts labels at position t+1. That is why
        logits[:, :-1] is compared with labels[:, 1:].
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def pad_2d(sequences: Sequence[Sequence[int]], pad_id: int, device: torch.device) -> torch.Tensor:
    """Pad a list of variable-length integer sequences into a tensor.

    Inputs:
        sequences: List/tuple of token-id sequences, e.g. [[1, 2], [3]].
        pad_id: Value used to fill shorter rows.
        device: CPU/GPU device for the output tensor.

    Output:
        Long tensor shaped [B, max_length].

    Example:
        >>> pad_2d([[1, 2, 3], [4]], pad_id=0, device=torch.device("cpu"))
        tensor([[1, 2, 3],
                [4, 0, 0]])
    """
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    for row, seq in enumerate(sequences):
        out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return out


def sample_pretrain_batch(
    token_ids: Sequence[int],
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a pre-training batch from one long token stream.

    Inputs:
        token_ids: Long Python sequence representing the whole toy corpus.
        batch_size: Number of random chunks to sample.
        block_size: Number of tokens per chunk.
        device: CPU/GPU device for tensors.

    Outputs:
        input_ids: Long tensor shaped [B, block_size].
        labels: Long tensor shaped [B, block_size]. For pre-train this is a
            clone of input_ids; `causal_lm_loss` handles the one-token shift.

    Example:
        >>> ids = list(range(100))
        >>> x, y = sample_pretrain_batch(ids, batch_size=4, block_size=16, device=torch.device("cpu"))
        >>> x.shape, y.shape
        (torch.Size([4, 16]), torch.Size([4, 16]))
    """
    max_start = len(token_ids) - block_size - 1
    if max_start < 1:
        raise ValueError("token_ids is too short for the requested block_size")
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack(
        [torch.tensor(token_ids[i : i + block_size], dtype=torch.long) for i in starts]
    ).to(device)
    labels = x.clone()
    return x, labels


def make_sft_sequence(
    tokenizer: CharTokenizer,
    user: str,
    assistant: str,
    max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    """Build one SFT training sequence with user tokens masked out.

    Inputs:
        tokenizer: CharTokenizer instance.
        user: User instruction/question text.
        assistant: Desired assistant answer.
        max_seq_len: Truncate sequence to this length.

    Outputs:
        ids: Token ids for "<bos>User: ...\\nAssistant: answer<eos>".
        labels: Same length as ids. Prompt/user positions are IGNORE_INDEX;
            assistant answer positions are real token ids.

    Example:
        >>> tok = CharTokenizer(["<bos>User: hi\\nAssistant: hello<eos>"])
        >>> ids, labels = make_sft_sequence(tok, "hi", "hello", max_seq_len=64)
        >>> len(ids) == len(labels)
        True

    Teaching point:
        Masking prompt tokens prevents the model from being trained to imitate
        the user's question. It trains only the assistant response.
    """
    prompt = f"<bos>User: {user}\nAssistant: "
    answer = f"{assistant}<eos>"
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    ids = (prompt_ids + answer_ids)[:max_seq_len]
    labels = [IGNORE_INDEX] * min(len(prompt_ids), len(ids))
    labels.extend(ids[len(prompt_ids) :])
    labels = labels[: len(ids)]
    return ids, labels


def make_sft_batch(
    tokenizer: CharTokenizer,
    examples: Sequence[Tuple[str, str]],
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a padded SFT mini-batch.

    Inputs:
        examples: Sequence of (user, assistant) pairs.
        batch_size: Number of examples sampled with replacement.
        max_seq_len: Truncation length.
        device: CPU/GPU device.

    Outputs:
        input_ids: Long tensor shaped [B, T].
        label_ids: Long tensor shaped [B, T], with prompt/padding positions set
            to IGNORE_INDEX.
    """
    sampled = random.choices(examples, k=batch_size)
    ids, labels = zip(*(make_sft_sequence(tokenizer, u, a, max_seq_len) for u, a in sampled))
    input_ids = pad_2d(ids, tokenizer.pad_id, device)
    label_ids = pad_2d(labels, IGNORE_INDEX, device)
    return input_ids, label_ids


def make_response_sequence(
    tokenizer: CharTokenizer,
    prompt: str,
    response: str,
    max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    """Build one prompt+response sequence for DPO scoring.

    Inputs:
        prompt: User prompt shared by chosen and rejected responses.
        response: Either a chosen or rejected assistant response.
        max_seq_len: Truncation length.

    Outputs:
        ids: Token ids for the whole conversation.
        labels: Prompt tokens are IGNORE_INDEX; response tokens are scored.

    Example:
        >>> tok = CharTokenizer(["<bos>User: q\\nAssistant: a<eos>"])
        >>> ids, labels = make_response_sequence(tok, "q", "a", 64)
        >>> labels.count(IGNORE_INDEX) > 0
        True
    """
    prefix = f"<bos>User: {prompt}\nAssistant: "
    suffix = f"{response}<eos>"
    prefix_ids = tokenizer.encode(prefix)
    suffix_ids = tokenizer.encode(suffix)
    ids = (prefix_ids + suffix_ids)[:max_seq_len]
    labels = [IGNORE_INDEX] * min(len(prefix_ids), len(ids))
    labels.extend(ids[len(prefix_ids) :])
    labels = labels[: len(ids)]
    return ids, labels


def make_dpo_batch(
    tokenizer: CharTokenizer,
    preferences: Sequence[Tuple[str, str, str]],
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create a padded DPO batch from preference triples.

    Inputs:
        preferences: Sequence of (prompt, chosen_response, rejected_response).
        batch_size: Number of triples sampled with replacement.
        max_seq_len: Truncation length for each prompt+response sequence.
        device: CPU/GPU device.

    Output:
        Dictionary with four tensors:
            chosen_input_ids: [B, T_chosen]
            chosen_labels: [B, T_chosen]
            rejected_input_ids: [B, T_rejected]
            rejected_labels: [B, T_rejected]

    Example:
        >>> prefs = [("q", "good answer", "bad answer")]
        >>> tok = CharTokenizer(["<bos>User: q\\nAssistant: good answer bad answer<eos>"])
        >>> batch = make_dpo_batch(tok, prefs, 2, 64, torch.device("cpu"))
        >>> sorted(batch)
        ['chosen_input_ids', 'chosen_labels', 'rejected_input_ids', 'rejected_labels']
    """
    sampled = random.choices(preferences, k=batch_size)
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = [], [], [], []
    for prompt, chosen, rejected in sampled:
        c_ids, c_labels = make_response_sequence(tokenizer, prompt, chosen, max_seq_len)
        r_ids, r_labels = make_response_sequence(tokenizer, prompt, rejected, max_seq_len)
        chosen_ids.append(c_ids)
        chosen_labels.append(c_labels)
        rejected_ids.append(r_ids)
        rejected_labels.append(r_labels)
    return {
        "chosen_input_ids": pad_2d(chosen_ids, tokenizer.pad_id, device),
        "chosen_labels": pad_2d(chosen_labels, IGNORE_INDEX, device),
        "rejected_input_ids": pad_2d(rejected_ids, tokenizer.pad_id, device),
        "rejected_labels": pad_2d(rejected_labels, IGNORE_INDEX, device),
    }


def sequence_logprob(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute summed log-probability of supervised response tokens.

    Inputs:
        model: TinyGPT-like model returning logits shaped [B, T, V].
        input_ids: Long tensor shaped [B, T].
        labels: Long tensor shaped [B, T]. IGNORE_INDEX positions are skipped.

    Output:
        Float tensor shaped [B]. Each value is the sum of log-probabilities for
        the non-ignored target tokens in one sequence.

    Example:
        >>> tok = CharTokenizer(["<bos>User: q\\nAssistant: a<eos>"])
        >>> model = TinyGPT(tok.vocab_size, max_seq_len=32, d_model=32, n_heads=4, n_layers=1)
        >>> ids, labels = make_response_sequence(tok, "q", "a", 32)
        >>> x = pad_2d([ids], tok.pad_id, torch.device("cpu"))
        >>> y = pad_2d([labels], IGNORE_INDEX, torch.device("cpu"))
        >>> sequence_logprob(model, x, y).shape
        torch.Size([1])
    """
    logits, _ = model(input_ids)

    # Shift because logits at position t predict the label at position t+1.
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = labels[:, 1:].clone()
    mask = target.ne(IGNORE_INDEX)

    # `gather` needs a valid index everywhere, so ignored positions are
    # temporarily set to 0 and then removed by multiplying with mask.
    target = target.masked_fill(~mask, 0)
    token_logp = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(
    policy: nn.Module,
    reference: nn.Module,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Direct Preference Optimization loss.

    Inputs:
        policy: Trainable model pi.
        reference: Frozen reference model ref, usually copied from SFT model.
        batch: Output of make_dpo_batch.
        beta: Preference strength. Higher beta pushes harder away from the
            reference model.

    Outputs:
        loss: Scalar tensor used for backprop.
        stats: Python dict with monitoring values:
            dpo_loss, preference_accuracy, policy_margin, reference_margin.

    Example:
        >>> prefs = build_preference_examples()
        >>> tok = CharTokenizer(collect_all_texts())
        >>> policy = TinyGPT(tok.vocab_size, 64, d_model=32, n_heads=4, n_layers=1)
        >>> reference = TinyGPT(tok.vocab_size, 64, d_model=32, n_heads=4, n_layers=1)
        >>> batch = make_dpo_batch(tok, prefs, 2, 64, torch.device("cpu"))
        >>> loss, stats = dpo_loss(policy, reference, batch)
        >>> loss.ndim, "preference_accuracy" in stats
        (0, True)

    Formula:
        -log sigmoid(beta * ((log pi_chosen - log pi_rejected)
                          - (log ref_chosen - log ref_rejected)))
    """
    policy_chosen = sequence_logprob(policy, batch["chosen_input_ids"], batch["chosen_labels"])
    policy_rejected = sequence_logprob(policy, batch["rejected_input_ids"], batch["rejected_labels"])
    with torch.no_grad():
        ref_chosen = sequence_logprob(reference, batch["chosen_input_ids"], batch["chosen_labels"])
        ref_rejected = sequence_logprob(reference, batch["rejected_input_ids"], batch["rejected_labels"])

    # Positive margin means the model assigns higher probability to chosen.
    policy_margin = policy_chosen - policy_rejected
    reference_margin = ref_chosen - ref_rejected

    # Compare policy preference against reference preference. This keeps the
    # optimized model near the reference instead of blindly maximizing chosen.
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()
    stats = {
        "dpo_loss": loss.item(),
        "preference_accuracy": (logits > 0).float().mean().item(),
        "policy_margin": policy_margin.mean().item(),
        "reference_margin": reference_margin.mean().item(),
    }
    return loss, stats


@torch.no_grad()
def generate(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    device: torch.device | None = None,
) -> str:
    """Generate text autoregressively from a prompt.

    Inputs:
        model: TinyGPT model.
        tokenizer: CharTokenizer used by the model.
        prompt: Text prompt. Usually already includes chat template text.
        max_new_tokens: Maximum number of tokens to sample.
        temperature: Sampling temperature. Lower is more deterministic.
        device: Optional device override. Defaults to model's device.

    Output:
        Decoded string containing prompt plus generated continuation.

    Example:
        >>> tok = CharTokenizer(collect_all_texts())
        >>> model = TinyGPT(tok.vocab_size, 64, d_model=32, n_heads=4, n_layers=1)
        >>> text = generate(model, tok, "<bos>User: hi\\nAssistant: ", max_new_tokens=5)
        >>> isinstance(text, str)
        True
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        # Keep only the last max_seq_len tokens because the model has a fixed
        # position-embedding table.
        x_cond = x[:, -model.max_seq_len :]
        logits, _ = model(x_cond)

        # Use the final position to sample the next token.
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if int(next_id.item()) == tokenizer.eos_id:
            break
    return tokenizer.decode(x[0].tolist())


def build_toy_corpus() -> str:
    """Return repeated toy text for pre-training.

    Output:
        One long string. The notebook tokenizes it and samples random chunks.

    Example:
        >>> corpus = build_toy_corpus()
        >>> "Pre-train" in corpus
        True
    """
    passages = [
        "语言模型通过预测下一个 token 学习文本分布。",
        "Pre-train 使用大量原始文本，目标是 next-token prediction。",
        "SFT 使用指令和答案，让模型学会按照人类期望的格式回答。",
        "DPO 使用 chosen 和 rejected 响应，让策略模型相对参考模型更偏好 chosen。",
        "Attention 让当前 token 读取历史 token 的信息。",
        "Residual connection 和 RMSNorm 可以改善深层 Transformer 的训练稳定性。",
        "学习率、batch size、数据质量和 tokenizer 都会影响训练结果。",
        "评估模型不能只看训练 loss，还要看验证集、偏好准确率和生成样例。",
        "A small model is useful for understanding the training loop before scaling up.",
        "Post-RL training aligns a model with preferences, rewards, or verifiable outcomes.",
    ]
    return "\n".join(passages * 80)


def build_sft_examples() -> List[Tuple[str, str]]:
    """Return toy SFT examples.

    Output:
        List of (user_prompt, assistant_answer) pairs.

    Example:
        >>> examples = build_sft_examples()
        >>> examples[0][0]
        '什么是 pre-train？'
    """
    return [
        ("什么是 pre-train？", "Pre-train 是用大规模文本做 next-token prediction，让模型学习语言、知识和通用模式。"),
        ("什么是 SFT？", "SFT 是 supervised fine-tuning，用高质量指令和答案教模型按人类期望回答。"),
        ("什么是 DPO？", "DPO 是一种偏好优化方法，使用 chosen/rejected 响应直接训练模型偏好更好的答案。"),
        ("为什么要只训练 assistant token？", "因为 user token 是条件输入，不是模型应该模仿生成的目标。"),
        ("pre-train 和 SFT 的区别？", "Pre-train 学文本分布，SFT 学任务格式和指令跟随。"),
        ("post-RL training 解决什么？", "它进一步对齐模型行为，让输出更符合偏好、奖励或可验证目标。"),
    ]


def build_preference_examples() -> List[Tuple[str, str, str]]:
    """Return toy preference examples for DPO.

    Output:
        List of (prompt, chosen_response, rejected_response) triples.

    Example:
        >>> prompt, chosen, rejected = build_preference_examples()[0]
        >>> len((prompt, chosen, rejected))
        3
    """
    return [
        (
            "什么是 SFT？",
            "SFT 是用指令-答案数据做监督微调，让模型学会按用户请求回答。",
            "SFT 是一种神秘技巧，随便给模型一些文本就会变聪明。",
        ),
        (
            "为什么 pre-train 后还要 SFT？",
            "因为 pre-train 主要学习文本分布，SFT 让模型学会对话格式和指令跟随。",
            "因为 pre-train 没有任何作用，所以必须重新训练。",
        ),
        (
            "DPO 需要什么数据？",
            "DPO 需要 prompt、chosen response 和 rejected response 组成的偏好对。",
            "DPO 只需要无标注网页文本，不需要偏好数据。",
        ),
        (
            "post-RL training 的目标是什么？",
            "目标是让模型输出更符合人类偏好、规则奖励或可验证任务目标。",
            "目标是让模型忘掉 pre-train 学到的所有知识。",
        ),
        (
            "如何判断 DPO 是否有效？",
            "可以看 chosen/rejected logprob margin、preference accuracy 和人工评测。",
            "只要训练步数变多，就一定说明 DPO 有效。",
        ),
    ]


def collect_all_texts() -> List[str]:
    """Collect every text fragment used to build the character vocabulary.

    Output:
        List of strings including pre-training corpus, SFT examples, and DPO
        chosen/rejected responses.

    Example:
        >>> texts = collect_all_texts()
        >>> len(texts) > 1
        True
    """
    corpus = [build_toy_corpus()]
    for user, assistant in build_sft_examples():
        corpus.append(f"<bos>User: {user}\nAssistant: {assistant}<eos>")
    for prompt, chosen, rejected in build_preference_examples():
        corpus.append(f"<bos>User: {prompt}\nAssistant: {chosen}<eos>")
        corpus.append(f"<bos>User: {prompt}\nAssistant: {rejected}<eos>")
    return corpus
