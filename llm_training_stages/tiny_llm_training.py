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
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharTokenizer:
    """A tiny character tokenizer for local educational experiments."""

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
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id}
        return "".join(self.itos[int(i)] for i in ids if int(i) not in skip)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class CausalSelfAttention(nn.Module):
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
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        y = probs @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out(y)


class MLP(nn.Module):
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
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
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
        self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
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
    """Next-token cross entropy. labels use IGNORE_INDEX for unsupervised positions."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def pad_2d(sequences: Sequence[Sequence[int]], pad_id: int, device: torch.device) -> torch.Tensor:
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
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = labels[:, 1:].clone()
    mask = target.ne(IGNORE_INDEX)
    target = target.masked_fill(~mask, 0)
    token_logp = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(
    policy: nn.Module,
    reference: nn.Module,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    policy_chosen = sequence_logprob(policy, batch["chosen_input_ids"], batch["chosen_labels"])
    policy_rejected = sequence_logprob(policy, batch["rejected_input_ids"], batch["rejected_labels"])
    with torch.no_grad():
        ref_chosen = sequence_logprob(reference, batch["chosen_input_ids"], batch["chosen_labels"])
        ref_rejected = sequence_logprob(reference, batch["rejected_input_ids"], batch["rejected_labels"])

    policy_margin = policy_chosen - policy_rejected
    reference_margin = ref_chosen - ref_rejected
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
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.max_seq_len :]
        logits, _ = model(x_cond)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if int(next_id.item()) == tokenizer.eos_id:
            break
    return tokenizer.decode(x[0].tolist())


def build_toy_corpus() -> str:
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
    return [
        ("什么是 pre-train？", "Pre-train 是用大规模文本做 next-token prediction，让模型学习语言、知识和通用模式。"),
        ("什么是 SFT？", "SFT 是 supervised fine-tuning，用高质量指令和答案教模型按人类期望回答。"),
        ("什么是 DPO？", "DPO 是一种偏好优化方法，使用 chosen/rejected 响应直接训练模型偏好更好的答案。"),
        ("为什么要只训练 assistant token？", "因为 user token 是条件输入，不是模型应该模仿生成的目标。"),
        ("pre-train 和 SFT 的区别？", "Pre-train 学文本分布，SFT 学任务格式和指令跟随。"),
        ("post-RL training 解决什么？", "它进一步对齐模型行为，让输出更符合偏好、奖励或可验证目标。"),
    ]


def build_preference_examples() -> List[Tuple[str, str, str]]:
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
    corpus = [build_toy_corpus()]
    for user, assistant in build_sft_examples():
        corpus.append(f"<bos>User: {user}\nAssistant: {assistant}<eos>")
    for prompt, chosen, rejected in build_preference_examples():
        corpus.append(f"<bos>User: {prompt}\nAssistant: {chosen}<eos>")
        corpus.append(f"<bos>User: {prompt}\nAssistant: {rejected}<eos>")
    return corpus
