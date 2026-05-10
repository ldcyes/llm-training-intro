from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/mnt/c/Users/Administrator/Desktop/模型训练")
OUT = ROOT / "printable_tutorials"


def register_fonts() -> Tuple[str, str, str]:
    cjk = Path("/mnt/c/Windows/Fonts/simhei.ttf")
    cjk_bold = Path("/mnt/c/Windows/Fonts/msyhbd.ttc")
    mono = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    latin = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    latin_bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

    if cjk.exists():
        pdfmetrics.registerFont(TTFont("DocCJK", str(cjk)))
    else:
        pdfmetrics.registerFont(TTFont("DocCJK", str(latin)))

    if cjk_bold.exists():
        # TTC fonts work on this ReportLab version; if not, fall back silently.
        try:
            pdfmetrics.registerFont(TTFont("DocCJKBold", str(cjk_bold)))
        except Exception:
            pdfmetrics.registerFont(TTFont("DocCJKBold", str(cjk if cjk.exists() else latin_bold)))
    else:
        pdfmetrics.registerFont(TTFont("DocCJKBold", str(cjk if cjk.exists() else latin_bold)))

    pdfmetrics.registerFont(TTFont("DocMono", str(mono)))
    pdfmetrics.registerFont(TTFont("DocLatin", str(latin)))
    pdfmetrics.registerFont(TTFont("DocLatinBold", str(latin_bold)))
    return "DocCJK", "DocCJKBold", "DocMono"


FONT, FONT_BOLD, FONT_MONO = register_fonts()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=26,
            leading=32,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#374151"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=24,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor("#111827"),
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=14,
            leading=19,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1F2937"),
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.5,
            leading=14.5,
            spaceAfter=5,
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.2,
            leading=11.5,
            spaceAfter=4,
            textColor=colors.HexColor("#374151"),
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.2,
            leading=13.5,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontName=FONT_MONO,
            fontSize=7.8,
            leading=10.5,
            leftIndent=6,
            rightIndent=6,
            backColor=colors.HexColor("#F3F4F6"),
            borderColor=colors.HexColor("#E5E7EB"),
            borderWidth=0.25,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8,
            leading=10.5,
        ),
        "table_bold": ParagraphStyle(
            "table_bold",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=8,
            leading=10.5,
        ),
    }


S = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), S["code"])


def bullets(items: Sequence[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, "bullet"), bulletColor=colors.HexColor("#374151")) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=14,
    )


def table(rows: Sequence[Sequence[str]], widths: Sequence[float]) -> Table:
    data = []
    for row_idx, row in enumerate(rows):
        style = "table_bold" if row_idx == 0 else "table"
        data.append([p(cell, style) for cell in row])
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


class PipelineDiagram(Flowable):
    def __init__(self, labels: Sequence[str], width: float = 16.6 * cm, height: float = 2.0 * cm):
        super().__init__()
        self.labels = labels
        self.width = width
        self.height = height

    def wrap(self, avail_width: float, avail_height: float):
        return min(self.width, avail_width), self.height

    def draw(self):
        c = self.canv
        w = self.width
        h = self.height
        n = len(self.labels)
        gap = 0.32 * cm
        box_w = (w - gap * (n - 1)) / n
        box_h = 0.9 * cm
        y = h / 2 - box_h / 2
        for i, label in enumerate(self.labels):
            x = i * (box_w + gap)
            c.setFillColor(colors.HexColor("#EFF6FF"))
            c.setStrokeColor(colors.HexColor("#2563EB"))
            c.roundRect(x, y, box_w, box_h, 5, fill=1, stroke=1)
            c.setFillColor(colors.HexColor("#111827"))
            c.setFont(FONT_BOLD, 7.5)
            lines = label.split("\n")
            for j, line in enumerate(lines):
                c.drawCentredString(x + box_w / 2, y + box_h / 2 + (len(lines) - 1) * 4 - j * 8 - 2, line)
            if i < n - 1:
                x1 = x + box_w
                x2 = x + box_w + gap
                mid_y = y + box_h / 2
                c.setStrokeColor(colors.HexColor("#6B7280"))
                c.line(x1 + 2, mid_y, x2 - 7, mid_y)
                c.setFillColor(colors.HexColor("#6B7280"))
                c.line(x2 - 7, mid_y + 3, x2 - 2, mid_y)
                c.line(x2 - 7, mid_y - 3, x2 - 2, mid_y)


def cover(title: str, subtitle: str, author: str, version: str) -> List:
    return [
        Spacer(1, 3.4 * cm),
        p(title, "title"),
        p(subtitle, "subtitle"),
        Spacer(1, 0.8 * cm),
        p(f"Author / 作者: {author}", "subtitle"),
        p(version, "subtitle"),
        Spacer(1, 1.4 * cm),
        PipelineDiagram(["Pre-train", "SFT", "Post-RL\nDPO / GRPO", "Attention\nMetrics", "Practice\nNotebook"]),
        PageBreak(),
    ]


ZH_SECTIONS = [
    ("1. 学习目标", [
        ("p", "这份讲义把两个入门 notebook 合并成一份可打印材料：LLM 三阶段训练，以及 attention 设计指标。目标是先用小模型和可解释实验建立直觉，再迁移到真实模型、真实数据和分布式训练。"),
        ("bullets", [
            "理解 pre-train、SFT、post-RL training 的目标、数据格式和 loss。",
            "理解 embedding、attention、RoPE、residual、RMSNorm/LayerNorm、MLP 为什么存在。",
            "掌握不完整训练 LLM 时评估 attention 设计的代理指标。",
            "能用本地 conda 环境运行 PyTorch 和 Jupyter 入门实验。"
        ]),
    ]),
    ("2. 三阶段训练总览", [
        ("diagram", ["Raw Text\nCorpus", "Pre-train\nNext Token", "Base\nModel", "SFT\nInstruction", "Post-RL\nPreference"]),
        ("p", "Pre-train 教模型学习文本分布；SFT 教模型按照指令格式回答；post-RL training 用偏好、奖励或验证器进一步对齐输出行为。"),
        ("table", [
            ["阶段", "数据", "核心 loss / 方法", "输出模型"],
            ["Pre-train", "大规模原始文本", "causal LM next-token cross entropy", "Base model"],
            ["SFT", "指令、问题、标准答案", "assistant token masked cross entropy", "Instruction model"],
            ["DPO", "prompt + chosen + rejected", "relative preference objective with reference model", "Preference-aligned model"],
            ["GRPO/RLHF", "prompt + reward/verifier 或偏好数据", "policy optimization / reward model", "Aligned model"]
        ], [2.7*cm, 4.0*cm, 6.1*cm, 3.8*cm]),
    ]),
    ("3. Pre-train", [
        ("p", "Pre-train 的基本目标是预测下一个 token。输入和标签几乎相同，只是在 loss 里 shift 一个位置。"),
        ("code", """
input:  x0 x1 x2 ... xT
target:    x1 x2 ... xT
loss:   CE(model(x_<=t), x_{t+1})
"""),
        ("p", "真实 pre-train 的难点通常不在公式，而在数据工程、tokenizer、吞吐、稳定性、checkpoint、验证集污染和规模化训练。入门时建议从 100M 以内 tiny GPT 或 0.5B 级 continued pretraining 开始。"),
    ]),
    ("4. SFT", [
        ("p", "SFT 仍然是 causal LM loss，但只应该监督 assistant 需要生成的 tokens。user prompt 是条件输入，不是模仿目标。"),
        ("code", """
input:  <bos>User: 什么是 SFT?\\nAssistant: SFT 是...
labels: -100 -100 -100 ...          SFT 是...
"""),
        ("bullets", [
            "chat template 必须和模型/tokenizer 匹配。",
            "长回答要注意截断策略，否则 assistant 监督信号会被截掉。",
            "高质量小数据通常胜过低质量大数据。",
            "评估不能只看训练 loss，还要看格式遵循、事实性、拒答、任务集和人工样例。"
        ]),
    ]),
    ("5. Post-RL Training：DPO、RLHF、GRPO", [
        ("p", "Post-RL training 的目的不是重新教语言，而是让模型输出更符合偏好、规则、奖励或可验证目标。DPO 是最适合入门的 post-RL 方法，因为它不需要单独训练 reward model。"),
        ("code", """
L_DPO = -log sigmoid(beta * ((log pi_chosen - log pi_rejected)
                           - (log ref_chosen - log ref_rejected)))
"""),
        ("table", [
            ["方法", "需要的数据", "适合场景"],
            ["DPO", "chosen/rejected 偏好对", "SFT 后的偏好对齐入门"],
            ["Reward Model + PPO", "偏好对 + RL 采样", "传统 RLHF 管线"],
            ["GRPO", "prompt + 可验证 reward function", "数学、代码、规则可验证任务"],
            ["RLAIF", "AI 反馈或规则生成偏好", "人类标注不足时的扩展方案"],
        ], [3.0*cm, 5.6*cm, 7.4*cm]),
    ]),
    ("6. Transformer 核心结构", [
        ("diagram", ["Token IDs", "Embedding", "Attention\n+ RoPE", "Residual\n+ Norm", "MLP", "LM Head"]),
        ("table", [
            ["结构", "解决的问题", "去掉后的常见问题", "改进方向"],
            ["Embedding", "离散 token 到连续向量", "token id 没有语义距离", "tokenizer、weight tying、domain vocabulary"],
            ["Attention", "上下文信息路由", "长距离依赖弱", "FlashAttention、GQA/MQA、sparse/window attention"],
            ["RoPE/位置编码", "注入顺序和相对距离", "顺序、括号、代码位置能力差", "YaRN、LongRoPE、ALiBi、xPos"],
            ["Residual", "深层信息和梯度通路", "深层训练不稳定", "residual scaling、DeepNorm、gated residual"],
            ["RMSNorm/LayerNorm", "稳定激活分布", "loss spike、梯度爆炸/漂移", "Pre-Norm、QK-Norm、RMSNorm"],
            ["MLP/SwiGLU", "非线性容量和知识存储", "表达力不足", "SwiGLU、MoE、扩展 ratio"],
        ], [2.5*cm, 4.0*cm, 4.4*cm, 5.2*cm]),
    ]),
    ("7. Attention 设计代理指标", [
        ("p", "不完整训练 LLM 时，可以用 proxy 指标提前筛掉风险设计。单个指标不能替代最终 loss，但组合后很有价值。"),
        ("table", [
            ["维度", "指标", "主要预测"],
            ["信息流", "graph diameter, receptive field, rollout mass", "长距离信息是否可传递"],
            ["稳定性", "QK logits std/p99/p99.9, normalized entropy", "attention 是否饱和或训崩"],
            ["表达力", "effective rank, head diversity, ablation", "attention 是否退化或 head 浪费"],
            ["位置能力", "retrieval vs distance, extrapolation ratio", "长上下文和长度外推"],
            ["硬件", "tokens/sec, TTFT, TPOT, MFU, KV cache", "实际训练和部署可用性"],
            ["小任务", "copy, key-value retrieval, multi-hop", "结构能力下限"],
        ], [2.3*cm, 7.2*cm, 6.5*cm]),
    ]),
    ("8. 关键公式", [
        ("code", """
Receptive field:
R_0(t) = {t}
R_l(t) = union over s in Attend_l(t) of R_{l-1}(s)

Attention logits:
z_{t,s} = q_t · k_s / sqrt(d_head)
p_{t,s} = softmax(z_{t,s})

Normalized entropy:
H_t = - sum_s p_{t,s} log p_{t,s}
H_norm = H_t / log(N_t)

Effective rank:
p_i = sigma_i / sum_j sigma_j
r_eff = exp(- sum_i p_i log p_i)

KV cache:
KV bytes = layers * batch * seq_len * 2 * n_kv_heads * head_dim * dtype_bytes
"""),
    ]),
    ("9. 推荐实验协议", [
        ("bullets", [
            "先静态计算 FLOPs、KV cache、参数量。",
            "随机输入下统计 QK logits、entropy、effective rank、head diversity。",
            "短训 1k-5k steps，观察 loss spike、NaN/Inf、logits 漂移。",
            "跑 copy、key-value retrieval、multi-hop 小任务。",
            "做 retrieval vs distance，观察 lost-in-the-middle 和长度外推。",
            "最后再做小规模 LLM pre-train loss 和真实任务评估。"
        ]),
    ]),
    ("10. 本地运行", [
        ("p", "当前工作目录已经包含 conda 环境文件、Jupyter 启动脚本和两个 notebook。"),
        ("code", """
cd /mnt/c/Users/Administrator/Desktop/模型训练
./verify_llm_intro_env.sh
./start_llm_intro_jupyter.sh
"""),
        ("table", [
            ["文件", "用途"],
            ["llm_training_stages/llm_pretrain_sft_postrl_tutorial.ipynb", "Pre-train、SFT、DPO 教学实验"],
            ["attention_design_metrics/attention_metrics_tutorial.ipynb", "Attention 设计指标教学实验"],
            ["llm_training_stages/tiny_llm_training.py", "tiny GPT、SFT batch、DPO loss"],
            ["attention_design_metrics/attention_metrics_torch.py", "attention 指标工具函数"],
            ["environment.yml", "conda 环境定义"],
        ], [6.8*cm, 9.2*cm]),
    ]),
]


EN_SECTIONS = [
    ("1. Learning Goals", [
        ("p", "This handout merges two introductory notebooks into one printable guide: LLM training stages and attention-design metrics. The goal is to build intuition with small, inspectable experiments before scaling to real models, real datasets, and distributed training."),
        ("bullets", [
            "Understand the goal, data format, and loss for pre-training, SFT, and post-RL training.",
            "Understand why embedding, attention, RoPE, residual paths, RMSNorm/LayerNorm, and MLP blocks exist.",
            "Use proxy metrics to evaluate attention designs before running full LLM training.",
            "Run the local PyTorch and Jupyter intro experiments from the provided conda environment."
        ]),
    ]),
    ("2. Three Training Stages", [
        ("diagram", ["Raw Text\nCorpus", "Pre-train\nNext Token", "Base\nModel", "SFT\nInstruction", "Post-RL\nPreference"]),
        ("p", "Pre-training teaches the model a text distribution. SFT teaches the model to follow instructions and answer in the desired format. Post-RL training further aligns behavior with preferences, rewards, or verifiable objectives."),
        ("table", [
            ["Stage", "Data", "Core loss / method", "Output model"],
            ["Pre-train", "Large raw text corpora", "causal LM next-token cross entropy", "Base model"],
            ["SFT", "Instructions, questions, reference answers", "assistant-token masked cross entropy", "Instruction model"],
            ["DPO", "prompt + chosen + rejected", "relative preference objective with reference model", "Preference-aligned model"],
            ["GRPO/RLHF", "prompts + rewards/verifiers or preferences", "policy optimization / reward model", "Aligned model"]
        ], [2.7*cm, 4.0*cm, 6.1*cm, 3.8*cm]),
    ]),
    ("3. Pre-training", [
        ("p", "The basic objective is next-token prediction. Inputs and labels are almost identical, but the loss is shifted by one token."),
        ("code", """
input:  x0 x1 x2 ... xT
target:    x1 x2 ... xT
loss:   CE(model(x_<=t), x_{t+1})
"""),
        ("p", "In real pre-training, the hard parts are usually not the formula. The hard parts are data quality, tokenization, throughput, stability, checkpointing, contamination control, and scale-out training. For learning, start with a tiny GPT below 100M parameters or a 0.5B-class continued pre-training run."),
    ]),
    ("4. Supervised Fine-Tuning", [
        ("p", "SFT is still causal LM training, but labels should supervise only the assistant tokens. The user prompt is conditioning context, not a target to imitate."),
        ("code", """
input:  <bos>User: What is SFT?\\nAssistant: SFT is...
labels: -100 -100 -100 ...          SFT is...
"""),
        ("bullets", [
            "The chat template must match the model and tokenizer.",
            "Watch truncation: it can remove the assistant tokens that carry the training signal.",
            "A small high-quality dataset often beats a large noisy one.",
            "Do not evaluate only training loss; also check format following, factuality, refusals, task suites, and generated samples."
        ]),
    ]),
    ("5. Post-RL Training: DPO, RLHF, GRPO", [
        ("p", "Post-RL training is not meant to relearn language from scratch. It aligns output behavior with preferences, rules, rewards, or verifiable targets. DPO is the best entry point because it does not require a separate reward model."),
        ("code", """
L_DPO = -log sigmoid(beta * ((log pi_chosen - log pi_rejected)
                           - (log ref_chosen - log ref_rejected)))
"""),
        ("table", [
            ["Method", "Required data", "Best use case"],
            ["DPO", "chosen/rejected preference pairs", "Intro alignment after SFT"],
            ["Reward Model + PPO", "preference pairs + RL sampling", "Classic RLHF pipeline"],
            ["GRPO", "prompts + verifiable reward function", "Math, code, and rule-verifiable tasks"],
            ["RLAIF", "AI feedback or rule-generated preferences", "When human labels are scarce"],
        ], [3.0*cm, 5.6*cm, 7.4*cm]),
    ]),
    ("6. Transformer Core Structures", [
        ("diagram", ["Token IDs", "Embedding", "Attention\n+ RoPE", "Residual\n+ Norm", "MLP", "LM Head"]),
        ("table", [
            ["Structure", "Problem solved", "Common issue if removed", "Improvement directions"],
            ["Embedding", "Map discrete tokens to vectors", "Token IDs carry no semantic distance", "tokenizer, weight tying, domain vocabulary"],
            ["Attention", "Route context information", "Weak long-range dependency handling", "FlashAttention, GQA/MQA, sparse/window attention"],
            ["RoPE/position encoding", "Inject order and relative distance", "Poor sequence order, brackets, and code position handling", "YaRN, LongRoPE, ALiBi, xPos"],
            ["Residual path", "Information and gradient highway", "Deep training becomes unstable", "residual scaling, DeepNorm, gated residual"],
            ["RMSNorm/LayerNorm", "Stabilize activation distribution", "loss spikes, exploding gradients, drift", "Pre-Norm, QK-Norm, RMSNorm"],
            ["MLP/SwiGLU", "Nonlinear capacity and knowledge storage", "Insufficient expressivity", "SwiGLU, MoE, larger expansion ratio"],
        ], [2.5*cm, 4.0*cm, 4.4*cm, 5.2*cm]),
    ]),
    ("7. Proxy Metrics for Attention Design", [
        ("p", "Before running full LLM training, proxy metrics can eliminate risky designs. No single metric replaces final loss, but the combination is useful."),
        ("table", [
            ["Dimension", "Metrics", "Main prediction"],
            ["Information flow", "graph diameter, receptive field, rollout mass", "Whether long-range information can travel"],
            ["Stability", "QK logits std/p99/p99.9, normalized entropy", "Whether attention saturates or training breaks"],
            ["Expressivity", "effective rank, head diversity, ablation", "Whether attention collapses or heads are wasted"],
            ["Positional ability", "retrieval vs distance, extrapolation ratio", "Long-context and length extrapolation"],
            ["Hardware", "tokens/sec, TTFT, TPOT, MFU, KV cache", "Practical training and serving viability"],
            ["Small tasks", "copy, key-value retrieval, multi-hop", "Lower bound of structural capability"],
        ], [2.7*cm, 7.0*cm, 6.3*cm]),
    ]),
    ("8. Key Formulas", [
        ("code", """
Receptive field:
R_0(t) = {t}
R_l(t) = union over s in Attend_l(t) of R_{l-1}(s)

Attention logits:
z_{t,s} = q_t · k_s / sqrt(d_head)
p_{t,s} = softmax(z_{t,s})

Normalized entropy:
H_t = - sum_s p_{t,s} log p_{t,s}
H_norm = H_t / log(N_t)

Effective rank:
p_i = sigma_i / sum_j sigma_j
r_eff = exp(- sum_i p_i log p_i)

KV cache:
KV bytes = layers * batch * seq_len * 2 * n_kv_heads * head_dim * dtype_bytes
"""),
    ]),
    ("9. Recommended Experiment Protocol", [
        ("bullets", [
            "First compute FLOPs, KV cache, and parameter count statically.",
            "On random inputs, inspect QK logits, entropy, effective rank, and head diversity.",
            "Train for 1k-5k steps and watch for loss spikes, NaN/Inf, and logit drift.",
            "Run copy, key-value retrieval, and multi-hop synthetic tasks.",
            "Measure retrieval vs distance, including lost-in-the-middle and length extrapolation.",
            "Only then run small-scale LLM pre-training loss and real task evaluation."
        ]),
    ]),
    ("10. Local Execution", [
        ("p", "The working directory already contains the conda environment file, Jupyter startup script, and both notebooks."),
        ("code", """
cd /mnt/c/Users/Administrator/Desktop/模型训练
./verify_llm_intro_env.sh
./start_llm_intro_jupyter.sh
"""),
        ("table", [
            ["File", "Purpose"],
            ["llm_training_stages/llm_pretrain_sft_postrl_tutorial.ipynb", "Pre-train, SFT, and DPO teaching experiment"],
            ["attention_design_metrics/attention_metrics_tutorial.ipynb", "Attention-design metric teaching experiment"],
            ["llm_training_stages/tiny_llm_training.py", "tiny GPT, SFT batch, DPO loss"],
            ["attention_design_metrics/attention_metrics_torch.py", "attention metric utilities"],
            ["environment.yml", "conda environment definition"],
        ], [6.8*cm, 9.2*cm]),
    ]),
]


def md_escape(text: str) -> str:
    return text


def write_markdown(path: Path, title: str, subtitle: str, author: str, sections: Sequence):
    lines = [f"# {title}", "", subtitle, "", f"**Author / 作者:** {author}", ""]
    for heading, blocks in sections:
        lines.extend([f"## {heading}", ""])
        for block in blocks:
            kind = block[0]
            payload = block[1]
            if kind == "p":
                lines.extend([payload, ""])
            elif kind == "bullets":
                for item in payload:
                    lines.append(f"- {item}")
                lines.append("")
            elif kind == "code":
                lines.extend(["```text", payload.strip("\n"), "```", ""])
            elif kind == "table":
                rows = payload
                header = rows[0]
                lines.append("| " + " | ".join(header) + " |")
                lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                for row in rows[1:]:
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")
            elif kind == "diagram":
                lines.append(" -> ".join(payload))
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_pdf(path: Path, title: str, subtitle: str, author: str, version: str, sections: Sequence, lang: str):
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.5 * cm,
        title=title,
        author=author,
        subject="LLM training stages and attention design metrics",
    )
    story: List = []
    story.extend(cover(title, subtitle, author, version))

    story.append(p("Contents" if lang == "en" else "目录", "h1"))
    for heading, _blocks in sections:
        story.append(p(heading, "body"))
    story.append(PageBreak())

    for heading, blocks in sections:
        story.append(p(heading, "h1"))
        for block in blocks:
            kind = block[0]
            payload = block[1]
            if kind == "p":
                story.append(p(payload, "body"))
            elif kind == "bullets":
                story.append(bullets(payload))
            elif kind == "code":
                story.append(code(payload))
            elif kind == "table":
                rows = payload
                widths = block[2] if len(block) > 2 else [4 * cm] * len(payload[0])
                story.append(table(rows, widths))
                story.append(Spacer(1, 0.2 * cm))
            elif kind == "diagram":
                story.append(PipelineDiagram(payload))
                story.append(Spacer(1, 0.2 * cm))
        story.append(Spacer(1, 0.15 * cm))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(1.8 * cm, 0.8 * cm, f"{title} | {author}")
        canvas.drawRightString(A4[0] - 1.8 * cm, 0.8 * cm, str(doc_obj.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    author = "liangdacheng / 梁达成"
    zh_title = "LLM 训练与 Attention 设计入门讲义"
    zh_subtitle = "Pre-train、SFT、Post-RL Training 与 Attention 指标"
    en_title = "Introductory Guide to LLM Training and Attention Design"
    en_subtitle = "Pre-training, SFT, Post-RL Training, and Attention Metrics"
    bilingual_title = "LLM Training and Attention Design Tutorial / LLM 训练与 Attention 设计入门讲义"
    bilingual_subtitle = "Chinese and English Printable Edition / 中英双语打印版"
    version = "Printable PDF version"

    build_pdf(
        OUT / "llm_attention_training_tutorial_zh.pdf",
        zh_title,
        zh_subtitle,
        author,
        version,
        ZH_SECTIONS,
        "zh",
    )
    build_pdf(
        OUT / "llm_attention_training_tutorial_en.pdf",
        en_title,
        en_subtitle,
        author,
        version,
        EN_SECTIONS,
        "en",
    )
    write_markdown(
        OUT / "llm_attention_training_tutorial_zh.md",
        zh_title,
        zh_subtitle,
        author,
        ZH_SECTIONS,
    )
    write_markdown(
        OUT / "llm_attention_training_tutorial_en.md",
        en_title,
        en_subtitle,
        author,
        EN_SECTIONS,
    )
    bilingual_sections = (
        [("中文版本 / Chinese Version", [("p", "以下为中文版本。")])]
        + ZH_SECTIONS
        + [("English Version / 英文版本", [("p", "The following section is the English version.")])]
        + EN_SECTIONS
    )
    build_pdf(
        OUT / "llm_attention_training_tutorial_bilingual.pdf",
        bilingual_title,
        bilingual_subtitle,
        author,
        version,
        bilingual_sections,
        "en",
    )
    write_markdown(
        OUT / "llm_attention_training_tutorial_bilingual.md",
        bilingual_title,
        bilingual_subtitle,
        author,
        bilingual_sections,
    )
    print("generated:")
    for name in [
        "llm_attention_training_tutorial_zh.pdf",
        "llm_attention_training_tutorial_en.pdf",
        "llm_attention_training_tutorial_bilingual.pdf",
        "llm_attention_training_tutorial_zh.md",
        "llm_attention_training_tutorial_en.md",
        "llm_attention_training_tutorial_bilingual.md",
    ]:
        path = OUT / name
        print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
