# LLM 训练与 Attention 设计入门讲义

Pre-train、SFT、Post-RL Training 与 Attention 指标

**Author / 作者:** liangdacheng / 梁达成

## 1. 学习目标

这份讲义把两个入门 notebook 合并成一份可打印材料：LLM 三阶段训练，以及 attention 设计指标。目标是先用小模型和可解释实验建立直觉，再迁移到真实模型、真实数据和分布式训练。

- 理解 pre-train、SFT、post-RL training 的目标、数据格式和 loss。
- 理解 embedding、attention、RoPE、residual、RMSNorm/LayerNorm、MLP 为什么存在。
- 掌握不完整训练 LLM 时评估 attention 设计的代理指标。
- 能用本地 conda 环境运行 PyTorch 和 Jupyter 入门实验。

## 2. 三阶段训练总览

Raw Text
Corpus -> Pre-train
Next Token -> Base
Model -> SFT
Instruction -> Post-RL
Preference

Pre-train 教模型学习文本分布；SFT 教模型按照指令格式回答；post-RL training 用偏好、奖励或验证器进一步对齐输出行为。

| 阶段 | 数据 | 核心 loss / 方法 | 输出模型 |
| --- | --- | --- | --- |
| Pre-train | 大规模原始文本 | causal LM next-token cross entropy | Base model |
| SFT | 指令、问题、标准答案 | assistant token masked cross entropy | Instruction model |
| DPO | prompt + chosen + rejected | relative preference objective with reference model | Preference-aligned model |
| GRPO/RLHF | prompt + reward/verifier 或偏好数据 | policy optimization / reward model | Aligned model |

## 3. Pre-train

Pre-train 的基本目标是预测下一个 token。输入和标签几乎相同，只是在 loss 里 shift 一个位置。

```text
input:  x0 x1 x2 ... xT
target:    x1 x2 ... xT
loss:   CE(model(x_<=t), x_{t+1})
```

真实 pre-train 的难点通常不在公式，而在数据工程、tokenizer、吞吐、稳定性、checkpoint、验证集污染和规模化训练。入门时建议从 100M 以内 tiny GPT 或 0.5B 级 continued pretraining 开始。

## 4. SFT

SFT 仍然是 causal LM loss，但只应该监督 assistant 需要生成的 tokens。user prompt 是条件输入，不是模仿目标。

```text
input:  <bos>User: 什么是 SFT?\nAssistant: SFT 是...
labels: -100 -100 -100 ...          SFT 是...
```

- chat template 必须和模型/tokenizer 匹配。
- 长回答要注意截断策略，否则 assistant 监督信号会被截掉。
- 高质量小数据通常胜过低质量大数据。
- 评估不能只看训练 loss，还要看格式遵循、事实性、拒答、任务集和人工样例。

## 5. Post-RL Training：DPO、RLHF、GRPO

Post-RL training 的目的不是重新教语言，而是让模型输出更符合偏好、规则、奖励或可验证目标。DPO 是最适合入门的 post-RL 方法，因为它不需要单独训练 reward model。

```text
L_DPO = -log sigmoid(beta * ((log pi_chosen - log pi_rejected)
                           - (log ref_chosen - log ref_rejected)))
```

| 方法 | 需要的数据 | 适合场景 |
| --- | --- | --- |
| DPO | chosen/rejected 偏好对 | SFT 后的偏好对齐入门 |
| Reward Model + PPO | 偏好对 + RL 采样 | 传统 RLHF 管线 |
| GRPO | prompt + 可验证 reward function | 数学、代码、规则可验证任务 |
| RLAIF | AI 反馈或规则生成偏好 | 人类标注不足时的扩展方案 |

## 6. Transformer 核心结构

Token IDs -> Embedding -> Attention
+ RoPE -> Residual
+ Norm -> MLP -> LM Head

| 结构 | 解决的问题 | 去掉后的常见问题 | 改进方向 |
| --- | --- | --- | --- |
| Embedding | 离散 token 到连续向量 | token id 没有语义距离 | tokenizer、weight tying、domain vocabulary |
| Attention | 上下文信息路由 | 长距离依赖弱 | FlashAttention、GQA/MQA、sparse/window attention |
| RoPE/位置编码 | 注入顺序和相对距离 | 顺序、括号、代码位置能力差 | YaRN、LongRoPE、ALiBi、xPos |
| Residual | 深层信息和梯度通路 | 深层训练不稳定 | residual scaling、DeepNorm、gated residual |
| RMSNorm/LayerNorm | 稳定激活分布 | loss spike、梯度爆炸/漂移 | Pre-Norm、QK-Norm、RMSNorm |
| MLP/SwiGLU | 非线性容量和知识存储 | 表达力不足 | SwiGLU、MoE、扩展 ratio |

## 7. Attention 设计代理指标

不完整训练 LLM 时，可以用 proxy 指标提前筛掉风险设计。单个指标不能替代最终 loss，但组合后很有价值。

| 维度 | 指标 | 主要预测 |
| --- | --- | --- |
| 信息流 | graph diameter, receptive field, rollout mass | 长距离信息是否可传递 |
| 稳定性 | QK logits std/p99/p99.9, normalized entropy | attention 是否饱和或训崩 |
| 表达力 | effective rank, head diversity, ablation | attention 是否退化或 head 浪费 |
| 位置能力 | retrieval vs distance, extrapolation ratio | 长上下文和长度外推 |
| 硬件 | tokens/sec, TTFT, TPOT, MFU, KV cache | 实际训练和部署可用性 |
| 小任务 | copy, key-value retrieval, multi-hop | 结构能力下限 |

## 8. 关键公式

```text
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
```

## 9. 推荐实验协议

- 先静态计算 FLOPs、KV cache、参数量。
- 随机输入下统计 QK logits、entropy、effective rank、head diversity。
- 短训 1k-5k steps，观察 loss spike、NaN/Inf、logits 漂移。
- 跑 copy、key-value retrieval、multi-hop 小任务。
- 做 retrieval vs distance，观察 lost-in-the-middle 和长度外推。
- 最后再做小规模 LLM pre-train loss 和真实任务评估。

## 10. 本地运行

当前工作目录已经包含 conda 环境文件、Jupyter 启动脚本和两个 notebook。

```text
cd /mnt/c/Users/Administrator/Desktop/模型训练
./verify_llm_intro_env.sh
./start_llm_intro_jupyter.sh
```

| 文件 | 用途 |
| --- | --- |
| llm_training_stages/llm_pretrain_sft_postrl_tutorial.ipynb | Pre-train、SFT、DPO 教学实验 |
| attention_design_metrics/attention_metrics_tutorial.ipynb | Attention 设计指标教学实验 |
| llm_training_stages/tiny_llm_training.py | tiny GPT、SFT batch、DPO loss |
| attention_design_metrics/attention_metrics_torch.py | attention 指标工具函数 |
| environment.yml | conda 环境定义 |
