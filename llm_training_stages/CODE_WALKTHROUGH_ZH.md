# LLM 三阶段 Notebook 逐行解释

对应文件：`llm_pretrain_sft_postrl_tutorial.ipynb`

这份说明按 notebook 的代码单元解释。重点看每一行在训练流程里的作用，而不是背 API。

## Cell 2：导入依赖与选择设备

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `import copy` | 用于深拷贝 SFT 后的模型，得到冻结的 reference model。 |
| 2 | `import time` | 用于记录 pre-train 循环耗时。 |
| 4 | `import torch` | 导入 PyTorch 主包，负责 tensor、模型、优化器和 CUDA。 |
| 5 | `import torch.nn.functional as F` | 导入函数式 API；当前 notebook 主要在配套文件里用交叉熵等函数。 |
| 6 | `import matplotlib.pyplot as plt` | 用于画 loss 曲线和 DPO 指标曲线。 |
| 8 | `from tiny_llm_training import (` | 从本地教学脚本导入 tiny GPT、数据构造和训练工具。 |
| 9 | `CharTokenizer` | 字符级 tokenizer，避免下载外部 tokenizer。 |
| 10 | `TinyGPT` | 教学用 decoder-only Transformer。 |
| 11 | `build_toy_corpus` | 构造 toy pre-train 文本。 |
| 12 | `build_sft_examples` | 构造 toy 指令微调样本。 |
| 13 | `build_preference_examples` | 构造 toy chosen/rejected 偏好样本。 |
| 14 | `collect_all_texts` | 收集所有文本，用于建立 tokenizer 词表。 |
| 15 | `dpo_loss` | 手写 DPO loss。 |
| 16 | `generate` | 用模型自回归生成文本。 |
| 17 | `make_dpo_batch` | 把偏好样本变成 DPO batch。 |
| 18 | `make_sft_batch` | 把 SFT 样本变成 input/label batch。 |
| 19 | `sample_pretrain_batch` | 从原始 token 序列采样 pre-train batch。 |
| 20 | `set_seed` | 固定随机种子，让实验更可复现。 |
| 23 | `set_seed(7)` | 设置 Python/PyTorch 随机种子。 |
| 24 | `device = ...` | 如果有 CUDA 就用 GPU，否则用 CPU。 |
| 25 | `device` | 在 notebook 中显示当前设备。 |

## Cell 4：构造 tokenizer、语料和模型

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `tokenizer = CharTokenizer(collect_all_texts())` | 用全部教学文本建立字符级词表。 |
| 2 | `corpus = build_toy_corpus()` | 生成 pre-train 阶段使用的小语料。 |
| 3 | `corpus_ids = tokenizer.encode(...)` | 把文本转成 token id，并加 `<bos>/<eos>`。 |
| 5 | `max_seq_len = 96` | 设置模型最大上下文长度。 |
| 6 | `model = TinyGPT(` | 开始实例化 tiny GPT。 |
| 7 | `vocab_size=tokenizer.vocab_size` | 词表大小必须等于 tokenizer 可输出的 token 数。 |
| 8 | `max_seq_len=max_seq_len` | 位置 embedding 的长度上限。 |
| 9 | `d_model=128` | hidden size，越大容量越强但更慢。 |
| 10 | `n_heads=4` | attention head 数。 |
| 11 | `n_layers=2` | Transformer block 层数。 |
| 12 | `dropout=0.0` | 教学实验关闭 dropout，便于观察。 |
| 13 | `).to(device)` | 把模型参数移动到 CPU 或 GPU。 |
| 15 | `num_params = ...` | 统计模型总参数量。 |
| 16 | `print('vocab_size =', ...)` | 打印词表大小。 |
| 17 | `print('corpus_tokens =', ...)` | 打印 toy 语料 token 数。 |
| 18 | `print('parameters =', ...)` | 打印模型参数量。 |

## Cell 6：Pre-train 训练循环

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `pretrain_optimizer = torch.optim.AdamW(...)` | 创建 AdamW 优化器，学习率 `3e-4`，带 weight decay。 |
| 2 | `pretrain_losses = []` | 存每一步的 pre-train loss，后面画图。 |
| 4 | `start = time.time()` | 记录开始时间。 |
| 5 | `for step in range(160):` | 训练 160 步，教学用小步数。 |
| 6 | `model.train()` | 切换到训练模式。 |
| 7 | `input_ids, labels = sample_pretrain_batch(...)` | 从语料随机抽 batch；pre-train 的 label 基本就是 input。 |
| 8 | `_, loss = model(input_ids, labels=labels)` | 前向传播并计算 next-token loss。 |
| 9 | `zero_grad(set_to_none=True)` | 清空上一步梯度，`set_to_none=True` 更省内存。 |
| 10 | `loss.backward()` | 反向传播，计算梯度。 |
| 11 | `clip_grad_norm_(..., 1.0)` | 梯度裁剪，避免训练不稳定。 |
| 12 | `pretrain_optimizer.step()` | 用梯度更新模型参数。 |
| 13 | `pretrain_losses.append(loss.item())` | 记录 Python 数字形式的 loss。 |
| 14 | `if (step + 1) % 40 == 0:` | 每 40 步打印一次。 |
| 15 | `print(...)` | 输出当前 step 和 loss。 |
| 17 | `print('seconds =', ...)` | 打印训练耗时。 |
| 18 | `print(generate(...))` | 用 pre-train 后模型生成样例，看它学到的文本分布。 |

## Cell 8：SFT 训练循环

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `sft_examples = build_sft_examples()` | 构造 toy 指令-答案数据。 |
| 2 | `sft_optimizer = torch.optim.AdamW(...)` | 新建 SFT 优化器，学习率比 pre-train 略小。 |
| 3 | `sft_losses = []` | 记录 SFT loss。 |
| 5 | `for step in range(180):` | 训练 180 步。 |
| 6 | `model.train()` | 切换到训练模式。 |
| 7 | `input_ids, labels = make_sft_batch(...)` | 构造 SFT batch；user prompt 的 label 被设为 `-100`。 |
| 8 | `_, loss = model(input_ids, labels=labels)` | 只在非 `-100` 的 assistant token 上计算 loss。 |
| 9 | `sft_optimizer.zero_grad(...)` | 清空梯度。 |
| 10 | `loss.backward()` | 反向传播。 |
| 11 | `clip_grad_norm_(..., 1.0)` | 梯度裁剪。 |
| 12 | `sft_optimizer.step()` | 更新模型。 |
| 13 | `sft_losses.append(loss.item())` | 记录 SFT loss。 |
| 14 | `if (step + 1) % 45 == 0:` | 每 45 步打印一次。 |
| 15 | `print(...)` | 输出当前 SFT loss。 |
| 17 | `prompt = ...` | 构造 chat 格式 prompt。 |
| 18 | `print(generate(...))` | 查看 SFT 后模型是否更像 assistant。 |

## Cell 10：冻结 reference model

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `reference_model = copy.deepcopy(model).to(device)` | 把 SFT 后模型复制一份，作为 DPO 的 reference。 |
| 2 | `reference_model.eval()` | reference 只用于打分，不训练。 |
| 3 | `for p in reference_model.parameters():` | 遍历 reference 的所有参数。 |
| 4 | `p.requires_grad_(False)` | 禁止计算 reference 梯度，节省显存并防止被更新。 |
| 6 | `policy_model = model` | 当前模型继续作为 DPO 要训练的 policy。 |

## Cell 12：DPO 训练循环

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `preferences = build_preference_examples()` | 构造 prompt/chosen/rejected 偏好对。 |
| 2 | `dpo_optimizer = torch.optim.AdamW(...)` | 创建 DPO 优化器，学习率更小。 |
| 3 | `dpo_losses = []` | 记录 DPO loss。 |
| 4 | `dpo_accs = []` | 记录偏好准确率。 |
| 5 | `policy_margins = []` | 记录 policy 对 chosen 和 rejected 的 logprob 差。 |
| 7 | `for step in range(120):` | DPO 训练 120 步。 |
| 8 | `policy_model.train()` | policy 切换到训练模式。 |
| 9 | `batch = make_dpo_batch(...)` | 构造 chosen/rejected 两组序列和 labels。 |
| 10 | `loss, stats = dpo_loss(...)` | 计算 DPO loss 和监控指标，`beta=0.2` 控制偏好强度。 |
| 11 | `dpo_optimizer.zero_grad(...)` | 清空梯度。 |
| 12 | `loss.backward()` | 反向传播，只更新 policy。 |
| 13 | `clip_grad_norm_(..., 1.0)` | 梯度裁剪。 |
| 14 | `dpo_optimizer.step()` | 更新 policy 参数。 |
| 15 | `dpo_losses.append(...)` | 记录 DPO loss。 |
| 16 | `dpo_accs.append(...)` | 记录 chosen 是否优于 rejected。 |
| 17 | `policy_margins.append(...)` | 记录 chosen/rejected margin。 |
| 18 | `if (step + 1) % 30 == 0:` | 每 30 步打印一次。 |
| 19 | `print('dpo step', ...)` | 打印 DPO 指标字典。 |
| 21 | `print(generate(...))` | 用 DPO 后模型生成答案样例。 |

## Cell 14：画训练曲线

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `fig, axes = plt.subplots(1, 3, ...)` | 创建一行三列的图。 |
| 3 | `axes[0].plot(pretrain_losses)` | 画 pre-train loss。 |
| 4 | `set_title(...)` | 设置第一张图标题。 |
| 5 | `set_xlabel('step')` | 设置横轴。 |
| 6 | `grid(True)` | 显示网格。 |
| 8 | `axes[1].plot(sft_losses)` | 画 SFT loss。 |
| 9 | `set_title(...)` | 设置第二张图标题。 |
| 10 | `set_xlabel('step')` | 设置横轴。 |
| 11 | `grid(True)` | 显示网格。 |
| 13 | `axes[2].plot(dpo_losses, ...)` | 画 DPO loss。 |
| 14 | `axes[2].plot(dpo_accs, ...)` | 画 preference accuracy。 |
| 15 | `axes[2].plot(policy_margins, ...)` | 画 policy margin。 |
| 16 | `set_title(...)` | 设置第三张图标题。 |
| 17 | `set_xlabel('step')` | 设置横轴。 |
| 18 | `grid(True)` | 显示网格。 |
| 19 | `legend()` | 显示曲线标签。 |
| 21 | `plt.tight_layout()` | 自动调整布局，避免文字重叠。 |
| 22 | `plt.show()` | 在 notebook 中显示图。 |

