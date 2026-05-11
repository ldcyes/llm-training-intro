# Attention 指标 Notebook 逐行解释

对应文件：`attention_metrics_tutorial.ipynb`

这份说明按代码单元解释每一步在评估 attention 设计时的作用。

## Cell 1：导入依赖与初始化

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `import math` | 导入数学函数；本 notebook 中主要供公式实验扩展使用。 |
| 2 | `import torch` | PyTorch 主包。 |
| 3 | `import torch.nn.functional as F` | 函数式 API，用于交叉熵等训练 loss。 |
| 4 | `import matplotlib.pyplot as plt` | 画图工具。 |
| 6 | `from attention_metrics_torch import (` | 从本地工具脚本导入 attention 指标函数。 |
| 7 | `make_causal_mask` | 构造 full causal attention mask。 |
| 8 | `make_sliding_window_mask` | 构造局部窗口 causal mask。 |
| 9 | `receptive_field_matrix` | 计算多层后理论感受野。 |
| 10 | `graph_diameter` | 计算信息流图直径。 |
| 11 | `qk_logit_stats` | 统计 QK logits 的均值、方差、p99 等。 |
| 12 | `attention_probs` | 计算 masked attention softmax。 |
| 13 | `normalized_attention_entropy` | 计算归一化 attention entropy。 |
| 14 | `batch_effective_rank` | 批量计算 effective rank。 |
| 15 | `head_diversity` | 估算不同 attention head 的差异。 |
| 16 | `estimate_kv_cache_bytes` | 估算 KV cache 显存。 |
| 17 | `benchmark_forward` | 简单 benchmark forward 延迟和吞吐。 |
| 18 | `TinyKVModel` | 用于 key-value retrieval 的 tiny Transformer。 |
| 19 | `make_kv_retrieval_batch` | 生成 synthetic retrieval 数据。 |
| 20 | `retrieval_accuracy` | 计算 retrieval 准确率。 |
| 23 | `torch.manual_seed(42)` | 固定随机种子。 |
| 24 | `device = ...` | 有 CUDA 用 GPU，否则用 CPU。 |
| 25 | `device` | 在 notebook 中显示当前设备。 |

## Cell 3：信息流、感受野和图直径

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `seq_len = 64` | 设置序列长度。 |
| 2 | `layers_list = [1, 2, 4, 8, 16]` | 要比较的层数。 |
| 3 | `window = 8` | sliding-window attention 的窗口大小。 |
| 5 | `masks = {` | 构造要比较的 attention mask 字典。 |
| 6 | `'full': make_causal_mask(seq_len)` | full causal attention：每个 token 可看所有历史 token。 |
| 7 | `sliding_window...` | 局部窗口 attention：只能看最近 `window` 个 token。 |
| 10 | `for name, mask in masks.items():` | 遍历两种 mask。 |
| 11 | `d = graph_diameter(...)` | 计算图直径和不可达 token 对数量。 |
| 12 | `print(...)` | 输出 diameter 和 unreachable pairs。 |
| 14 | `plt.figure(...)` | 创建图像画布。 |
| 15 | `for name, mask in masks.items():` | 再次遍历两种 mask，准备画感受野增长。 |
| 16 | `sizes = []` | 保存不同层数下的感受野大小。 |
| 17 | `for layers in layers_list:` | 遍历层数。 |
| 18 | `rf = receptive_field_matrix(...)` | 计算该层数下哪些输入 token 能影响输出 token。 |
| 19 | `sizes.append(rf[-1].sum().item())` | 取最后一个 token 的感受野大小。 |
| 20 | `plt.plot(...)` | 画出层数 vs 感受野大小。 |
| 21 | `plt.xlabel('layers')` | 横轴是层数。 |
| 22 | `plt.ylabel(...)` | 纵轴是最后 token 的感受野大小。 |
| 23 | `plt.title(...)` | 图标题。 |
| 24 | `plt.grid(True)` | 显示网格。 |
| 25 | `plt.legend()` | 显示 full/sliding 标签。 |
| 26 | `plt.show()` | 显示图。 |

## Cell 6：QK logits、entropy、rank、head diversity

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `B, H, T, D = 4, 8, 128, 64` | batch、head 数、序列长度、head dim。 |
| 2 | `q = torch.randn(...)` | 随机生成 query tensor。 |
| 3 | `k = torch.randn(...)` | 随机生成 key tensor。 |
| 5 | `for name, mask in {...}.items():` | 比较 full 和 sliding mask。 |
| 6 | `'full': make_causal_mask(T)` | full causal mask。 |
| 7 | `'sliding_32': make_sliding_window_mask(T, 32)` | 窗口为 32 的局部 mask。 |
| 9 | `stats = qk_logit_stats(...)` | 统计 QK logits 分布。 |
| 10 | `probs = attention_probs(...)` | 计算 attention probability。 |
| 11 | `ent = normalized_attention_entropy(...)` | 计算归一化 entropy。 |
| 12 | `ranks = batch_effective_rank(probs[0])` | 对第一个 batch 的每个 head 计算 effective rank。 |
| 13 | `div = head_diversity(...)` | 计算 head pattern 差异。 |
| 14 | `print('\n', name)` | 打印当前 mask 名称。 |
| 15 | `print('logits:', stats)` | 输出 logits 统计。 |
| 16 | `print('entropy mean/std:', ...)` | 输出 entropy 均值和标准差。 |
| 17 | `print('effective rank mean:', ...)` | 输出平均 effective rank。 |
| 18 | `print('head diversity:', div)` | 输出 head diversity。 |

## Cell 9：KV cache 显存估算

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `layers = 32` | 模型层数。 |
| 2 | `batch_size = 8` | 推理 batch size。 |
| 3 | `seq_len = 8192` | 上下文长度。 |
| 4 | `n_heads = 32` | query head 数；这里用于说明，不直接参与计算。 |
| 5 | `head_dim = 128` | 每个 head 的维度。 |
| 6 | `dtype_bytes = 2` | bf16/fp16 每个数 2 bytes。 |
| 8 | `for label, n_kv_heads in ...` | 比较 MHA/GQA/MQA 的 KV head 数。 |
| 9 | `gb = estimate_kv_cache_bytes(...) / 1024**3` | 估算 KV cache 并转成 GiB。 |
| 10 | `print(label, ...)` | 输出每种 attention 的 KV cache 成本。 |

## Cell 10：Forward benchmark

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `tokens, targets, vocab_size = make_kv_retrieval_batch(` | 生成一批 synthetic key-value retrieval 数据。 |
| 2 | `batch_size=32` | batch 大小。 |
| 3 | `num_pairs=16` | 每条样本包含 16 个 key/value 对。 |
| 4 | `num_keys=128` | key token 的候选数量。 |
| 5 | `num_values=128` | value token 的候选数量。 |
| 6 | `device=device` | 数据放到 CPU/GPU。 |
| 9 | `model = TinyKVModel(` | 创建 tiny retrieval 模型。 |
| 10 | `vocab_size=vocab_size` | 词表大小来自数据生成器。 |
| 11 | `max_seq_len=tokens.shape[1]` | 模型最大长度等于当前输入长度。 |
| 12 | `d_model=128` | hidden size。 |
| 13 | `n_heads=4` | attention head 数。 |
| 14 | `n_layers=2` | Transformer block 数。 |
| 15 | `attn_kind='full'` | 使用 full causal attention。 |
| 16 | `).to(device)` | 把模型移动到设备。 |
| 18 | `result = benchmark_forward(...)` | 跑 forward benchmark，含 warmup 和多次计时。 |
| 19 | `result` | 显示 median 延迟、p95 延迟和 tokens/sec。 |

## Cell 12：Key-value retrieval 小任务训练

| 行 | 代码 | 解释 |
|---|---|---|
| 1 | `def train_retrieval(...):` | 定义训练函数，可切换 full/sliding attention。 |
| 2 | `torch.manual_seed(123)` | 固定该实验随机种子。 |
| 3 | `sample_tokens, _, vocab_size = ...` | 生成样本以确定 seq_len 和 vocab_size。 |
| 4 | `model = TinyKVModel(` | 创建 tiny retrieval 模型。 |
| 5 | `vocab_size=vocab_size` | 设置词表大小。 |
| 6 | `max_seq_len=sample_tokens.shape[1]` | 设置最大输入长度。 |
| 7 | `d_model=128` | hidden size。 |
| 8 | `n_heads=4` | head 数。 |
| 9 | `n_layers=2` | 层数。 |
| 10 | `attn_kind=attn_kind` | 使用传入的 attention 类型。 |
| 11 | `window=window` | sliding attention 的窗口参数。 |
| 12 | `).to(device)` | 模型移动到设备。 |
| 13 | `opt = torch.optim.AdamW(...)` | 创建优化器。 |
| 14 | `losses = []` | 保存训练 loss。 |
| 15 | `for step in range(steps):` | 训练指定步数。 |
| 16 | `model.train()` | 切换训练模式。 |
| 17 | `tokens, targets, _ = ...` | 每步生成新的 retrieval batch。 |
| 18 | `logits = model(tokens)` | 前向传播，输出最后位置的 logits。 |
| 19 | `loss = F.cross_entropy(logits, targets)` | 监督目标是 query key 对应的 value。 |
| 20 | `opt.zero_grad(set_to_none=True)` | 清空梯度。 |
| 21 | `loss.backward()` | 反向传播。 |
| 22 | `clip_grad_norm_(..., 1.0)` | 梯度裁剪。 |
| 23 | `opt.step()` | 更新参数。 |
| 24 | `losses.append(loss.item())` | 记录 loss。 |
| 25 | `if (step + 1) % 100 == 0:` | 每 100 步验证一次。 |
| 26 | `val_tokens, val_targets, _ = ...` | 生成验证 batch。 |
| 27 | `acc = retrieval_accuracy(...)` | 计算 retrieval accuracy。 |
| 28 | `print(...)` | 打印 attention 类型、步数、loss、acc。 |
| 29 | `return model, losses` | 返回训练后的模型和 loss 曲线。 |
| 31 | `full_model, full_losses = ...` | 训练 full attention 模型。 |
| 32 | `local_model, local_losses = ...` | 训练 sliding-window attention 模型。 |
| 34 | `plt.figure(...)` | 创建画布。 |
| 35 | `plt.plot(full_losses, ...)` | 画 full attention loss。 |
| 36 | `plt.plot(local_losses, ...)` | 画 sliding attention loss。 |
| 37 | `plt.xlabel('step')` | 横轴是 step。 |
| 38 | `plt.ylabel('cross entropy')` | 纵轴是交叉熵。 |
| 39 | `plt.title(...)` | 图标题。 |
| 40 | `plt.grid(True)` | 显示网格。 |
| 41 | `plt.legend()` | 显示图例。 |
| 42 | `plt.show()` | 显示图。 |

