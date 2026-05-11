# Jupyter 断点与变量查看指南

可以。当前 `llm-intro` 环境使用的 `ipykernel` 支持 Jupyter 调试。是否能用“图形化断点”取决于你打开的是 JupyterLab 还是经典 Notebook。

## 1. 最推荐：JupyterLab 图形化 Debugger

启动：

```bash
./start_llm_intro_jupyter.sh
```

在 notebook 里选择 kernel：

```text
Python (llm-intro)
```

如果 JupyterLab 顶部或右侧有小虫子/Debugger 图标：

1. 打开 Debugger。
2. 在代码单元左侧行号区域点击，设置断点。
3. 运行代码单元。
4. 右侧变量面板可以看局部变量、全局变量、调用栈。
5. 用 Step Over / Step Into / Continue 单步执行。

## 2. 通用方法：在代码里写 `breakpoint()`

任何 Jupyter 前端都可以用：

```python
for step in range(160):
    if step == 0:
        breakpoint()
    model.train()
```

运行到 `breakpoint()` 时会进入 Python 调试器。常用命令：

```text
n        下一行，step over
s        进入函数，step into
c        继续执行到下一个断点
l        显示附近代码
p x      打印变量 x
pp x     pretty print 变量 x
q        退出调试
```

例如在 pre-train 循环里查看张量：

```text
p input_ids.shape
p labels.shape
p loss.item()
p input_ids[0, :20]
```

## 3. 异常后调试：`%debug`

如果某个单元报错，下一格运行：

```python
%debug
```

它会进入报错位置的调试上下文。适合检查：

```text
p input_ids.shape
p logits.shape
p labels.unique()
```

还可以自动在异常后进入调试：

```python
%pdb on
```

关闭：

```python
%pdb off
```

## 4. 快速查看变量

不进断点时，常用这些：

```python
%whos
```

查看当前 notebook 里的变量清单。

```python
input_ids.shape, input_ids.dtype, input_ids.device
```

查看张量形状、数据类型和设备。

```python
loss.item()
```

把 0 维 tensor 转成 Python 数字。

```python
print(model)
```

查看模型结构。

```python
sum(p.numel() for p in model.parameters())
```

查看参数量。

## 5. 调试训练循环的建议

训练循环里不要每一步都断点，容易卡住。建议使用条件断点：

```python
if step in [0, 1, 10]:
    breakpoint()
```

或者只检查异常值：

```python
if not torch.isfinite(loss):
    breakpoint()
```

调试张量时优先看摘要，而不是直接打印整个 tensor：

```python
print(input_ids.shape)
print(logits.mean().item(), logits.std().item())
print(loss.item())
```

对于 LLM 训练，最常看的变量是：

- `input_ids.shape`
- `labels.shape`
- `logits.shape`
- `loss.item()`
- `attention mask`
- `tokenizer.decode(input_ids[0])`
- `labels == -100` 的位置
- optimizer 的学习率

