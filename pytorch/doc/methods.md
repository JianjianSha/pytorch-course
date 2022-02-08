# 常用方法

## argmax
```python
torch.argmax(input) -> LongTensor
```
返回 `input` 中最大值的索引，注意：返回值是一个具有单个值的 Tensor，其值为 Tensor 展开为一维数组后的最大值的索引，例如输入 Tensor 的 shape 为 $(a,b,c,d)$，那么返回值所表示的索引位于 $[0, a\cdot b\cdot c\cdot d)$ 之间 。

```{note}
如果有最大值不唯一，那么返回第一个最大值的索引
```

```python
a = torch.randn(2,2,2,2)
torch.argmax(a)
```

argmax 还有另一种形式，

```python
torch.argmax(input, dim, keepdim=False) -> LongTensor
```

`dim`：int 类型，可选。如为 `None`，则为 input 展开后最大值的索引；如不为 `None`，则在 `dim` 指定维度上求最大值索引，索引范围则位于 `[0, input.size(dim)）`。

`keepdim`：bool 类型。是否保持返回 Tensor 与 input Tensor 的维度相同。如果 `keepdim=False`，那么 return Tensor 比 input Tensor 少一个由 `dim` 指定的维度，否则维度相同。如果 `dim=None`，那么忽略 `keepdim`。

```python
a = torch.randn(2,2,2,2)
torch.argmax(a, dim=2)
```

