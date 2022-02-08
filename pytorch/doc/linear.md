# 线性操作
线性层其实就是全连接层 （full connection FC）。通常卷积层的最后会接上一个或若干个 FC 层，以使得输出单元数符号任务期望。

```python
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

数学表达式为 $\mathbf y = \mathbf x A^{\top} + \mathbf b$ 。


input shape：$(\star, H_{in})$，`*` 表示任意数量，$H_{in}$ 表示 FC 层的输入特征数。

output shape：$(*, H_{out})$，$H_{out}$ 表示 FC 层输出特征数，`*` 与 input shape 中的相同。

例如，input 为 $(N,C,H,W)$ 为上一个卷积层的输出，到达 FC 层时，通常 view 成 $(N, C*H*W)$ ，然后经 FC 变换为 $(N, H_{out})$ 。

权重参数的 shape 为 $(H_{out}, H_{in})$，初始化为均匀分布 $\mathcal U(-\sqrt k, \sqrt k)$ ，其中 $k=1/H_{in}$

如果 `bias=True`，那么还有偏差参数，其 shape 为 $(H_{out},)$ ，初始化为均匀分布 $\mathcal U(-\sqrt k, \sqrt k)$ ，其中 $k=1/H_{in}$ 。

使用示例：
```python
m = nn.Linear(256, 30)
input = torch.randn(2,4,8,8)
input = input.view(2, 4*8*8)
output = m(input)
print(output.size())
```