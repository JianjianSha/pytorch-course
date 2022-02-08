# 池化

## MaxPool2d
```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```
最大值池化。

参数 `kernel_size`，`stride`，`padding` 和 `dilation` 可以是单个 int 值，或者两个 int 组成的 tuple（分表表示 H,W 方向），各参数含义可参考 `conv` 中的说明 。

`return_indices`：如果设为 `True`，那么除了返回 output，还返回 output 值在 input 中对应的 indices 。

`ceil_mode`：如为 `True`，那么使用 ceil 而非 floor 来计算 output 的 shape 。

input shape：$(N,C,H_{in},W_{in})$

output shape：$(N,C,H_{out},W_{out})$

当 `ceil_mode=False` 时，

$$H_{out}=\lfloor \frac {H_{in} + 2 p_H - d_H(k_H-1)-1}{s_H} +1\rfloor$$  (pool1)

$$W_{out}=\lfloor \frac {W_{in} + 2 p_W - d_W(k_W-1)-1}{s_W} +1\rfloor$$ (pool2)

其中，$p, d, s$ 分别表示 `padding, dilation, stride`。

膨胀后 kernel 大小为 $d \cdot (k-1)+1$ 。

注意：边缘填充的像素值为 $-\infty$ 。

## MaxUnpool2d

```python
torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
```
计算 `MaxPool2d` 的部分逆向值，这是因为 `MaxPool2d` 损失了部分数据，所以说是 “部分” 。

`MaxUnpool2d` 的输入为 `MaxPool2d` 的输出，包括最大值对应的 indices，对于非最大值，则直接设置为 `0`。


input shape：$(N,C,H_{in},W_{in})$

output shape: $(N,C,H_{out},W_{out})$

$$H_{out}=(H_{in}-1) \cdot s_H-2p_H+k_H$$ (pool3)

$$W_{out}=(W_{in}-1) \cdot s_W-2p_W+k_W$$ (pool4)

上两式几乎是 {eq}`pool1` 和 {eq}`pool2` 的反函数，只是这里 kernel 没有膨胀一说。

output shape 还可以由 `forward` 函数的参数给定，如果是这样，那么最大值所在的 indices 按如下规则确定：将 Tensor 展开成一维数组，然后确定在一维数组中的位置，最后再还原为指定的 output shape 。

例：

```python
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)

# shape: (1,1,4,4)
input = torch.tensor([[[[1,2,3,4],
                        [5,6,7,8],
                        [9,10,11,12],
                        [13,14,15,16]]]])

output, indices = pool(input)
unpool(output, indices)

unpool(output, indices, output_size=torch.Size([1,1,5,3]))
```

## AvgPool2d
```python
torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
```

均值池化。

参数与 `MaxPool2d` 的差不多。其中不一样的是，`count_include_pad` 为 True 时，计算均值时，将 zero-padding 的像素数量计入在内。`divisor_override` 如果指定，那么将取代 池化 窗口中的像素数量，作为均值计算的 分母。kernel 没有膨胀。

使用示例：

```python
m = nn.AvgPool2d(3, stride=2)
input = torch.randn(1,2,30,40)
output = m(input)
```