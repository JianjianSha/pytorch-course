# 转置卷积

也称为 fractionally-strided 卷积或者反卷积，实际上并不是卷积的逆操作，本质上还是一个卷积，输入输出的 spatial size 的计算是普通卷积 spatial size 计算的逆操作，关于这一点请看下文的 $H,W$ 的计算公式。

```python
ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, 
                bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
```



文章介绍：[PyTorch 方法总结](https://jianjiansha.github.io/2019/11/01/pytorch/PyTorch-mtd/)

shape：

1. input $(N, C, H, W)$
2. output $(N, C', H', W')$

$$H'=(H-1) \times s - 2 \times p + d \times (k-1) + p' + 1
\\W'=(W-1) \times s - 2 \times p + d \times (k-1) + p' + 1$$ (convt1)
其中 $s, p, d$ 分表表示 stride， padding 和 dilation，$p'$ 表示 out_padding。这两个等式可能不好理解，但是我们可以借助普通卷积来理解：

$$H'=\frac {H + 2 \times p - [d \times (k-1) + 1]} s + 1
\\
\\W'=\frac {W + 2 \times p - [d \times (k-1) + 1]} s + 1$$ (convt2)

根据 {eq}`convt2` 式可得 
$$p=\frac {s(H'-1)-H+[d(k-1)+1]} 2$$ (convt3)

显然，将 {eq}`convt2` 中的 $H, W$ 和 $H', W'$ 的位置对调，就变成了 {eq}`convt1` 式，只是没有 $p'$ 项，而 $p'$ 表示在输出 tensor 的每个 spatial dimension 上单边（one side）填充的大小，故在输出 size 上再加上 $p'$ 就得到了 {eq}`convt1` 式。

举例说明：
1. 考虑最简单的情况，$s=d=1$，
    
    $p=p'=0$，此时 $H'=H-1+k$，即经过一次这样的卷积，每个维度的 size 增加 $k-1$，定长增加

    $p'=0, p=k//2$，此时 $H'=H-1 + k - (k-1)=H$，此时 spatial size 保持不变
2. 仍然是非空洞卷积 $d=1$，但是 $s>1$，此时属于上采样，

    $s=2$，此时 $H'=2\times H- 2 - 2 \times p + k+p'$，一种常见的 case 是，我们希望 $H'=2H$，此时要求 $-2-2p+k+p'=0$，如果 $p=k//2$ 且 $k$ 为奇数，（这也是一种常见的选择），那么 $-2-2p+k+p'=p'-1=0 \Rightarrow p'=1$。

    事实上，当 $s>1$ 时，考虑 $p=k//2$，且 $k-2p=1$，根据 {eq}`convt2` 式，
    $$H'=\lfloor \frac {H+2p-k} s \rfloor + 1=\lfloor \frac {H-1} s \rfloor + 1$$
    这就导致有 $s$ 个不同的 $H$ 值，其输出 $H'$ 均相同，在转置卷积中，根据这个 $H'$ 值，我们应该得到原来的 $s$ 个 $H$ 值中的哪一个呢？`output_padding` 的作用就是帮我们确定是哪一个 $H$ 值。


代码：上面三种情况的讨论分别对应下述代码中的三种情况
```python
input = torch.randn(8, 2, 2, 2)
m = nn.ConvTranspose2d(2, 3, 3)      # (in_feat, out_feat, kernel_size)
output = m(input)
output.shape            # (8, 3, 4, 4)      H'=H+k-1=H+2

m1 = nn.ConvTranspose2d(2, 3, 3, padding=1)
output1 = m1(input)
output1.shape           # (8, 3, 2, 2)      H'=H

m2 = nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1, output_padding=1)
output2 = m2(input)
output2.shape           # (8, 3, 4, 4)      H'=2H
```

## 实现原理

1. 上述 $d$ 对普通卷积和转置卷积含义是一样的，即膨胀扩大卷积核 size。
2. $p$ 则需要以普通卷积的角度来理解，而非转置卷积中 input tensor 要填充的 size。
3. $s$ 在转置卷积中表示上采样，也就是说每次移动 $1/s$ 个像素（fractionally-strided 的由来），然而无法做到移动分数个像素，故在每个 dimension 上相邻两个像素之间填充 $s-1$ 个 `0` 值，然后卷积滑动窗口时每次移动 1 个像素，共需 $s$ 次移动才能到达下一个像素（这样等价于每次移动 $1/s$ 个像素）。

例如上述第 `2` 点中，令 $s=2, \ d=1, \ p=k//2=1, \ p'=1$，那么 $H'=2H$，那么实际上转置卷积时，$H$ 先扩大为 $H+(H-1)\cdot (s-1)=2H-1$，然后此时的卷积 `stride=1`，要求此时的 padding size，然后根据 {eq}`convt2` 式（因为转置卷积本质上还是在做卷积操作），得到 $H'=(2H-1)+2p-k + 1=2H+2p-k$，显然需要设置 padding size 为 $p=k//2$，这里 padding size 恰好与普通卷积的 $p=2$ 相等（但是注意有的时候是不等的）。此时 $H'=2H-1$，故要想输出 size 是输入 size 的 2 倍，还需要 one-side 填充 $p'=1$，即 $H'=2H-1+p'=2H$。

当然，如果没有 one-side 填充，那么 $H'=2H-1$，将转置卷积反过来看成普通卷积，即 padding_size=1，stride=2，那么根据 {eq}`convt2` 式有 $H_{out}=\frac {2H-1 +2-3 } 2 +1 =H \neq \frac {H'} 2$

**代码**
```python
input = torch.ones(1,1,2,2)                     # H=2
m = nn.ConvTranspose2d(1,1,4,2,1, bias=False)   # in_feat, out_feat, k, s, p
m.weight = nn.Parameter(torch.ones(1,1,4,4))
output = m(input)
output.shape    # convtranspose's output H'=2(H-1)-2p+k+p'=2H-2p+k-2=2H
# convtranspose's input is zero-padding s-1 in two pixels, so input H:=1+s(H-1)=2H-1
#   and then use stride s'=1 to conv, the padding size is p'=[s'(H'-1)-(2H-1)+k]/2= k/2=2
output
```
上面的代码中，首先在将 input 膨胀到 $2H-1$，然后 padding size $p=2$ ，
```
            0 0 0 0 0 0 0
            0 0 0 0 0 0 0
1 0 1       0 0 1 0 1 0 0
0 0 0   ->  0 0 0 0 0 0 0
1 0 1       0 0 1 0 1 0 0
            0 0 0 0 0 0 0
            0 0 0 0 0 0 0
```
然后使用 $4 \times 4$ 的全 1 矩阵，进行卷积，结果为
```
1 2 2 1
2 4 4 2
2 4 4 2
1 2 2 1
```


## 逆卷积

相关论文：[Deconvolutional Networks](https://matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)

## 逆卷积的过程详解

以下内容完全根据论文 Deconvolutional Networks 进行解读。

对于某个 image 记为 $y^i$，其 channel 数量为 $K_0$，记为 $y_1^i,\cdots, y_{K_0}^i$，image 的每个 channel 由 $K_1$ 个 feature maps $z_k^i$ 与卷积核 $f_{k,c}$ 卷积而来，
$$\sum_{k=1}^{K_1} z_k^i \oplus f_{k,c}=y_c^i, \quad c=1,\cdots,K_0$$ (deconv1)

除特别说明，变量均表示矩阵。

记 $y_c^i$ 的 size 为 $N_r \times N_c$，卷积核 size 为 $H\times H$，那么 latent feature $z_k^i$ 的 size 为 $(N_r+H-1)\times (N_c+H-1)$。{eq}`deconv1` 式是 under-determined 方程组（未知量数量大于等式数量），要求的唯一解，需要引入一个正则项，使得 feature $z_k^i$ 的幅值尽可能小，即，鼓励稀疏性，于是有待优化的目标函数：
$$C_1(y^i)=\frac {\lambda} 2 \sum_{c=1}^{K_0} ||\sum_{k=1}^{K_1} z_k^i \oplus f_{k,c}-y_c^i||_2^2+\sum_{k=1}^{K_1} |z_k^i|^p$$ (deconv2)

上式中，$|w|^p=\sum_{i,j} |w(i,j)|^p$，通常取 $p=1$。

根据 {eq}`deconv2` 式，我们可以从一幅多通道图像得到稀疏的 feature maps。

现在使用一个图像集合 $y=\{y^1,\cdots, y^I\}$，我们希望找出 $\arg \min_{f,z} \ C_1(y)$，注意，不同的图像，其得到的 feature maps 也会不同，但是卷积核 $f$ 则是共享的，也就是说，我们要找出一组合适的卷积核 $f$，使得我们能用尽可能稀疏的 feature maps 与卷积核进行卷积，能得到这些 images。

这看上去已经有了逆卷积的意思。我们还可以 stack 多个这样的操作/layers 从而得到一个网络即，第 `l` 个 layer 的输出 $z_{k,l}^i$ ，其输入是第 `l-1` 个 layer 的输出 $z_{c,l-1}^i$，满足  $z_{k,l}^i$ 与卷积核 $f_{k,c}^l$ 卷积得到  $z_{c,l-1}^i$，
$$\sum_{k=1}^{K_l} z_{k,l}^i \oplus f_{k,c}^l = z_{c,l-1}^i$$ (deconv3)
其中，$i$ 表示网络（也是第 `1` 个 layer）的输入为第 $i$ 个 image，$k$ 表示 `l-th` layer 某输出 channel，$c$ 表示 `(l-1)-th` layer 某输出 channel，于是对所有 images 而言，`l-th` layer 总的损失函数为

$$C_l(y)=\frac {\lambda} 2 \sum_{i=1}^I\sum_{c=1}^{K_{l-1}} ||\sum_{k=1}^{K_l} g_{k,c}^l(z_{k,l}^i \oplus f_{k,c}^l)-z_{c,l-1}^i||_2^2 + \sum_{i=1}^I\sum_{k=1}^{K_l}|z_{k,l}^i|^p$$ (deconv4)

上式中，$g_{k,c}^l \in \{0,1\}^{H_{l-1} \times W_{l-1}}$ 表示 `l-th` layer 的输出 channel $k$ 和 `(l-1)-th` layer 的输出 channel $c$ 是否有连接。

{eq}`deconv4` 式有两类未知变量 $f, z_{k,l}^i$，要想学习得到卷积核参数 $f$，我们交替进行：
1. 固定 $f$ 不变，求使得 $C_l(y)$ 最小的 feature maps $z_{k,l}^i$。 (inference 阶段)
2. 固定 feature maps $z_{k,l}^i$ 不变，求使得 $C_l(y)$ 最小的 $f$

未完待续，由于 deconv 比较复杂，且实际应用并不广泛，暂时不完成本节内容了。