# 批归一化

## BatchNorm2d

```python
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```
参数

`num_features`：特征数量，即 input $(N,C,H,W)$ 中的 $C$ 。

`eps`：一个非常小的值，加到分母上，防止除零错误 。

`momentum`：动量系数

`affine`：是否是仿射函数（是否有可学习的 $\gamma, \ \beta$）

`track_running_stats`：是否追踪运行均值，和运行方差。

```{note}
momentum 与优化器中的 momentum 稍有不同，这里的动量更新满足 $\hat x := (1-momentum) \times \hat x + momentum \times x_t$ 。
```

批归一化的公式为 

$$y=\frac {x-E[x]} {\sqrt{V[x]+\epsilon}} * \gamma + \beta$$ (batchnorm1)

归一化是在各个 channel 上独立进行的，故参数 $\gamma$ 有 $C$ 个，所以需要提供 `num_features`，以便确定本层参数。若 `affine=True`，那么还有 $C$ 个 $\beta$ 参数。$\gamma, \ \beta$ 均是可学习参数。$\epsilon$ 就是 `eps`，$x$ 是 feature 上各像素点的值。

**批归一化的作用：**

1. 减均值除方差，远离饱和区（梯度消失）
2. 缩放加移位：避免线性区

    仅使用 “减均值除方差”，使得数据近似标准正态分布，大部分数据位于 $[-1, 1]$ 之间，而激活函数 sigmoid 在这个区间近似线性，没有起到非线性变化的效果，所以通过缩放加移位，使得大部分数据不在 $[-1,1]$ 这个区间。为了一定程度上恢复学习到的特征分布，缩放和移位的参数是可学习的。

由于卷积神经网络的参数共享机制，每一个卷积核的参数在不同位置的神经元是共享的，所以应该被一起归一化，这就是为什么 output 的 channel 上分别独自进行批归一化的原因。

**批归一化的位置：**

1. 激活函数类似 sigmoid 有一定的饱和区间，那么批归一化位于激活函数之前，可以缓解梯度消失问题。
2. 激活函数类似 relu ，由于不存在饱和区，所以批归一化位于激活函数之后（如果放在激活函数之前，容易使得非线性特征趋于同化，因为 relu 有一半是直接置零）。

**批大小：**
批大小不能太小，否则 BN 性能降低（有论文实验验证），数据样本太少，得不到有效的统计量，噪音干扰较大。一般 16 以上 。

**BN 使模型正则化：**
BN 使得数据归一化，故一般不需要 dropout ， L2 正则约束系数也可以选用更小的即可。

**BN 的缺点：**

1. 分类任务属于粗粒度的任务，适合使用 BN，但是如果是像素级的图片生成任务，那么 BN 效果可能不佳，因为 batch 中的图片在任务所关注的一些点上相互无关，计算统计量弱化了单张图片的细节信息。
2. RNN 网络不适合使用 BN，因为 batch  中 各个样本长度可能不同，不同时间步的隐层看到的输入数据量也不同，所以 BN 效果大打折扣。

3. 训练和推理两个阶段的统计量不一致。推理阶段可能是单个样本，无法计算统计量，一般解决办法是，推理阶段采用训练阶段的统计量，但两个阶段统计量可能并不一致，故导致问题。（后续也有人提出了 instance normalization 等多种归一化方法）

### 源码

构造 BatchNorm2d 实例

```python
if self.affine:     # factory_kwargs: {'device': device, 'dtype': dtype}
    self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
    self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
else:
    self.register_parameter('weight', None)
    self.register_parameter('bias', None)

if self.track_running_stats:
    self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
    self.register_buffer('running_var', torch.zeros(num_features, **factory_kwargs))
    self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, ...))
else:
    self.register_buffer('running_mean', None)
    self.register_buffer('running_var', None)
    self.register_buffer('num_batches_tracked', None)
```

重置运行统计量
```python
if self.track_running_stats:
    self.running_mean.zero_()
    self.running_var.fill_(1)
    self.num_batches_tracked.zero_()
```

重置可学习参数
```python
if self.affine:
    init.ones_(self.weight) 
    init.zeros_(self.bias)
    # 不能使用 self.weight.fill_(1)，否则报错：a leaf Variable that requires grad
    #   is being used in an in-place operation. 简单说，就是这种对自身的操作无法求导。
    # 事实上，init.ones_(self.weight) 内部就是 
    #   with torch.no_grad():
    #       self.weight.fill_(1)
```

**前向传播**

获取计算统计量所需要的系数，采用 “移动平均”

```
running_mean = exponential_average_factor * running_mean + (1-exponential_average_factor) * sample_mean
```

这里的 “移动平均因子” 有前缀 `exponential` 表示这是一个指数型的因子，事实的确如此，使用数学表示上面这个更新公式为

$$\begin{aligned} \mu_{t} &= (1-a) \cdot \mu_{t-1}  + a \cdot s_t
\\&=(1-a) \cdot[(1-a) \cdot \mu_{t-2}+a\cdot s_{t-1}] + a \cdot s_t
\\&=(1-a)^2 \mu_{t-2} + (1-a)a\cdot s_{t-1}+a\cdot s_t
\\&= \cdots
\\&=(1-a)^t \mu_0+(1-a)^{t-1}a\cdot s_1 + \cdots + (1-a)a \cdot s_{t-1} + a \cdot s_t
\\&= a\sum_{i=1}^t (1-a)^{t-i} s_i
\end{aligned}$$ (batchnorm2)
上式中，$\mu_t$ 表示 $t$ 时刻的统计量（均值，layer 接收到 $t$ 时刻的 mini-batch 之后计算的移动统计量），$s_t$ 表示 $t$ 的 mini-batch 统计量，$a$ 表示 `momentum` 。$t=0$ 时刻，$\mu_0=0$ ，因为没有看到任务 mini-batch 。

如果未设置 `momentum`，那么

$$\begin{aligned}\mu_1&=s_1
\\ \mu_2&=\frac 1 2 (s_1+s_2)=\frac 1 2 \mu_1+\frac 1 2 s_2
\\ \mu_3&=\frac 1 3(s_1+s_2+s_3)=\frac 2 3 \mu_2+ \frac 1 3s_3 
\\ \cdots
\\ \mu_t&=\frac 1 t \sum_{i=1}^t s_i=\frac {t-1} t\mu_{t-1}+ \frac 1 ts_t
\\ &=(1-a) \cdot \mu_{t-1} + a \cdot s_t
\end{aligned}$$ (batchnorm3)

其中 $a=1/t$ 。

相关代码为，

```python
if self.momentum is None:
    exponential_average_factor = 0.0
else:   # 如果设置了 momentum，那么就使用 momentum 的值
    exponential_average_factor = self.momentum

if self.training and self.track_running_stats:
    if self.num_batches_tracked is not None:
        # 训练阶段，追踪遇到的批的次数。
        self.num_batches_tracked = self.num_batches_tracked + 1
        # 如果未设置 momentum，但是又处在训练阶段，根据上面的推导公式
        #   mu_t = (t-1)/t * mu_{t-1} + 1/t * s_t
        #   可知，因子取值为 1/t，t 为遇见 mini-batch 的次数
        if self.momentum is None:
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:
            exponential_average_factor = self.momentum
```

是否计算 mini-batch 的 BN 统计量
```python
if self.training:       # 训练阶段，需要计算
    bn_training = True
else:
    # eval 阶段，如果 running_mean running_var 均为 None，那么需要计算统计量，
    #   否则，使用统计量缓存即可，例如 eval 阶段预测单个实例，单个实例组成的 mini-batch
    #   计算统计量显然不合适，样本太少
    bn_training = (self.running_mean is None) and (self.running_var is None)
```

## LazyBatchNorm2d

```python
torch.nn.LazyBatchNorm2d(eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

`BatchNorm2d` 的 lazy 版本，指 `weight, bias, running_mean, running_var` 可以被延迟初始化，根据 forward 时的 input.size(1) 进行初始化。

