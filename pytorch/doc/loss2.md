# 损失（二）

## MarginRankingLoss

```python
torch.nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```
**数学表示：**

给定两个输入 $x_1, \ x_2$，以及一个 label 值 $y \in \{1,-1\}$。当 $y=1$，认为 $x_1$ 应该比 $x_2$ 大；当 $y=-1$，认为 $x_1$ 应该比 $x_2$ 小，所以损失为
$$l=\max(0, -y(x_1-x_2) + \text{margin})$$
上式中增加了一个 `margin` 项，当
$$-y(x_1-x_2)+\text{margin} \le  0$$

时，损失为 0（没有损失），分以下两种情况讨论：

1. $y=1$ ，需要 $x_1 \ge x_2 + \text{margin}$ 损失才降为 0
2. $y=-1$ ，需要满足 $x_1 \le x_2-\text{margin}$ 损失才降为 0。

故 `margin` 的含义为：如果 $x_1$ 需要比 $x_2$ 大，那么至少要大 `margin`；如果 $x_1$ 需要比 $x_2$ 小，那么至少要小 `margin` 。

以下代码显示了 margin 为 1， 且 $y=1$ 时， MarginRankingLoss 的图。

```python
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

x1 = np.arange(-5, 5, 0.25)
x2 = np.arange(-5, 5, 0.25)



X1, X2 = np.meshgrid(x1, x2)

y = 1.0     # x_1 need to be large than x2
m = 1.0     # margin

# loss = max(0, -y(x1-x2)+m)
#      = max(0, -(x1-x2)+1)
# when -x1 + x2 + 1 <= 0, the loss is 0
# so, the key line is -x1+x2+1=0
# the directional vector is (1,1)
# the normal vector is (-1,1) or (1, -1)
# To let loss > 0, then  -x1+x2+1 > 0, 
#   the related normal vector is (-1, 1)
#   let's plot the normal vector, 
#   the normal line is x1+x2=0
#   combine x1+x2=0 and -x1+x2+1=0
#   we get the intersection point (1/2,-1/2)


x2 = x1 - m

l = np.maximum(0, -y * (X1 - X2) + m)
surf = ax.plot_surface(X1, X2, l, cmap=cm.coolwarm)

# 画 loss 临界线
ax.plot(x1, x2, zs=0, color='k')
# 画 loss > 0 的方向
ax.quiver(0.5,-0.5,0,-2,2,0, color='r', arrow_length_ratio=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('margin loss')

plt.show()
```

**shape：**

列出 forward 方法的各参数

input1 shape：$(N)$，$N$ 表示一个批大小

input 2 shape：$(N)$

target shape：$(N)$， binary value，指示 $y=1$ 或 $y=-1$

output shape：归约则为一个标量，否则为 $(N)$

## MultiLabelMarginLoss
```python
torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
```

适用于多标签多分类问题。

**原理：**

每个类别独立进行二分类（为正 or 为负）预测，预测值 x 是一个 2D tensor，shape 为 $(N,C)$，其中 $N$ 表示批大小，$C$ 表示类别数。target 与 x 同 shape。暂且考虑单个样本，此时 x 和 target 均为长度 `C` 的向量，x 表示各分类的预测概率，target （用 y 表示）表示样本所属分类索引，例如 $y=(3,0,-1,1)$，表示样本属于 `0` 分类和 `3` 分类，从第一个负值开始，之后的全部忽略。借鉴 `MarginRankingLoss` 思想，对于预测值 x，认为其中<b>样本所属分类的元素值比样本不属分类的元素值大</b>，这个例子中，样本所属分类为 $\{0,3\}$，所以认为应该是 $x_0,\ x_3 > x_1,\ x_2$，据此不难理解单个样本的损失为
$$l=\sum_{i,j} \frac {\max[0, 1-(x_{y_j} - x_i)]} C$$

其中，$j \in \mathcal J=\{0,1,...,k-1\}$，且 $y_k<0$，$i \in \{0,1,...,C-1\}-\{y_j|j \in \mathcal J\}$，即， $j$ 为 target 向量中开始的连续非负元素索引，$y_j$ 表示样本所属分类索引，$i$ 为样本不属分类索引，并且 $x_{y_j}$ 比 $x_i$ 至少大 $1$ 时，损失为 0 。 实际上 $x_{y_j}, x_i \in [0,1]$ 故 $x_{y_j} - x_i \in [-1, 1]$。

当分类正确时，损失为0，此时需要满足条件 $1-(x_{y_j}-x_i)\le 0 \Rightarrow x_{y_j}\ge 1+x_i$，这说明降低损失会使得样本所属分类的预测概率 $x_{y_j} \rightarrow 1$，样本不属分类的预测概率 $x_i \rightarrow 0$。在 test 阶段，对预测值 x 设置一个低阈值即可，大于阈值的均认为属于样本分类。

**shape：**

input：$(C)$ 或者 $(N,C)$

target：与 input 有相同的 shape。target 值的范围为 $[0,C-1]$，如果样本的分类标签数量不足 C，那么在后面填充 $-1$，以便与 input 同 shape。

output：归约时为一标量。未归约时，为 $(1)$ 或者  $(N)$ 。

使用示例：
```python
loss = nn.MultiLabelMarginLoss()
x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
y = torch.LongTensor([[3,0,-1,1]])
output = loss(x, y)
```


## SoftMarginLoss
```python
torch.nn.SoftMarginLoss(size_average=None, reduce=None, reduction='mean')
```
适用于二分类问题。

**原理：**

上面两种 MarginLoss 均采用了 `max(0,x)` 函数，这个函数在 `x=0` 处不可导。`SoftMarginLoss` 借助 logistic 函数解决了这个问题。Logistic 函数
$$\sigma(x)=\frac 1 {1+\exp (-x)}$$
预测值 x，分类 $y\in \{1,-1\}$，似然函数为
$$\mathcal L =\mathbb I(y=1)f(x)+\mathbb I(y=-1)(1-f(x))=[1+\exp(-yx)]^{-1}$$
 负对数似然函数（损失）为
$$l= \log(1+\exp(-yx))$$
所以 `SoftMarginLoss` **就是 logistic 回归的负对数似然损失**。

**shape：**

input ： $(\star)$，其中 $\star$ 表示任意维度，element 值表示正例的预测概率 。对每个值分别计算损失。

target：与 input 的 shape 相同。

output：损失按像素计算，输出与 input 同 shape，如果按求和或平均归约，那么输出为一标量。

## MultiLabelSoftMarginLoss

```python
torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=None, reduce=None, reduction='mean')
```

适用于多标签多分类任务。

**参数：**

`weight`：每个分类的权重。1-D Tensor，包含 $C$ 个值。

**原理：**

每个类别各自独立做二分类（为正或负）。input 和 target 有相同的 shape：$(N,C)$，target 值为 0 或 1（这与 SoftMarginLoss 的 1 或 -1 是不同的）。于是，单个样本的损失为，

$$l=-\frac 1 C \sum_{i=1}^C w_c \left[y_i \log \left(\frac 1 {1+\exp(-x_i)}\right )+(1-y_i)\log \left(\frac {\exp(-x_i)} {1+\exp(-x_i)}\right)\right]$$
由于这里考虑单个样本，所以上式 $x, \ y$ 均为长度 $C$ 的向量，由于 y 值取值范围不同，所以上式与 `SoftMarginLoss` 的损失表达式略有不同，但是本质上都是 logistic 负对数似然损失。

**shape：**

input：$(N,C)$。 input 为 model 的输出，表示各分类未归一化的得分。

target：$(N,C)$，值为 `0` 或者 `1`。分类总数量为 $C$，每个样本的 target 为长度 $C$ 的向量，如果分类 $c$ 属于样本分类标签，那么向量 $c-1$ 位置为 `1`，否则为 `0`。

output：标量。如果未归约，那么是 $(N)$ 。

**注：**

多标签分类，例如一部电影，可以有“情感片”，“动作片” 等多个分类，但是这些分类是不分先后顺序的，对于一个数字字母的验证码图片，也可以看作是多个分类，然而这些分类是分先后顺序的，不能像前面那样，使用 $C$ 长度的向量作为 target。 事实上，target 中不同位置代表不同的分类，对于验证码图片而言，例如 $(1,2,3,1)$，第一个 `1` 和 最后一个 `1` 虽然字符一样，但是在字符序列中位置不一样，所以 **target 既要考虑到 分类，也要考虑到 位置，记验证码的字符数量为 $n$，那么可以使用 $n\cdot C$ 长度的向量来表示 target，向量中每个元素值依然为 `0` 或 `1`，向量中每 $C$ 个 element 划分为一个桶（bin），每个桶是一个 one-hot，表示 $C$ 个分类中的一个分类（对应的字符），然后桶的位置就是桶所表示的字符在验证码中的位置。**

