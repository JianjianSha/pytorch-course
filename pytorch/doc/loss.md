# 损失

```{note}
损失通常不作为 model 的 一个 layer，而是一个独立的函数，其输入为 model 的 output 和 target。
```

## L1Loss

```python
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
```

计算 target $y$ 和 输入 $x$ （实际为 model 的输出）绝对误差的均值（如果 `reduction='sum'`，那么求和，不求平均）。

参数：

`size_average`：弃用，改用 `reduction`。

`reduce`：弃用，改用 `reduction`。

使用示例：
```python
loss = nn.L1Loss()
x = torch.randn(3,5, requires_grad=True)    # model 的输出
y = torch.randn(3, 5)                       # target
# x, y  同 shape
l = loss(x, y)
l.backward()
```

## MSELoss
```python
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
```
参数含义同 `L1Loss`。 `MSELoss` 常用于回归任务。

## NLLLoss

负对数似然损失，适用于分类问题。
```python
torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```
假设分类数量为 $C$，单个样本的似然函数

$$L=\prod_{i=1}^C x_i^{y_i}$$

其中 $\mathbf x = (x_1, \cdots, x_C)$ 是各分类的概率（归一化，且和为 1），target 为 $\mathbf y=(y_1, \cdots, y_C)$ 是 one-hot 向量，负对数似然 NLL 为

$$\mathcal L = -\sum_{i=1}^C y_i \log x_i$$

NLL 与交叉熵形式上是相同的，根据交叉熵的定义，

$$CE = -\sum_c p_c \log \hat p_c = -\sum_{i=1}^C y_i \log x_i$$

上式中，$p_c$ 是样本属于 $c$ 分类的真实概率，$\hat p_c$ 是模型对样本属于分类 $c$ 的预测概率，可见 NLL 与 CE 确实具有相同的形式。


**参数：**

`weight`: 1-D Tensor ，可选。每个分类的权重。这在不均衡数据集中非常有用。

`ignore_index`：int 类型，可选。表示分类 index，指定 target 中某些值不对梯度作贡献。例如某些任务将背景也看作分类，分类 index 为 0 ，但是求损失时需要忽略背景分类的损失，参见 {eq}`loss1` 。



`reduction`：字符串，可选。取值为 `'none'|'mean'|'sum'`。如果是取平均，那么仅对 非 `ignore_index` 的 loss element 取平均。


**forward 方法中的 input 为各分类的概率的 log 值，input shape 为 $(N,C)$，或者 $(N,C,d_1,\cdots,d_K)$，后者表示高维 input ，例如计算 2D 图像的每个像素的 NLL loss（图像分割任务）。获取各分类概率的 log 值非常简单，只要在 model 的最后增加一个 `LogSoftmax` 层即可，就可以实现归一化，然后取 log 值。**

设置 `reduction='None'` 即可使得损失未归约，保持 Tensor shape，如果归约，那么则为标量。

未归约的 NLL Loss 为 

$$L=\{l_1,\cdots,l_N\}^{\top}, \ l_n=-w_{y_n} x_{n,y_n}$$ (loss1)

其中 $n$ 为下标，取值范围 $[1, N]$ 或者 $[1, \ N \prod_i d_i]$ ，$y_n$ 表示第 $n$ 个（样本 or 样本像素）的分类索引 $y_n \in [1, C]$，$w_c$ 表示分类 c 对应的权重，

$$w_c = \text{weight}[c] \cdot \mathbb I [c \neq \text{ignore\_index}]$$

由于存在分类的加权权重，所以，平均也是取加权平均，

$$l= \frac 1 {\sum_{n=1}^N w_{y_n}} \sum_{n=1}^N l_n$$

**shape：**

input：$(N,C)$，其中 $N$ 是样本数，$C$ 是分类数。 或者是 $(N,C,d_1,\cdots, d_K)$

target：$(N)$，其中每个值表示对应样本的分类 index，值范围 $[0, C-1]$； 或者说  $(N,d_1,\cdots,d_K)$

output：如果 `reduction='none'` ，那么是 $(N)$ 或者是 $(N,d_1,\cdots,d_K)$ ；否则 输出为 标量 。以未归约为例，input 为 $(N,C)$，target 为 $(N)$，那么每个样本的 $C$ 个分类概率 （log 值），仅有 target 对应的那个分类概率参与 loss 计算，所以 output 为 $(N)$ 。

使用示例：
```python
m = nn.LogSoftmax(dim=1)    # 沿着 channel 方向做归一化，归一化后 shape 保持不变
loss = nn.NLLLoss()
input = torch.randn(3, 5, requires_grad=True)   # shape: (3, 5)
target = torch.tensor([1, 0, 4])                # shape: (3)
output = loss(m(input), target)                 # 标量
output.backward()
```

## CrossEntropyLoss

```python
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
```
计算 输入 和 target 之间的 交叉熵。常用于分类问题。PyTorch 中的 `CrossEntropyLoss` 相当于 `LogSoftmax` 和 `NLLLoss` 结合起来，故 forward 方法的 input 包含了原始的未归一化的得分（model 的最后不需要再使用 Softmax 或 LogSoftmax）。

**参数：**

参数大部分在 `NLLLoss` 已经有了，且含义相同，参考上面的 `NLLLoss` 即可。以下列举不同的：


`label_smoothing`：float ，可选 。范围为 [0.0, 1.0] 。用于平滑所计算的损失。

forward 方法接收的 target 分两种情况：

1. target 中 element 是分类索引，取值范围 $[0,C-1]$ 。未归约损失如下，

    $$L=\{l_1,\cdots, l_N\}, \quad l_n=-w_{y_n} \log \frac {\exp (x_{n,y_n})} {\sum_{c=1}^C \exp(x_{n,c})} \cdot \mathbb I[y_n \neq \text{ignore\_index}]$$

    跟 NLLLoss 比起来，就是输入 $x$ 多了一个 $\log (\exp(\cdot)/\sum \exp (\cdot))$ 这一 “归一化然后求 log” 的操作。

    归约版本的 loss 为

    $$l=\begin{cases} \frac 1 {\sum_{n=1}^N w_{y_n} \cdot \mathbb I[y_n \neq \text{ignore\_index}]}\sum_{n=1}^N l_n , & 'mean'
    \\
    \\ \sum_{n=1}^N l_n , & 'sum'
    \end{cases}$$

2. target 中 element 是一个向量，长度为 $C$，表示各分类的概率。常用于样本不是单一分类标签，而是多分类标签（multi label），或者平滑分类标签。此时未归约 loss 为
    $$L=\{l_1,\cdots,l_N\}, \quad l_n = -\sum_{c=1}^C w_c \log \frac {\exp x_{n,c}} {\sum_{i=1}^C \exp x_{n,i}} y_{n,c}$$

    事实上，第 `1` 种情况正是第 `2` 种情况 target 取 one-hot 向量 。

    归约 loss 如下：

    $$l=\begin{cases}\frac 1 N \sum_{n=1}^N l_n , & 'mean' 
    \\
    \\ \sum_{n=1}^N l_n , & 'sum' 
    \end{cases}$$
    其中，均值需要除以分母 $N$，这是因为 $\sum_{n=1}^N \sum_{c=1}^C w_{n,c}=\sum_{n=1}^N 1 = N$ 。

### label smoothing
专门用一小节讨论标签平滑。相关论文为 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) 。

对于训练样本 $x$，模型输出的分类预测概率为 

$$p(k|x)=\frac {\exp z_k} {\sum_{i=1}^K \exp z_i}$$ (loss2)

其中 $z_k$ 为模型输出的未归一化得分向量中类别 $k$ 的分量。 ground-truch 分类概率记为 $q(k|x)$，为了叙述简洁，省略概率关于 $x$ 的条件，损失采用交叉熵损失 

$$l=-\sum_{k=1}^K \log (p(k)) \cdot q(k)$$ (loss3)

计算梯度

$$\begin{aligned}\frac {\partial l} {\partial z_k}&=-\sum_{i=1}^K \frac {p'(i)} {p(i)} q(i)
\\&= \sum_{i \neq k} \frac {q(i) }{p(i)} \frac {\exp z_i \cdot \exp z_k}{(\sum_j \exp z_j)^2} + q(k)\cdot [p(k)-1]
\\&= \sum_{i \neq k} q(i) \cdot \frac {\exp z_k}{\sum_j \exp z_j}+ q(k)\cdot [p(k)-1]
\\&=\sum_i q(i) \cdot p(k)-q(k)
\\&= p(k) - q(k)
\end{aligned}$$

上面推导用到了 $\sum_i q(i)=1$ 。可见梯度的范围固定在 $[-1,1]$ 之间。

考虑单标签情况，即每个样本只对应一个分类标签，记为 $y$，于是 $q(y)=1$，并且 $q(k)=0, \ k \neq y$，也可以写作 $q(k)=\delta_{k,y}$，此时要最小化损失，等价于最大化 $y$ 所对应的似然，根据 {eq}`loss2` 式，要尽可能让 $z_y \gg z_k, \ k \neq y$，但是这存在两个问题：

1. 容易导致过拟合，对样本拟合的太好，从而影响泛化。
2. 最小化损失等价于让 $z_y - z_k$ 的差尽可能地大，而上面说到梯度 $\partial l / \partial z_k$ 值有限，位于 $[-1,1]$ 之间，限制了模型调整的能力，所以较难使得 $z_y-z_k$ 尽可能大 。

直觉上，是因为模型对预测太过于自信。我们增加一个方法机制使得模型变得不那么自信，虽然这个做法与我们在训练集上最大化似然函数这个初衷相违背，但是却可以对模型起到一个正则化的效果，从而使模型泛化能力更强。这个方法机制如下：

考虑标签上的一个分布 $u(k)$，这个分布独立于训练集。一个标签平滑因子 $\epsilon$。对某个样本 $(x,y)$ ，其中 $y$ 是样本分类标签，将标签分布 $q(k|x)=\delta_{k,y}$ 替换为

$$q'(k|x)=(1-\epsilon) \delta_{k,y} + \epsilon \cdot u(k)$$

即，使用先验概率进行平滑校正，这个方法也称为 “标签平滑正则化” （LSR）。常用的先验分布为 $u(k)=1/K$ 即等概率分布。于是上式变为 

$$q'(k|x)=(1-\epsilon) \delta_{k,y} + \frac {\epsilon} K$$

LSR 可以有效地阻止最大的 $z_k$ 远大于 $z_i, \ i \neq k$ 这一现象的发生，这是因为，如果发生这种现象，那么 $p(k) \rightarrow 1$，且 $p(i) \rightarrow 0, \ i \neq k$，那么根据 {eq}`loss3` 式，交叉熵损失 $l$ 在 $q'(k)$ 上比在 $q(k)$ 上大 ，

$$l'-l=\sum_{i=1}^K -[\log p(i)][q'(i)-q(i)] \approx \sum_{i \neq k} -[\log p(i)]\cdot q'(i) > 0$$

上式中，$-[\log p(i)]>0$ 且 $q'(i)>0$  。



**shape：**

input： $(N,C)$，或者 $(N,C,d_1,\cdots, d_K)$

target: 情况 1：$(N)$，或者 $(N,d_1,\cdots, d_K)$，其元素取值范围 $[0, C-1]$； 情况 2：shape 与 input shape 相同，元素取值范围 $[0.0,1.0]$

output：输出为损失。未归约时，为 $(N)$ 或者 $(N,d_1,\cdots, d_K)$，对于 target 是情况 2，output shape 也是如此，因为计算损失时有 $\sum_{c=1}^C$ 。归约时，损失为一标量。