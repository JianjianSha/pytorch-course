# 卷积

## 1. Conv2d

```python
torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,
dilation=1,groups=1,bias=True, padding_mode='zeros', device=None,dtype=None)
```

二维卷积是最常用的一个卷积。输入 tensor 的 shape 为 $(N,C_{in},H_{in},W_{in})$，输出 tensor 的 shape 为 $(N,C_{out},H_{out},W_{out})$，其中 $N$ 表示 Batch 大小。二维卷积，需要指定 $C_{in}$ 、 $C_{out}$ 和卷积核 size $k$ 等，以便确定本 layer 的 参数量 $k \times k \times C_{in} \times C_{out}$ 。当然如果指定 `bias=True`，表示卷积还需要增加一个偏置，那么每个 filter 使用一个 bias，参数量增加 $C_{out}$ 。

### 1.1 参数
**in_channels** 输入 tensor 的通道数。整型

**out_channels** 输出 tensor 的通道数。整型

**kernel_size** filter size 。可以是整型，或者两个整型组成的 tuple 

**stride** 步长。可以是整型，或者两个整型组成的 tuple 

**padding** 填充，指示填充量。可以是字符串，或者可以是整型，或者两个整型组成的 tuple，为字符串时，必须是 `same` 或者 `valid`。

**dilation** 膨胀率。可以是整型，或者两个整型组成的 tuple 

**groups** 分组数量，用于分组卷积。整型

**bias** 卷积是否使用偏置。bool 类型

**padding_mode** 填充模式。字符串类型，必须是 `zeros, reflect, replicate, circular` 之一

**device** 指定本 layer 的参数在哪个设备上

**dtype** 指定本 layer 的参数是什么类型

以上参数中，如果是 `可以是整型，或者两个整型组成的 tuple`，那么整型会被 repeat 成两个元素的 tuple，表示 H, W 方向上使用相同的参数值。

### 1.2 膨胀卷积
也称空洞卷积，在 filter 中注入空洞，填充 `0`。膨胀率指的是 kernel 中点的间隔数，正常情况下，kernel 的 H，W 方向的点间隔为 1（即，移动 1 个单位可以使得重合）。记膨胀率为 $(d_H,d_W)$，那么卷积核变为 $k_i'=d_i \times (k_i-1)+1, \ i=H,W$，当 $d_i=1$ 时，$k_i'=k_i$ 。

### 1.3 填充

根据 `padding` 确定填充量，记填充量为 $(p_l,p_r,p_t,p_b)$，分别表示左右上下的填充量（注，习惯上这里是先行后列的顺序，与维度中先 H 后 W 的顺序相反），`padding` 的几种取值：

1. `padding='valid'` ：没有任何填充，填充量为 $(0,0,0,0)$
2. `padding='same'` ：对 input 进行填充，使得输出与输入的 H, W 分别相等。此时 stride 必须为 `1` 或者 `(1,1)`，否则报错。

    由于步长为 1，要使得输出 size 与输入 size 相等，那么填充量为卷积核 大小减 1，即 $p_l+p_r=d_W \times (k_W-1)=P_W$，$p_t+p_b=d_H \times (k_H-1)=P_H$，故填充量为 $(P_W // 2, P_W-p_l, p_t=P_H//2,P_H-p_t)$ 

3. `padding` 为数值类型： 可以是整数 $p$ ，此时会展开为 tuple $(p,p)$，或者直接就是 tuple $(p_H, p_W)$。分别指定列和行的一半的填充量，故最终填充量为 $(p_t,p_b, p_l,p_r)$，分别表示上下左右的填充量。


根据 `padding_mode` 确定填充什么数据。`padding_mode` 的几种取值：

1. `zeros`：全部填充 `0` 值
2. `reflect`：使用 input 边界的镜像元素值。例如，

    ```python
    import torch
    import torch.nn as nn
    m = nn.ReflectionPad2d(2)   # 左右上下均填充两个
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
    m(input)
    m2 = nn.ReflectionPad2d(5)
    m2(input)   # 报错
    ```
3. `replicate`：重复使用边界值。例如，

    ```python
    m = nn.ReplicationPad2d(2)   # 左右上下均填充两个
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
    m(input)
    ```

4. `circular`：循环填充。例如一维数据 `[0,1,2,3]`，填充量为 `[1,1]`，那么填充后为 `[3,0,1,2,3,0]`，填充量 `[2,2]` 时填充结果为 `[2,3,0,1,2,3,0,1]`，也就是说保持 `0,1,2,3` 这个循环顺序。更多例子，
    ```python
    import torch.functional as F
    input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
    padding = (1,1,1,1)
    out = F.pad(input, padding, mode='circular')
    padding = (-1,-1,-1,-1)
    out2 = F.pad(input, padding, mode='circular')
    ```

### 1.4 输出 size

前面确定了填充大小之后，再结合输入 size，步长，kernel size（经过膨胀处理，默认情况为膨胀率=1），于是可以确定输出 size 为

$$W_{out}=\lfloor \frac {W_{in}+p_l+p_r-d_W \times (k_W-1)-1} {s_W} +1\rfloor$$

$$H_{out}=\lfloor \frac {H_{in}+p_t+p_b-d_H \times (k_H-1)-1} {s_H} +1\rfloor$$

向下取整表示：右侧或者底部不足一个 kernel size 的 feature 将被忽略，不做卷积。
### 1.5分组卷积

分组数记为 $g$，输入输出均沿着 channel 方向分成 $g$ 组，这表示输入输出的 channel $C_{in}, \ C_{out}$ 均能被 $g$ 整除，分组后，每组输入 shape 为 $(N,C_{in}/g, H_{in}, W_{in})$，即，输入沿着 channel 方向切割分成 $g$ 份，**kernel 沿着数量方向分成 $g$ 等份，也就是 $g$ 组 kernel，每组 kernel 的数量为 $C_{out}/g$，同时需要将 kernel 沿着 channel 方向也切割成 $g$ 等份**，这样 kernel 才能与切割后的输入做卷积，故每组 kernel shape 为 $(C_{out}/g, C_{in}/g,k_H,k_W)$，每组输出 shape 为 $(N, C_{out}/g,H_{out},W_{out})$，最后将所有组的输出沿着 channel 方向 concatenate 起来，得到最终输出 shape 为 $(N, C_{out}, H_{out}, W_{out})$ 。

分组卷积的 kernel 参数量为 

$$(k_W \times k_H \times C_{in} \times C_{out} / g^2) \times g=k_W \times k_H \times C_{in} \times C_{out} / g$$

是不分组时参数量的 $1/g$，故参数量降低，一定程度上避免过拟合，同时加快计算效率。

