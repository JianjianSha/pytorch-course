# 调整学习率

学习率不应该保持不变，这是因为刚开始，学习率不能太小，否则收敛速度太慢，而随着迭代的进行，越来越靠近最优解，这时学习率不能太大，否则会在最优解附近振荡，所以学习率应该先大后小。

`torch.optim.lr_scheduler` 模块提供了很多调整学习率的方法。

```{note}
学习率的调整必须要在优化器更新参数之后进行
```

## LambdaLR

```python
lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
```
参数说明：

1. `optimizer` 指定优化器
2. `lr_lambda` 指定函数或函数列表，如果是函数列表，那么每个函数负责一个参数组 （one of optimizer.param_groups）。
3. `last_epoch` 指示上一个 epoch。默认为 `-1`，表示本次是从头开始训练，否则 `last_epoch` 表示上一次训练结束时的 epoch 值，本次是接着上次继续训练。
4. `verbose` bool 类型，是否打印更新信息。

使用示例：

```python
lambda1 = lambda epoch: epoch // 30 # 前 30 epoch 学习率均为 0 。之后，每 30 个 epoch 则倍乘初始学习率。
lambda2 = lambda epoch: 0.95 ** epoch   # 缓慢变化，每一个 epoch 均改变
scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()  # 在优化器更新参数之后，调整学习率
```

**更新规则：**

1. 必须要有初始学习率，每个参数组（dict 类型）必须要有 `initial_lr`。`last_epoch=-1` 时，初始学习率 `initial_lr` 可以没有，如果没有，采用学习率 `lr` 。
    ```python
    if last_epoch == -1:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    else:
        for i, group in enumerate(optimizer.param_groups):
            if 'initial_lr' not in group:
                raise KeyError()
    # base_lrs 保存了各参数组对应的初始学习率
    self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
    ```

2. 每次执行 `step()`，进行学习率的修改：对 `last_epoch` 应用 lambda 函数，得到修改系数，然后乘上初始学习率，如下
    ```python
    # 获取各参数组调整后的学习率
    lrs = [base_lr * lmbda(self.last_epoch) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)

    # 各参数组的学习率更改
    for i, data in enumerate(zip(self.optimizer.param_groups, lrs)):
        param_group, lr = data
        param_group['lr'] = lr
    ```

### 注意

1. 每次执行 `step()`，`last_epoch` 增 1 。在 LambdaLR 的构造函数（其实是 `__init__` 函数），会执行一次 `step()`，所以构造出 LambdaLR 对象记为 `scheduler`，

    - 若本次是从头开始训练，那么 `scheduler.last_epoch=0`
    - 若本次是接着上次训练继续进行，记上次训练结束时 epoch 值为 `e0`，那么 `scheduler.last_epoch=e0+1`

    上面第二点 “接着上次训练继续进行”：我们训练中途可以保存 checkpoint 文件，从而停止训练，下次我们可以从这个 checkpoint 文件开始接着训练，而不需要从头开始重新训练。这个 checkpoint 文件会保存我们上一次训练完成时的 epoch 值，然后我们加载这个 checkpoint 时，需要给 LambdaLR 对象构造参数 `last_epoch` 赋值为上一次训练停止时的 epoch 值，然后在构造函数（其实是初始化函数 `__init__`）中执行一次 `step()` ，使得 `last_epoch` 增 1，即从下一轮 epoch 开始训练。

2. `step(epoch=None)` 函数也可以主动给 `epoch` 参数赋值，这个值会赋给 `self.last_epoch`，也就是说，手动指定从哪一个 `epoch` 开始 。

3. 当训练是接着上次训练继续进行时，`last_epoch` 必须手动设置（非默认值 `-1`），且此时各参数组必须保证设置了 `initial_lr`，否则报错 KeyError。

### 总结
1. 获取各参数组的 `initial_lr` 值，如未设置这个 key 的值，那么 `last_epoch` 必须为 `-1`，此时 `initial_lr` 值取 `lr` 值，如果 `last_epoch` 不为 `-1`，报错。
2. 更新学习率时，首先根据当然 epoch 值（即 `last_epoch` 字段的值）应用 lambda 函数得到系数，然后乘以 `initial_lr` 的值即可。

## MultiplicativeLR

```python
lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
```

使用示例：

```python
lmbda = lambda epoch: 0.95
scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

`MultiplicativeLR` 与 `LambdaLR` 非常相似，只是更新公式不同，如下：

1. `last_epoch=0` （从头开始训练，第一轮训练开始之前）时，学习率使用 `initial_lr`，作为下一轮（即，第一轮）学习率。

2. `last_epoch>0` 从第二轮（`last_epoch=1`）开始，学习率为**上一轮的学习率乘以 lambda 函数结果**。

注意，`LambdaLR` 每一轮（包括第一轮 `last_epoch=0`）都是初始学习率 `initial_lr` 乘上 lambda 函数执行结果，然后作为下一轮学习率；而 `MultiplicativeLR` 则是**累乘**。


## StepLR

```python
lr_sheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)
```
使用示例：

```python
# 假设每个参数组的学习率均相同，为 `lr=0.05`
# lr=0.05               epoch < 30
# lr=0.005              30 <= epoch < 60
# lr=0.0005             60 <= epoch < 90
# ...
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

每经 `step_size` 个 epoch，阶跃改变学习率：为上一轮的学习率乘以 `gamma`。

## MultiStepLR

```python
lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)
```

参数：
1. `milestones`：索引列表，必须是从小到大排序，表示 epoch 每到达`milestone` 的一个元素值，就改变学习率为：上一轮的学习率乘以 `gamma`

使用示例：
```python
# 假设各参数组的学习率均相同，为 `lr=0.05`
# lr=0.05               epoch < 30
# lr=0.005              30 <= epoch < 80
# lr=0.0005             epoch >= 80
scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
```

## ConstantLR
```python
lr_scheduler.ConstantLR(optimizer, factor=1.0/3, total_iters=5, last_epoch=-1, verbose=False)
```
在 epoch 到达 `total_iters` 之前，学习率为 各参数组中 `lr` 值乘以 `factor` 的值，到达 `total_iters` 以及之后，学习率恢复到 `lr` 值。

使用示例：
```python
# 假设各参数组的学习率均为 `lr=0.05`
# lr=0.025      0 <= epoch <4
# lr=0.05       epoch >= 4
scheduler = ConstantLR(optimizer, factor=0.5, total_iters=4)
```

## LinearLR
```python
lr_scheduler.LinearLR(optimizer, start_factor=1.0/3, end_factor=1.0, total_iters=5, last_epoch=-1, verbose=False)
```

在 epoch 从 `0` 到 `total_iters` ，随着 epoch 增加线性改变 学习率：学习率 `lr` 乘以一个因子，这个因子从 `start_factor` 线性增大到 `end_factor` 。

学习率为 $\epsilon=\epsilon_0 \cdot f$，因子计算式为

$$f=f_s+(f_e-f_s)\cdot t / T$$ (scheduler-1)

其中，$f_s$ 为 `start_factor`，$f_e$ 为 `end_factor`，$T$ 为 `total_iters`，$t \in [0, T]$ 为 epoch 。


在 `total_iters` 之后，学习率保持不变。

使用示例：

```python
# 假设各参数组的学习率均为 0.05
# lr=0.025          epoch=0
# lr=0.03125        epoch=1
# lr=0.0375         epoch=2
# lr=0.04375        epoch=3
# lr=0.05           epoch=4
scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

注意：

由于 PyTorch 源码中，每次更新后的学习率均保存到各参数组中 `group['lr']`，所以没有直接使用 {eq}`scheduler1` 式，而是使用了迭代形式更新，记 学习率为 $\epsilon$，

$$\begin{aligned}\epsilon_{t+1}&=\epsilon_0 \cdot f_{t+1}
\\&=\epsilon_0 \cdot [f_s+(f_e-f_s)\cdot (t+1) /T]
\\&=\epsilon_0 \cdot [f_s+(f_e-f_s) \cdot t /T] +\epsilon_0 \cdot (f_e-f_s)/T
\\&=\epsilon_t + \epsilon_0 [f_s+(f_e-f_s) \cdot t /T] \frac {f_e-f_s}{f_s T+(f_e-f_s)t}
\\&=\epsilon_t + \epsilon_t \frac {f_e-f_s}{f_sT + (f_e-f_s)t}
\\&= \epsilon_t [1+\frac {f_e-f_s}{f_sT + (f_e-f_s)t}]
\end{aligned}$$

即 $$\epsilon_t=\epsilon_{t-1}[1+\frac {f_e-f_s}{f_sT + (f_e-f_s)(t-1)}], \quad t \in [1,T]$$

## ExponentialLR

```python
lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)
```

指数形式更新学习率：

$$\epsilon=\epsilon_0 \cdot \gamma^t, \quad t=0,1,2,\cdots$$

其中 $\epsilon$ 表示学习率，$t$ 表示 epoch 。

## SequentialLR

```python
lr_scheduler.SequentialLR(optimizer, schedulers, milestones, last_epoch=-1, verbose=False)
```

`milestones` 是一个索引列表，按从小到大排序，epoch 每次到达其中一个元素值时，调用对应的 `schedulers` 中的一个 `sheduler` ，并执行 `step(0)` 修改学习率，这里 `step` 函数参数为 `0` 表示对应的 `scheduler` 的 `last_epoch` 从 0 开始迭代。

注意：`milestones` 数量比 `schedulers` 数量小 1 。epoch 到达 `milestones` 第一个元素值之前，使用 `schedulers` 的第一个调度器进行迭代更新。到达 `milestones` 第一个元素值时，使用  `schedulers` 的第二个调度器，以后依次这样。
