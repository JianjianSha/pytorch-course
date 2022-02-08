# 优化器
优化器主要用于更新模型参数，例如 SGD 中的 
$$\theta_{t+1} = \theta_t - \epsilon \cdot d \theta_t$$

除了上述这种更新方式，还有很多其他更新方式的优化器。 PyTorch 中优化器基类为

```python
torch.optim.Optimizer(params, defaults)
```

- `params`（iterable）：指定被更新的参数。例如更新一个模型的参数 `model.parameters()`，这个参数必须是 Tensor 集合，或者 字典集合，如果是 Tensor 集合，那么在类内部，也会将其改造成 字典集合，
    ```python
    # 将 iterable 转为一个 list，如果已经是 list，那么保持不变
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]
    ```
    事实上，参数会被分组，然后每组单独更新，例如各组的学习率 `lr`， 可能会不同，当然对于其他参数更新方式，还有其他超参数，例如 `momentum` 动量系数，`weight_decay` 权值衰减率 等。

- `defaults`（dict）：一个包含优化选项默认值的字典。给每个参数组设置默认值，默认 key 和 默认值均由 `defaults` 指定
    ```python
    for name, default in self.defaults.item():
        # 这里 param_group 表示某一组参数，例如 {'params': xx}
        param_group.setdefault(name, default)
    ```

一个参数组 group，其类型为字典类型 dict，除了 `params` 这个 key 对应模型参数之外，还有其他 key，例如 `lr`（学习率），`momentum`（动量系数），`weight_decay`（权重衰减率，即正则项 $\frac 1 2 \lambda \mathbf w^2$ 中的系数 $\lambda$），`nesterov`（bool 类型，指示是否使用 nesterov 更新方式），这些更新超参数（group 中除 `params` 之外的 key），通常都是通过参数 `defaults` 参数设置）。

`Optimizer` 类的方法：

- `add_param_group(self, param_group)`：手动添加一个参数组

- `state_dict(self)`：返回状态字典，
    ```python
    return {
        'state': packed_state,
        'param_groups': param_groups # 参数组集合
    }
- `load_state_dict(self, state_dict)`：加载 `Optimizer` 类的状态字典，参数 `state_dict` 必须是类的 `state_dict(self)` 返回的对象。

- `zero_grad(self, set_to_none=False)`：所有被优化的参数 Tensor 的梯度设置为 0 。通常在后向传播之前，或者某次迭代更新之后，执行 `zero_grad`（当然，也有特殊情况，需要保持参数的梯度，此时不要执行 `zero_grad`）。如果某时刻之后不需要计算梯度，则可以通过 `set_to_none=True` 将梯度设置为 None，从而节省内存。

- `step(self, closure)`：执行一次参数更新。`closure` 是一个闭包函数，通常用于模型的前向传播计算，并返回最终损失。

几个重要字段：

- `self.param_groups`：保存了参数组列表，每个参数组为 dict 类型，包含这组所有参数 `params`，以及超参数学习率、动量等。

- `state` 优化器的内部状态。例如使用 动量 momentum 的更新方式，如下两式所示，那么需要保存中间状态，即动量 $v$ 。

    $$v_{t+1}=\mu \cdot v_t + d\theta_t$$
    $$\theta_{t+1}=\theta_t - \epsilon \cdot v_{t+1}$$

    （当动量系数 $\mu=0$ 时，退化为无动量形式的更新）

## 具体类型
具体类型的优化器包括
```
Adadelta
Adagrad
Adam
AdamW
SparseAdam
Adamax
ASGD
NAdam
RAdam
RMSprop
Rprop
SGD
```
部分优化器的具体说明可参考 [PyTorch.optim.SGD 数学表示](https://jianjiansha.github.io/2020/01/02/pytorch/optim_SGD/) ，这里就不涉及数学公示了。

# 示例

构造示例
```python
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

以上示例中，仅有一个参数组。当然，可以构造多个参数组，

```python
optim.SGD([
    {'params':model.base.parameters()},
    {'params':model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)
```
上面，构造了两个参数组，其中第二个参数组特别指定了 `lr=1e-3`，而第一个参数组则使用默认的 `lr=1e-2`。

```{note}
如果需要将 model 通过 `.cuda()` 转为 GPU 上执行，那么必须在构造优化器之前就执行 `.cuda()` 操作，否则优化器中的参数与模型的参数不同，导致无法更新模型
```

一个常用的优化器工作的示例，
```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

**step(closure) 提供闭包函数参数**

```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```

