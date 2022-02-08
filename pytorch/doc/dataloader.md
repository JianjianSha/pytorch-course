# 数据加载

PyTorch 数据加载的核心是 `DataLoader` 类，支持以下特性：

1. 映射风格和迭代风格的数据集，即，可根据 key（下标索引） 获取数据样本或者迭代获取数据样本。
2. 自定义数据加载顺序（顺序，随机等）
3. 自动分批
4. 单/多进程加载数据
5. memory pinning

以上特性可以通过 `DataLoader` 的构造器参数配置指定，构造器签名为
```python
torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,smapler=None,batch_sampler=None,num_workers=0,collate_fn=None,pin_memory=False,drop_last=False,timeout=0,worker_init_fn=None,multiprocessing_context=None,generator=None,*prefetch_factor=2,persistent_workers=False)
```


## 数据集类型

`DataLoader` 最重要的参数就是数据集了，毕竟有了数据集才能加载数据。有以下两种数据集。

### 映射风格的数据集
基类是 `torch.utils.data.Dataset` 。

这类数据集，其实现了 `__getitem__()` 和 `__len__()`，分别表示根据 key （下标索引）获取数据，以及获取数据集的大小。例如 `dataset[idx]` 获取数据目录中第 `idx` 个图像数据以及对应的标签。

### 迭代风格的数据集
基类是 `torch.utils.data.IterableDataset` 。

这类数据集实现了 `__iter__()` ，通过调用 `iter(dataset)`，可以返回一个数据流，其中数据可以来自数据库或者远程服务器，甚至是实时产生的日志。

## 数据加载顺序和采样器

对于迭代风格的数据集，数据加载顺序由用户在 `__iter__()` 中决定，在这个方法中可以实现按块读取以及动态批大小。

我们重点考虑 映射风格的数据集。 `torch.utils.data.Sampler` 类可以指定数据加载过程中 key（下标索引）的顺序，例如在 SGD 中，`Sampler` 可以随机重排下标索引，然后每一次 yield 一个样本或 mini-batch，从而实现 “随机”梯度下降。

根据 `DataLoader` 的 `shuffle` 参数，可以自动创建一个 顺序的或者是洗牌的 取样器，也可以通过 `sampler` 参数采用一个自定义的取样器。

`Sampler` 返回的是下标索引（int 类型），根据下标索引，在通过数据集的 `__getitem__()` 得到数据。

一个可以返回下标索引列表的自定义取样器，可以通过参数 `batch_sampler` 传递。

```{note}
sampler 和 batch_sampler 只能用在映射风格的数据集上，不能用在迭代风格的数据集上，因为 sampler 返回的是下标索引，而迭代风格数据集不能通过 key 获取数据。
```

## 批（非批）加载数据

### 自动分批
构造器参数 `batch_size` 非 `None` 时，`DataLoader` 一次产生一个表示 minibatch 的 Tensor，而不是单个样本数据，这个 Tensor 的第一维就表示 batch 。

根据 sampler 提供的下标索引取得一个数据样本列表后，`collate_fn` 参数指定的函数，用于校正样本数据为 batches。例如

```python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

如果在迭代风格数据集上处理，那么大概是这个样子，
```python
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```

### 关闭自动分批
有时候，我们想在 dataset 中直接按批读取数据，例如，从数据库中中分批读取数据，有时候我们想读取单个样本数据，这时候需要关闭 DataLoader 的自动分批功能。

当 `batch_size` 和 `batch_sampler` 均为 `None` 时，自动分批功能被关闭。dataset 中的样本直接被 `collate_fn` 处理。

示例：
```python
# 映射风格的数据集
for index in sampler:
    yield collate_fn(dataset[index])

# 迭代风格的数据集
for data in iter(dataset):
    yield collate_fn(data)
```

### collate_fn

**关闭自动分批：**  `collate_fn` 应用于单个数据样本，此时仅仅是将 NumPy 数组转为 PyTorch tensor。

**开启自动分批：**  `collate_fn` 应用于样本列表。下面着重讨论这种情况。

假设每个数据样本是一个 3-channel 的图像以及一个数值型分类标签，即，dataset 的每个 element 为 `(image, class_index)`，那么 `collate_fn` 将这样的元组列表转为单个元组，包括 batched image tensor 和 batched class label tensor。`collate_fn` 有以下特性：

1. 前置增加一个维度作为 batch 维（first dim）
2. 将 NumPy array 和 Python 数值转为 PyTorch tensor
3. 保留数据结构。例如，列表中每个样本为 dict，那么函数返回仍是一个 dict，具有相同的 key，但是 value 被打包成 batched tensor（如果无法被打包成 Tensor，那么打包成 list 类型）

### 单/多进程加载数据

默认使用单进程加载数据，可以通过设置 `num_workers` 为一正整数，从而使用多进程加载。

单进程加载适用于数据集不大的情况，或者为了跟踪调试。如果数据集非常大，那么数据加载耗时太长，影响整个计算过程。

多进程加载模式下，当调用 `enumerate(dataloader)` 时，`num_worker` 个工作进程被创建，此时 `dataset`, `collate_fn` 和 `worker_init_fn` 被传给每个 worker 进行初始化。

可以使用 `torch.utils.data.get_worker_info()` 获取工作进程信息。

对于映射风格的数据集，主进程负责使用 `sampler` 产生下标索引，然后发给工作进程。

对迭代风格的数据集，每个工作进程都有一个 dataset 的备份（相对于主进程 dataset 而言），简单的多进程加载必然会导致加载到重复数据。

### Memory Pinning
当数据位于 pinned memory 时，宿主（CPU）到 GPU 的数据拷贝会很快。

上面的介绍已经包含了大多数构造器参数，所以对几个尚未涉及到的参数统计介绍，

`drop_last`：是否丢弃最后一个不完整的 batch（batch size 小于 `batch_size`）

`timeout`：工作进程加载数据的超时时间

`worker_init_fn`：工作进程初始化函数，函数输入为工作进程 id ，范围是 `[0, num_worker-1]`。

`generator`：随机数生成器

`prefetch_factor`：每个工作进程事先加载的样本数量。

`persistent_workers`：bool 类型。在数据集被使用完毕后，是否不关闭工作进程

