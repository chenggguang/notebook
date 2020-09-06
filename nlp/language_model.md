[toc]
# Language Model

本笔记基本参考[知乎文章](https://zhuanlan.zhihu.com/p/52061158)

- 标准定义：对于语言序列$w_1, w_2, \ldots, w_n$，语言模型就是计算该序列的概率，即 $P(w_1, w_2, \ldots, w_n)$。
- 从机器学习的角度来看：语言模型是对语句的**概率分布**的建模。
- 通俗解释：判断一个语言序列是否是正常语句，即是否是**人话，**例如$P(\text{how are you}) > P(\text{you are how})$。

## Statistic Language Model

首先，由chain rule可以得到
$$
P(w_1,w_2,\ldots, w_n) = P(w_1)P(w_2|w_1)\ldots P(w_n|w_{n-1}\ldots w_1)
\tag{1}
$$

在统计语言模型中，使用极大似然估计来计算每个词出现的概率。
$$
\begin{aligned}
P(w_i|w_{i-1}\ldots w_1) 
&= \frac{C(w_1, \ldots, w_{i-1}, w_i)}{\sum_w C(w_1, \ldots, w_{i-1}, w)}\\
&\overset{?}{=} \frac{C(w_1, \ldots, w_{i-1}, w_i)}{C(w_1, \ldots, w_{i-1})}
\end{aligned}
\tag{2}
$$

公式(2)中$C(\cdot)$代表第二个通常是不等的，第二个等式的值是明显小于第一个等式的，但是通常使用第二个等式来计算(是为了LM能够对变长序列进行处理)。

对于任意长的自然语言语句，根据极大似然估计直接计算$P(w_i|w_{i-1}\ldots w_1)$不现实。参数量过大，而且出现数据稀疏。

### N-gram
此时，就要引入Markov Assumption,即当前词只依赖前$n-1$个词。可以得到
$$
P(w_i|w_{i-1}\ldots w_1)  = P(w_i|w_{i-n+1}, \ldots, w_{i-1})
\tag{3}
$$

那么，根据公式(3)

- N=1 Unigram
  $$
  P(w_1, \ldots, w_n) = \displaystyle\prod^n_{i=1}P(w_1)
  \tag{4}
  $$

- N=2 Bigram
  $$
  P(w_1, \ldots, w_n) = \displaystyle\prod^n_{i=1}P(w_i|w_{i-1})
  \tag{5}
  $$

- N=3 Trigram
  $$
  P(w_1, \ldots, w_n) = \displaystyle\prod^n_{i = 1}P(w_i|w_{i-1}w_{i-2})
  \tag{6}
  $$

其中，当$n \gt 1$时，为了使句首词的条件概率有意义，
需要给原序列加上一个或多个**起始符**$<start>$可以说起始符的作用就是为了**表征句首词出现的条件概率**。
当然，在一些教程上，句首的条件概率直接使用$P(w_1)$，但是通过参考[HMM](../ml/hmm.ipynb)中参数$\pi$的统计方法。
还是认为采用在句首添加$<start>$的方式更为合理。
同理还需要在句末添加一个$<end>$，这个结束符的目的是让LM能够处理变长序列。
使得模型在L的所有序列上的概率和为1。同时，也是使得公式(2)中第二个等式成立的条件。

当然，我们训练模型的语料库不能穷尽语言中所有的序列，那么总会有在训练语言中没有出现过的grams。
这就要涉及到[平滑](../math/smoothing.ipynb)处理以及为登录词的处理。我们可以在语言添加一个$\text{<UNK>}$用来代替训练预料中没有出现的词的位置。该处同样经过平滑处理。

## Neural Netword Language Model

可以参考[NNLM](https://www.jianshu.com/p/a02ea64d6459)

神经网络语言模型中的神经网络指的是[FNN](../dl/fnn.ipynb)，[RNN](../dl/rnn.ipynb)、[CNN](../dl/cnn.ipynb)等等基础的网络，在未来，可能还会有GNN的语言模型出现。

### Based on FeedForward Neural Network

基于前馈神经网络的语言模型借鉴了N-gram的思想，当前词同样只依赖于前$n-1$个词。其模型训练过程如下。假设词典大小为$|V|$，那么有将前$n-1$个词的one-hot向量叠加为一个size为$1\times |V|$的向量。设置隐藏层的参数size为$|V|\times H$,输出层的参数为$H\times|V|$，然后经过softmax激活之后与当前词的$ont-hot$求交叉熵。

其具体的实现类似与CBOW

利用神经网络去建模当前词出现的概率与其前 n-1 个词之间的约束关系。很显然这种方式相比 N-gram 具有更好的**泛化能力**，**只要词表征足够好**。从而很大程度地降低了数据稀疏带来的问题。但是这个结构的明显缺点是仅包含了**有限的前文信息**。

### Based on Recurrent Neural Network

FNN不能解决时序一来的问题，而RNN天生就是为了解决时序问题的。所以使用RNN建立语言模型是一件理所应当的事情。

回顾RNN的运算公式，表达如下
$$
h_t = f(Uh_{t - 1} + Wx_t + b)
$$
RNNLM的操作是将RNN每一个时刻的输出经过一个FeedForward层，映射到词表大小的向量空间，在将FeedForward层的输出经过Softmax归一化，得到当前时刻为词表中各个词的概率。值得注意的是，它会对输入序列进行预处理.
在序列的开头加上$\text{<START>}$，那么也就是说，实际上使用"<START> + sequence"去预测"sequence + <EOS>"

### Based on Convolution Neural Network

使用CNN建立语言模型和RNN以及N-gram的语出方法都是类似的。
我们假设一个单词经过Embedding之后维度为$D$，以为卷积的高度为$D$, 宽度为$N$, 那么我们需要在文本序列的开头填充$N - 1$个$<START>$。

我们使用$\otimes$作为卷积运算符，那么CNNLM的运算过程为
$$
H_l(X) = (U\otimes X)\odot \sigma(W\otimes X)\\
Y = \text{softmax}(VH_l(X))
$$
如果模型的输出为$L\times |V|$，对应的每一行就是序列对应时刻为词表中各词的概率。

## 语言模型的评估

评价一个模型的好坏通常使用准确率这种指标，但是对于一个语言模型，需要测试其准确率需要大量的测试数据，而且耗费运算力，所以，通常是使用[Perplexity](perplexity.md)
