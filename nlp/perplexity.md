# Perplexity

perplexity是用于评测Language Model性能的一个指标。表现为一个数值，通常数值越小，说明LM的性能越好。即模型预测分布越接近自然语言的真实分布。

要理解perplexity需要引入一些香农的香农信息论中的概念。

### Entropy

1948年香农提出的信息熵解决了信息的量化度量的问题，信息熵用于描述信息的不确定度。假设$X$为随机变量，$P(X)$为该变量取某一个值得概率，那么
$$
H(X) = E[I(x_i)] = -\displaystyle\sum P(x_i)log_2(P(x_i))
$$



### Cross Entropy

在信息论中，交叉熵表示两个概率分布P、Q(P表示真实分布，Q表示非真实分布)在相同一组事件中使用Q分布来表示P分布所需要的平均编码长度.

对于离散型分布
$$
H(P, Q) = \displaystyle\sum P(i)log_2(\frac{1}{Q(i)})
$$
对于连续型分布
$$
H(P, Q) = -\int_XP(x)log_2(Q(x))dx
$$

### Entropy rate

对于索引可数的随机过程 $\chi = X_1,X_2,\ldots,X_n$ ,其熵率定义为:
$$
H(\chi) = \lim_{n\to\infty}\frac{1}{n}H(X_1, X_2, \ldots,X_n)\tag{1}
$$

对于独立同分布的序列，各随机变量的熵相等，则显然有：
$$
H(\chi) = \lim_{n\to\infty}\frac{1}{n}\displaystyle\sum_{i = 1}^{n}H(X_i) = H(X_i)\tag{2}
$$
所以熵率依其定义可粗略地类比于随机过程中 *per-random_variable* *entropy*。比如，考虑这样一台打字机，假设可输出$m$个等可能的符号。因此打字机可产生长度为$n$的共$m^n$个序列，并且都等可能出现。那么对于该打印机
$$
H(\chi) = log_2 m\tag{3}
$$

### Entropy of Language and Cross Entropy of Language

在语言模型中，对于语言$L$中的长度为的$n$的单词序列单词$S = {w_1, w_2, \ldots,w_n}$序列的熵的计算公式为 
$$
H(w_1, \ldots, w_n) = -\sum P(S)log_2P(S)\tag{4}
$$

那么在单词序列熵的基础上，根据熵率的定义，可粗略的定义 *per-word entropy*：
$$
\frac{1}{n}H(w_1, \ldots, w_n) = -\frac{1}{n}\sum P(S)log_2P(S)\tag{5}
$$

而若将语言 ![[公式]](https://www.zhihu.com/equation?tex=L) 视作随机过程，则有语言$L$的熵率：
$$
H(L) = -\lim_{n\to\infty}\frac{1}{n}\displaystyle\sum_{w_1,\ldots,w_n\in L}P(w_1,\ldots,w_n)log_2P(w_1,\ldots,w_n)\tag{6}
$$

对于上式，根据Shannon-McMillan-Breiman theorem又可以做如下近似：
$$
H(L) = -\lim_{n\to\infty}\frac{1}{n}log_2P(w_1,\ldots,w_n)\tag{7}
$$
那么，计算在语言$L$熵的交叉熵率为：
$$
H(P, Q) = -\lim_{n\to\infty}\frac{1}{n}
\displaystyle\sum_{L}P(w_1, \ldots,w_n)log_2Q(w_1, \ldots, w_n)\tag{8}
$$
上式可以近似为
$$
H(P, Q) = -\lim_{n\to\infty}\frac{1}{n}
\displaystyle\sum_{L}log_2Q(w_1, \ldots, w_n)\tag{9}
$$

### Perplexity

通过上面的内容，足以引入perplexity了。假设$P(S)$为语言模型计算出S的概率，那么对于整个语言L

- Log-likehood
  $$
  LL(L) = \displaystyle\sum_{S\in L}log_2P(S)\tag{10}
  $$

- Per-word log-likehood
  $$
  WLL(L) = \frac{1}{\sum_{S\in L}|S|}\displaystyle\sum_{S\in L}log_2P(S)\tag{11}
  $$

- Per-word cross likehood
  $$
  H(L) = - \frac{1}{\sum_{S\in L}|S|}\displaystyle\sum_{S\in L}log_2P(S)\tag{12}
  $$

- Perplexity 
  $$
  PPL(L) = 2^{H(L)} = 2^{-WLL(L)}\tag{13}
  $$

### Shannon-McMillan-Breiman theorem

//todo
