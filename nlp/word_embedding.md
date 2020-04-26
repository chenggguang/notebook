# Word Embedding

本文参考[word2vec教程](https://www.zybuluo.com/Dounm/note/591752#213-huffman树构造算法理解与证明)、[PyTorch教程](https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter10_natural-language-processing/10.1_word2vec)

自然语言是一套用来表达含义的复杂系统。在这套系统中，词是表义的基本单元。顾名思义，词向量是用来表示词的向量，也可被认为是词的特征向量或表征。把词映射为实数域向量的技术也叫词嵌入(word embedding)。近年来，词嵌入已逐渐成为自然语言处理的基础知识。

## 为什么不用one-hot呢

One-hot虽然可以在符号上等效地认为是一个词，且它的构造非常地容易，但是它有如下的缺点。

1. one-hot词向量无法准确表达不同词之间的相似度，不能使用向量余弦公式计算其相似度
2. 对于大的词汇表，极大的增加了模型的参数量，而且one-hot是稀疏的。模型不容易收敛

word2vec 的出现解决了这个问题。word2vec是训练[神经概率语言模型](nlp/language_model.ipynb)的中间产物。此笔记将介绍CBOW(Continuous Bag Of Word)以及Skip-gram。

## CBOW

连续词袋模型与跳字模型类似。与跳字模型最大的不同在于，连续词袋模型假设基于某中心词在文本序列前后的背景词来生成该中心词。在同样的文本序列$w_1, w_2, w_3, w_4, w_5$里，以$w_3$作为中心词，且背景窗口大小为2时，连续词袋模型关心的是，给定背景词$w_1, w_2, w_4, w_5$生成中心词$w_3$的条件概率，也就是
$$
P(w_3|w_1, w_2, w_4, w_5)\tag{1}
$$

因为连续词袋模型的背景词有多个，我们将这些背景词向量取平均。设$v_i \in R^d$和$u_i\in R^d$分别表示词典中索引为$i$的词作为背景词和中心词的向量。设中心词$w_c$在词典中索引为$c$，背景词$w_{o_1}, \ldots, w_{o_{2m}}$在词典中索引为$o_1, \ldots, o_{2m}$，那么给定背景词生成中心词的条件概率
$$
P(w_c |w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\exp\left(\frac{1}{2m}u_c^T(v_{o_1,\ldots, v_{o_{2m}}})\right)}{\sum_{i\in V}\exp\left(\frac{1}{2m}u_i^T(v_{o_1,\ldots, v_{o_{2m}}})\right)}\tag{2}
$$

为了让符号更加简单，我们记为$W_o = \{w_{o_1}, \ldots, w_{o_{2m}}\}$, 且$\bar{v}_o = \frac{\left(v_{o_1} + \ldots + v_{o_{2m}} \right)}{2m}$, 那么上式可以简写成
$$
P(w_c|W_o) = \frac{\exp(u_c^T\bar{v})}{\sum_{i\in V}\exp(u_i^T\bar{v})}\tag{3}
$$

给定一个长度为$T$的文本序列，设时间步$t$的词为$w^{(t)}$，背景窗口大小为$m$。连续词袋模型的似然函数是由背景词生成任一中心词的概率
$$
\arg \underset{\theta}{\max}\displaystyle\prod_{t - 1}^TP(w^{(t)}|w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t + 1)}, \ldots, w^{(t + m)})\tag{4}
$$


公式(4)得到的是CBOW的极大似然估计，那么等价于最小化下面公式

$$
\arg \underset{\theta}{\min}\displaystyle\sum_{t - 1}^T\log P(w^{(t)}|w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t + 1)}, \ldots, w^{(t + m)})\tag{5}
$$

注意到
$$
\log P(w_c| W_o) = u_c^T\bar{v}_o - \log \left(\Sigma_{i\in V}\exp(u_i^T\bar{v}) \right)\tag{6}
$$

通过微分，我们可以计算出上式中条件概率的对数有关任一背景词向量$v_{o_i}\ (i = 1, \ldots, 2m)$的梯度
$$
\begin{aligned}
\frac{\partial \log P(w_c| W_o)}{\partial v_{o_i}}
=& \frac{1}{2m}\left(u_c - \sum_{j \in V} \frac{\exp(u_j^T\bar{v}_o)u_j}{\sum_{i\in V}\exp(u_i^T\bar{v})} \right)\\
=& \frac{1}{2m}\left(u_c - \sum_{j \in V} P(w_j|W_ou_j) \right)\\
\end{aligned}\tag{7}
$$

## Skip-gram

Skip-gram假设基于某个词来生成它在文本序列周围的词。举个例子，假设文本序列是$w_1, w_2, w_3, w_4, w_5$。以$w_3$作为中心词，设背景窗口大小为2。Skip-gram所关心的是，给定中心词$w_3$，生成与它距离不超过2个词的背景词$w_1, w_2, w_4, w_5$的条件概率，即
$$
P(w_1, w_2, w_4, w_5|w_3)\tag{8}
$$
假设给定中心词的情况下，背景词的生成是相互独立的，那么上式可以改写成
$$
P(w_1|w_3)\cdot P(w_2|w_3)\cdot P(w_4|w_3)\cdot P(w_5|w_3)\tag{9}
$$

在skip-gram中，每个词被表示成两个$d$维向量，用来计算条件概率。假设这个词在词典中索引为$i$，当它为中心词时向量表示$v_i\in R^d$,而背景词向量表示为$u_i\in R^d$, 设中心词$w_c$在词典中索引为$c$, 背景词$w_o$在词典中索引为$o$，给定中心词生成背景词的概率可以通过对向量内积做softmax运算而得到
$$
P(w_o|w_c) = \frac{u_o^Tv_c}{\sum_{i \in V}\exp(u_i^Tv_c)}\tag{10}
$$

假设给定一个长度为$T$的文本序列，设时间步$t$的词为$w^{(t)}$。假设给定中心词的情况下背景词的生成相互独立，当窗口大小为$m$时， skip-gram的似然函数即给定任一中心词生成所有背景词的概率
$$
\arg\underset{\theta}{\max}\displaystyle\prod_{t = 1}^{T}\prod_{-m\le j \le m, j\not= t}P(w^{(t + j)}| w^t)\tag{11}
$$

skip-gram的参数是每个词所对应的中心词向量和背景词向量。训练中我们通过最大化似然函数来学习模型参数，即最大似然估计。这等价于最小化以下损失函数：
$$
\arg\underset{\theta}{\min} - \displaystyle\sum_{t = 1}^{T}\sum_{-m\le j \le m, j\not= t}\log P(w^{(t + j)}| w^t)\tag{12}
$$

如果使用随机梯度下降，那么在每一次迭代里我们随机采样一个较短的子序列来计算有关该子序列的损失，然后计算梯度来更新模型参数。梯度计算的关键是条件概率的对数有关中心词向量和背景词向量的梯度。根据定义，首先看到
$$
\log P(w_o|w_c) = u_o^Tv_c - \log\left(\sum_{i \in V}\exp(u_i^Tv_c)\right)\tag{13}
$$

通过微分，我们可以得到上式中$v_c$, $u_j$的梯度
$$
\begin{aligned}
\frac{\partial\log P(w_o|w_c)}{\partial v_c}
=& u_o - \frac{\sum_{j \in V}\exp(u_j^Tv_c)u_j}{\sum_{i \in V}\exp(u_i^Tv_c)}\\
=& u_o - \sum_{j \in V}\left(\frac{\exp(u_j^Tv_c)}{\sum_{i \in V}\exp(u_i^Tv_c)}\right)u_j\\
=& u_o - \displaystyle\sum_{j\in V}P(w_j|w_c)u_j
\end{aligned}\tag{14}
$$

若$j = o$
$$
\frac{\partial\log P(w_o|w_c)}{\partial u_j}
= v_c - P(w_j|w_c)v_c^T\tag{15}
$$

否则
$$
\frac{\partial\log P(w_o|w_c)}{\partial u_j}
=  - P(w_j|w_c)v_c^T\tag{16}
$$

它的计算需要词典中所有词以$w_c$为中心词的条件概率。有关其他词向量的梯度同理可得。训练结束后，对于词典中的任一索引为$i$的词，我们均得到该词作为中心词和背景词的两组词向量$v_i$和$u_i$。在自然语言处理应用中，一般使用skip-gram的中心词向量作为词的表征向量。

## 负采样

选取负样本需要按照一定的概率分布，Word2vec的作者们测试发现**最佳的分布是$\frac34$次幂的**Unigram Distribution。
$$
p(w) = \frac{[\text{count}(w)]^{\frac34}}{\sum_{i\in V}[\text{count}(w_i)]^{\frac34}}\tag{17}
$$
以上述概率从词典中选择一些词作为当前中心词的负样本。
