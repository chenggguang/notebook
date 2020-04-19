[toc]

# 关于这个仓库

这个仓库是我学习NLP相关知识的笔记，将结合一些示例，做成一个学习NLP的入门级教程。本仓库的代码或者公式文字若存在描述不清的地方或者错误的地方，还请直接指出。我期望在您的帮助下把这个教程做成更好。

教程采用jupyter lab结合一个Python内核编写，所以，若果您有兴趣读下去，那么请clone或者fork本仓库，运行一个jupyter环境浏览。

# 知识点

## 数学基础

[矩阵求导](math/derivative.ipynb)、是我们应该掌握的知识，这是机器学习和深度学习的数学基础。

[softmax](math/softmax.ipynb)在数据归一化已经分类问题中有着广泛的应用本笔记将其与交叉熵结合，并推到了其求导公式。

[Likehood][math/likehood.ipynb]在很多机器学习算法的公式推导中，优化的目标都是一个MLE或者MAP的表达式。

//todo 正则化添加文件链接



## 机器学习

[线性回归](ml/linear_regression.ipynb)是回归类问题中的一个经典算法，原理简单，在[代码](code/1_linear_regression.ipynb)上也及容易实现。同属于回归问题的还有[LASSO](ml/lasso.ipynb)和[Ridge](ml/ridge_regression.ipynb)，他们是普通线性回归经过L1和L2[正则化](ml/regularization.ipynb)后的模型。

说到回归，就不得不提一下[逻辑回归](ml/logistic_regression.ipynb), 当然，逻辑回归不是回归，而是解决分类的问题。

[朴素贝叶斯](ml/naive_bayes.ipynb)

[Smoothing]

[支持向量机](ml/svm.ipynb)是机器学习算法中，分类效果极好的一种，可以调用sklearn[实现](code/4_svm.ipynb)

[EM](ml/em.ipynb)

[隐马尔可夫模型](ml/hmm.ipynb)在序列标注任务上应用广泛。

[条件随机场]



## 深度学习

[前馈神经网络]

[卷积神经网络]

[循环神经网络]

[长短期记忆网络]

[Evalution Method]

[GRU]

[Dropout处理]

[梯度下降法]

[动量法]

[AdaGrad]

[RMSprop]

[AdaDelta]

[Adam]

## NLP

[分词]

[Language Model]

[PerPlexity]

[Sequence Tag]

[Word Embedding]

[fastText]

[GloVe]

[编辑距离]

[近义词]

[文本分类]

[信息抽取]

[文本摘要]

[seq2seq]

# 安装包

- Numpy
- Pandas
- Matplotplot
- Pytorch
- scikit-learn