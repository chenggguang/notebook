[toc]

# 关于这个仓库

这个仓库是我学习NLP相关知识的笔记，将结合一些示例，做成一个学习NLP的入门级教程。本仓库的代码或者公式文字若存在描述不清的地方或者错误的地方，还请直接指出。我期望在您的帮助下把这个教程做成更好。

教程采用jupyter lab结合一个Python内核编写，所以，若果您有兴趣读下去，那么请clone或者fork本仓库，运行一个jupyter环境浏览。

# 知识点

## 数学基础

[矩阵求导](math/derivative.ipynb)、是我们应该掌握的知识，这是机器学习和深度学习的数学基础。

[softmax](math/softmax.ipynb)在数据归一化已经分类问题中有着广泛的应用本笔记将其与交叉熵结合，并推到了其求导公式。

[Likehood][math/likehood.ipynb]在很多机器学习算法的公式推导中，优化的目标都是一个MLE或者MAP的表达式。

[Smoothing](math/smoothing.ipynb)  //todo 为语言模型添加链接

## 机器学习



[EM](ml/em.ipynb)

[Regularization](ml/regularization.ipynb)

### 回归类问题

|                  模型 or 算法                   |                                          examplee |
| :---------------------------------------------: | ------------------------------------------------: |
| [Linear Regression](ml/linear_regression.ipynb) | [pytorch exmaple](code/1_linear_regression.ipynb) |
|             [LASSO](ml/lasso.ipynb)             |          [Coordinate Descent](code/6_lasso.ipynb) |
|  [Ridge Regression](ml/ridge_regression.ipynb)  |                    [解析式法](code/7_ridge.ipynb) |



### 分类问题

|                    模型 or 算法                     |                       EXamples                       |
| :-------------------------------------------------: | :--------------------------------------------------: |
| [Logistic Regression](ml/logistic_regression.ipynb) | [pytorch example](code/2_logistic_regression.ipynb/) |
|         [Naive Bayes](ml/naive_bayes.ipynb)         |      [pandas example](code/3_naive_bayes.ipynb)      |
|       [Support Vector Machine](ml/svm.ipynb)        |       [sklearn api example](code/4_svm.ipynb)        |

​				



### 序列标注问题

|               模型 Or 算法               | ExAMPles                          |
| :--------------------------------------: | --------------------------------- |
|   [Hidden Markov Model](ml/hmm.ipynb)    | [numpy example](code/5_hmm.ipynb) |
| [Conditional Random Field](ml/crf.ipynb) | None                              |



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