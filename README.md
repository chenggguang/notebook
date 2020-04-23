[toc]

# 关于这个仓库

这个仓库是我学习NLP相关知识的笔记，将结合一些示例，做成一个学习NLP的入门级教程。本仓库的代码或者公式文字若存在描述不清的地方或者错误的地方，还请直接指出。我期望在您的帮助下把这个笔记仓库做成更好。

所有内容由jupyter lab结合一个Python内核编写，所以，若果您有兴趣读下去，那么请clone或者fork本仓库，运行一个jupyter环境浏览。

笔记的参考内容包括但不限于：

- 李航<<统计学习方法>>第二版
- 邱锡鹏[<<神经网络与深度学习>>](https://nndl.github.io/)
- B站[白板推导](https://www.bilibili.com/video/BV1aE411o7qd)系列
- 知乎上的一些专栏或者回答

# 知识点

如果需要直接查看整个仓库包含了那些文件可以参数[index](index), 或者在终端中使用tree工具。

## 工具

工欲善其事，必先利其器。掌握基础的工具才能让你实现想法的时候游刃有余

- Numpy 

- Pandas

- Matplotlib

- [PyTorch](https://pytorch.org)

  [链接](code/pytorch)只指向的code/pytorch目录，该目录下有六个示例文件。涵盖了PyTorch中的一些基础概念，若需要更多信息，请查看pytorch官网。

## 数学基础

|               方法                | 代码示例或者说明 |
| :-------------------------------: | :--------------: |
| [矩阵求导](math/derivative.ipynb) |                  |
|   [softmax](math/softmax.ipynb)   |                  |
|  [Likehood](math/likehood.ipynb)  |                  |
| [Smoothing](math/smoothing.ipynb) |                  |


## 机器学习



[EM](ml/em.ipynb)是一种思想，在无监督的情况下对参数进行估计。

[Regularization](ml/regularization.ipynb)是机器学习和深度学习算法都需要处理的问题，目的是防止模型过拟合。

### 回归类问题

|                  模型 OR 算法                   |                                             example |
| :---------------------------------------------: | ---------------------------------------------------: |
| [Linear Regression](ml/linear_regression.ipynb) | [pytorch exmaple](code/ml/1_linear_regression.ipynb) |
|             [LASSO](ml/lasso.ipynb)             |          [Coordinate Descent](code/ml/6_lasso.ipynb) |
|  [Ridge Regression](ml/ridge_regression.ipynb)  |                    [解析式法](code/ml/7_ridge.ipynb) |

### 分类问题

|                    模型 OR 算法                     |                        example                         |
| :-------------------------------------------------: | :-----------------------------------------------------: |
| [Logistic Regression](ml/logistic_regression.ipynb) | [pytorch example](code/ml/2_logistic_regression.ipynb/) |
|         [Naive Bayes](ml/naive_bayes.ipynb)         |      [pandas example](code/ml/3_naive_bayes.ipynb)      |
|       [Support Vector Machine](ml/svm.ipynb)        |       [sklearn api example](code/ml/4_svm.ipynb)        |

​				

### 序列标注问题

|               模型 OR 算法               | example                                                     |
| :--------------------------------------: | ------------------------------------------------------------ |
|   [Hidden Markov Model](ml/hmm.ipynb)    | [numpy example](code/ml/5_hmm.ipynb)                         |
| [Conditional Random Field](ml/crf.ipynb) | [LSTM + CRF](http://pytorch123.com/FifthSection/Dynamic_Desicion_Bi-LSTM/) |

## 深度学习

|              模型 or 算法               |                           example                            |
| :-------------------------------------: | :----------------------------------------------------------: |
|      [前馈神经网络](dl/fnn.ipynb)       |           [numpy实现简单FNN](code/dl/1_fnn.ipynb)            |
|      [卷积神经网络](dl/cnn.ipynb)       | [Pytorch官网示例总结](code/pytorch/2_try_neural_network.ipynb) |
|      [循环神经网络](dl/rnn.ipynb)       |                                                              |
|     [长短期记忆网络](dl/lstm.ipynb)     |       [PyTorch实现简单示例](code/pytorch/6_LSTM.ipynb)       |
|   [Transformer](dl/transformer.ipynb)   | [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)推荐必读 |
| [Evalution Method](dl/evalution.ipynb)  |                                                              |
| [参数优化](dl/parameter_optimize.ipynb) |                                                              |
|           [GRU](dl/gru.ipynb)           |                                                              |
|  [参数初始化](dl/parameter_init.ipynb)  |                                                              |
|              [网络归一化]               |                                                              |
|              [超参数优化]               |                                                              |



## NLP

|                模型 OR算法                 |   ExAMPLE    |
| :----------------------------------------: | :----------: |
|                   [分词]                   |              |
| [Language Model](nlp/language_model.ipynb) |              |
|     [PerPlexity](nlp/perplexity.ipynb)     |              |
|               [Sequence Tag]               |              |
|              [Word Embedding]              |              |
|                 [fastText]                 |              |
|                  [GloVe]                   |              |
|    [编辑距离](nlp/edit_distance.ipynb)     | 在同一文件内 |
|                                            |              |
|                 [文本分类]                 |              |
|                 [信息抽取]                 |              |
|                 [文本摘要]                 |              |
|                 [seq2seq]                  |              |



# 安装包

- Numpy
- Pandas
- Matplotplot
- Pytorch
- scikit-learn