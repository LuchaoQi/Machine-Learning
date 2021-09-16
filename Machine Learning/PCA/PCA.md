[TOC]







### [Why are principal component scores uncorrelated?](https://stats.stackexchange.com/questions/153928/why-are-principal-component-scores-uncorrelated)





### [Why log-transforming the data before performing principal component analysis?](https://stats.stackexchange.com/questions/164381/why-log-transforming-the-data-before-performing-principal-component-analysis)

### Does mean centering or feature scaling affect a Principal Component Analysis?

https://stats.stackexchange.com/a/69159/272857

| Before Normalization                     | After Normalization                      |
| ---------------------------------------- | ---------------------------------------- |
| ![](https://i.stack.imgur.com/lS3wA.png) | ![](https://i.stack.imgur.com/rqYHf.png) |

https://sebastianraschka.com/faq/docs/pca-scaling.html

<mark>We need to normalize / standardize data and make sure features are on the same unit. </mark>

https://builtin.com/data-science/step-step-explanation-principal-component-analysis



### [EigenValue Decomposition, EigenVectors, EigenValue](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)



![](https://i.loli.net/2020/08/13/yPjYXZG7qJLvuA3.png)



### [<mark>What are principal component scores?</mark>](https://stats.stackexchange.com/questions/222/what-are-principal-component-scores)

https://stats.stackexchange.com/a/282/272857



In NHANES project, our PC scores can be calculated

```R
load('analyticData.rda')

subject_scores = MINdata %>% select(-SEQN)
PC_scores = scale(subject_scores, pca_model$center, pca_model$scale) %*% pca_model$rotation

# or 
# prcomp(DF, scale = FALSE)$x
pca_model$x
```



[How to project a new vector onto PCA space?](https://stats.stackexchange.com/questions/2592/how-to-project-a-new-vector-onto-pca-space)



```R
# perform principal components analysis
pca <- prcomp(data) 

# project new data onto the PCA space
scale(newdata, pca$center, pca$scale) %*% pca$rotation 
```





### <mark>Loadings</mark>

loadings, or essentially vectors, are the linear combinations of raw axe directions to form the principal component axe (non-unit)

e.g. the first principal component axe = x * loading of x + y * loading of y = (x,y)  %*% (loading of x, loading of y)

magnitude of loadings reflect the weight of different raw axes to form the new principal component axe

https://stats.stackexchange.com/a/470545


$$
\text { Loadings }=\text { Eigenvectors } \cdot \sqrt{\text { Eigenvalues }}
$$



principal component scores = raw data %*% loadings = rotated data = projected vectors on the principal component direction

```R
loadings = prcomp(d)$rotation[,'PC1']
PC1_scores = d %*% ( prcomp(d)$rotation[,'PC1']) # d %*% loadings
PC1_scores = prcomp(d)$x[,'PC1']
```









### [PCA on a categorical variable?](https://piazza.com/class/k1zer6tg6s04bb?cid=443)

> Unfortunately, no. PCA is really only appropriate for real-valued features. Even ordinal features (categories with order, e.g. first, second, third, ...) shouldn't use PCA because there is a "distance" implied. Only use PCA for xi∈R.
>
> They mention correspondence analysis, which is a method I haven't looked into myself, but here's a link to the paper information (it's from 1935): https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/connection-between-correlation-and-contingency/D3A75249B56AF5DDC436938F1B6EABD1
>
> They also mention multi-factor analysis: https://www.sciencedirect.com/science/article/pii/016794739490135X?via%3Dihub

https://stackoverflow.com/a/40801571/11964524

> I disagree with the others.
>
> While **you can use PCA on binary data** (e.g. one-hot encoded data) that does not mean it is a good thing, or it will work very well.
>
> PCA is desinged for *continuous* variables. It tries to minimize variance (=squared deviations). The concept of squared deviations breaks down when you have binary variables.
>
> So yes, you can use PCA. And yes, you get an output. It even is a least-squared output - it's not as if PCA would segfault on such data. It works, but it is just much less *meaningful* than you'd want it to be; and supposedly less meaningful than e.g. frequent pattern mining.





## <mark>Intuition: why we use covariance matrix and why we use eigenpairs?</mark>

> PCA - using eigen-decomposition of covariance matrix



### Prerequisite

**Definition of covariance matrix**

[Covariance matrix - Wikipedia](https://en.wikipedia.org/wiki/Covariance_matrix#Definition)
$$
S \text{(covariance matrix)}=\frac{1}{N} \sum_{i=1}^{N}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{\top}=\frac{1}{N} X^{\top} H X

\\
H_{N} \text{(centering matrix)} = I_{N}-\frac{1}{N} 1_{N} 1_{N}^{T}
$$
**Linear Algebra**

[Transpose - Wikipedia](https://en.wikipedia.org/wiki/Transpose#Properties)
$$
\begin{aligned}A^{T}B &=-5\\ A^{T}B&=\left[ -5\right] \\ B^{T}A&=\left[ -5\right] ^{T}=\left[ -5\right] \end{aligned}
$$

**Projected distance** 

[Dot product - Wikipedia](https://en.wikipedia.org/wiki/Dot_product#Scalar_projection_and_first_properties)



$$
\vec{a} \cdot \vec{b} = \vert \vec{a}\vert \cdot \vert \vec{b}\vert \cdot cos\theta = \vec{a} \cdot \vec{b}^T = \vec{b} \cdot \vec{a^T}= \vec{a}^T \cdot \vec{b}
\\
\begin{aligned}proj &=\left\| \overrightarrow{a}\right\| \cdot \cos \theta \\ &=\dfrac{\overrightarrow{a}\cdot \overrightarrow{b}}{\left\| \overrightarrow{b}\right\| }\\ &=\overrightarrow{a}\cdot \overrightarrow{\widehat{b}}\end{aligned}
$$


Let's get started

**After centering the data, we want to maximize the projected variance**

> PCA is a distance-based algorithm

Our target is to maximize the sum of squared projections as some of projections might be negatives

Assuming the unit projected direction/vector is $u_1$ and we denote $S$ as covariance matrix

$$
\begin{aligned} J &=\frac{1}{N} \sum_{i=1}^{N}\left(\left(x_{i}-\bar{x}\right)^{\top} u_{1}\right)^{2} \quad \text { s.t. }\left|u_{1}\right|=1 \text { s.t. } u_{1}^{\top} u_{1}=1 \\ &=\sum_{i=1}^{N} \frac{1}{n} u_{1}^{\top}\left(x_{i}-\bar{x}\right) \cdot\left(x_{i}-\bar{x}\right)^{\top} u_{1} \\ &=u_{1}^{\top}\left(\sum_{i=1}^{N} \frac{1}{n}\left(x_{i}-\bar{x}\right) \cdot\left(x_{i}-\bar{x}\right)^{\top}\right) u_{1} \\ &=u_{1}^{\top} \cdot S \cdot u_{1} \end{aligned}
$$




so we need to solve an optimization problem


$$
\left\{\begin{array}{l}{\hat{u}_{1}=\arg \max u_{1}^{\top} \cdot S \cdot u_{1}} \\ {\text { s.t. } u_{1}^{\top} u=1}\end{array}\right.
$$


with [Lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier) and [matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus)

$$
\begin{array}{c}{\mathcal{L}\left(u_{1}, \lambda\right)=u_{1}^{\top} S u_{1}+\lambda\left(1-u_{1}^{\top} u\right)} \\ {\frac{\partial \mathcal{L}}{\partial u_{1}}=2 S \cdot u_{1}-\lambda \cdot 2 u_{1}=0} \\ {S u_{1}=\lambda u_{1}}\end{array}
$$
BAM! For covariance matrix S, $u_1$ is eigenvector and $\lambda$ is eigenvalue



### Example

centering the data before PCA

e.g. we have five 2-D data points after centering and the array is represented as numpy does here: [numpy.cov — NumPy v1.21 Manual](https://numpy.org/doc/stable/reference/generated/numpy.cov.html)

> m : array_like
>
> A 1-D or 2-D array containing multiple variables and observations. Each row of *m* represents a variable, and each column a single observation of all those variables. Also see *rowvar* below.

$$
A = \begin{pmatrix}
-1 & -1 & 0 & 2 & 0\\
-2 & 0 & 0 & 1 & 1
\end{pmatrix}
$$

then we calculate covariance matrix

$$
\begin{align*}
C = \frac{1}{5} * \begin{pmatrix}-1 & -1 & 0 & 2 & 0 \\
-2 & 0  & 0 & 1 & 1\end{pmatrix} * \begin{pmatrix}-1 & -2 \\ -1 & 0 \\ 0 & 0 \\ 2 & 1 \\ 0 & 1\end{pmatrix} = \begin{pmatrix} \frac{6}{5} & \frac{4}{5} \\ \frac{4}{5} & \frac{6}{5}\end{pmatrix}
\end{align*}
$$

eigenvalues/eigenvectors


$$
\begin{array}{c}{\lambda_{1}=2, \lambda_{2}=2 / 5} \\ {c_{1}\left(\begin{array}{c}{1} \\ {1}\end{array}\right), c_{2}\left(\begin{array}{c}{-1} \\ {1}\end{array}\right)}\end{array}
$$


sort the eigenvectors based on eigenvalues and cast them into a new matrix so eigenvectors in row
$$
P=\left(\begin{array}{cc}{1 } & {1 } \\ {-1} & {1 }\end{array}\right)
$$


standardized eigenvector matrix


$$
P=\left(\begin{array}{cc}{1 / \sqrt{2}} & {1 / \sqrt{2}} \\ {-1 / \sqrt{2}} & {1 / \sqrt{2}}\end{array}\right)
$$


which has the following feature


$$
P C P^{\top}=\left(\begin{array}{cc}{1 / \sqrt{2}} & {1 / \sqrt{2}} \\ {-1 / \sqrt{2}} & {1 / \sqrt{2}}\end{array}\right)\left(\begin{array}{cc}{6 / 5} & {4 / 5} \\ {4 / 5} & {6 / 5}\end{array}\right)\left(\begin{array}{cc}{1 / \sqrt{2}} & {-1 / \sqrt{2}} \\ {1 / \sqrt{2}} & {1 / \sqrt{2}}\end{array}\right)=\left(\begin{array}{cc}{2} & {0} \\ {0} & {2 / 5}\end{array}\right)
$$

use the first line (`first eigenvector/largest eigenvalue`)

project the original data points into new coordinate
$$
Y=\left(\begin{array}{cc}{1 / \sqrt{2}} & {1 / \sqrt{2}}\end{array}\right)\left(\begin{array}{ccccc}{-1} & {-1} & {0} & {2} & {0} \\ {-2} & {0} & {0} & {1} & {1}\end{array}\right)=\left(\begin{array}{cccc}{-3 / \sqrt{2}} & {-1 / \sqrt{2}} & {0} & {3 / \sqrt{2}} & {1 / \sqrt{2}}\end{array}\right)
$$
P.S. 我发现很多网上的例子都不自己算一下就直接复制 像上面计算Y简单的矩阵运算都是错的

DONE! the projected 1-D points are shown as below



![](https://i.loli.net/2020/06/05/Zhzjl2tHmn43vEU.png)





## Code

### R

R code for the above example

```R
# all features on top
d = matrix(c(-1,-1,0,2,0,-2,0,-0,1,1),ncol = 2)
d
pca_res = prcomp(d)
pca_res
```

```R
     [,1] [,2]
[1,]   -1   -2
[2,]   -1    0
[3,]    0    0
[4,]    2    1
[5,]    0    1
Standard deviations (1, .., p=2):
[1] 1.5811388 0.7071068

Rotation (n x k) = (2 x 2):
           PC1        PC2
[1,] 0.7071068  0.7071068
[2,] 0.7071068 -0.7071068
```

`Rotation` or `loadings` in princomp() is the matrix of **standardized** variable loadings (i.e., a matrix whose columns contain the eigenvectors).

sdev	
the standard deviations of the principal components (i.e., the square roots of the eigenvalues of the covariance/correlation matrix, though the calculation is actually done with the singular values of the data matrix). 

```R
> eigen(d_cov)$values ^ 0.5
[1] 1.5811388 0.7071068
```

So you can see `Standard deviations` are the list of square root of eigenvalues

```R
d_cov = cov(d)
d_cov
eigen(d_cov)


     [,1] [,2]
[1,]  1.5  1.0
[2,]  1.0  1.5

eigen() decomposition
$values
[1] 2.5 0.5

$vectors
          [,1]       [,2]
[1,] 0.7071068 -0.7071068
[2,] 0.7071068  0.7071068
```

<mark>i.e. if you want to know the rotated data</mark>

```R
prcomp(d)$x 

# PC scores e.g. data point (-1,-2) is represented as
#  -2.1213203 * PC1    + 0.7071068 * PC2
# https://stats.stackexchange.com/a/282/272857

            PC1        PC2
[1,] -2.1213203  0.7071068
[2,] -0.7071068 -0.7071068
[3,]  0.0000000  0.0000000
[4,]  2.1213203  0.7071068
[5,]  0.7071068 -0.7071068


# rotated data on PC1
res_PC1 = d %*% ( prcomp(d)$rotation[,'PC1'])

t(t(d) - rowMeans(t(d))) %*% eigen(cov(d))$vectors
```



```R
prcomp(d)$x
# 结果横过去看就是 data point = f(PCs)
# 用新的坐标PC来表示数据
prcomp(d)$rotation
# 结果竖过来看就是 PC = f(features)
# 表示怎么用原来的坐标体系表达新的PC坐标体系
# 横过来看就是怎么用PC来返回来表达feature(x/y/z)坐标
```



<mark>Note it is different from the Example b/c  we are using formula</mark>

$cov(x,y) = \frac{1}{N-1} Z^TZ$

instead of

$cov(x,y) = \frac{1}{N} Z^TZ$

So in the example it is $1/5 * Z^TZ$ while in code it is $1/4 * Z^TZ$

> ?prcomp
>
> Unlike princomp, variances are computed with the usual divisor N - 1.





## Resources



[<mark>Principal Component Analysis (sebastianraschka.com)</mark>](https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#a-summary-of-the-pca-approach)

[heucoder/dimensionality_reduction_alo_codes: 特征提取/数据降维:PCA、LDA、MDS、LLE、TSNE等降维算法的python实现 (github.com)](https://github.com/heucoder/dimensionality_reduction_alo_codes)

[机器学习-白板推导系列(五)-降维（Dimensionality Reduction）笔记 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/326074168)

[【机器学习】【白板推导系列】【合集 1～23】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1aE411o7qd?p=23)

[How to Calculate Principal Component Analysis (PCA) from Scratch in Python (machinelearningmastery.com)](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)





[Principal Component Analysis in 3 Simple Steps](https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html)

[Eigenvalues and Eigenvectors, Imaginary and Real](https://www.youtube.com/watch?v=8F0gdO643Tc&ab_channel=PhysicsVideosbyEugeneKhutoryansky)

[A STEP BY STEP EXPLANATION OF PRINCIPAL COMPONENT ANALYSIS](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

https://www.bilibili.com/video/av70839977?p=24

[STATQUEST Principal Component Analysis (PCA) Step-by-Step](https://youtu.be/FgakZw6K1QQ)

https://www.cnblogs.com/pinard/p/6239403.html

http://blog.codinglabs.org/articles/pca-tutorial.html

https://www.zhihu.com/question/30094611

[eigenvalues/eigenvectors](http://setosa.io/ev/eigenvectors-and-eigenvalues/)

[Understanding the Role of Eigenvectors and Eigenvalues in PCA Dimensionality Reduction.](https://medium.com/@dareyadewumi650/understanding-the-role-of-eigenvectors-and-eigenvalues-in-pca-dimensionality-reduction-10186dad0c5c)















