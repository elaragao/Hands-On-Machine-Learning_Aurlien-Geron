# Dimensionality Reduction

In Machine Learning problems, it is not unlikely that there will be thousands, or even millions of features, which can, in addition to slowing down training, make the solutions poor in quality, which is called the **Curse of Dimensionality**.

There are processes that will be discussed in this chapter that can minimize this problem, called **Dimensionality Reduction**. This process not only speeds up training, but is also useful for data visualization.

The two main approaches to this are **Projection** and **Manifold Learning**. The three most popular techniques are **PCA**, **Kernel PCA**, and **LLE**.
<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->
# The Curse of Dimensionality


Datasets with high dimensionality, i.e., many features, cause the risk of the data set being too sparse, i.e., each training instance tends to be too far apart from each other. This makes predictions less reliable than in lower dimensions, since they have to be extrapolated more. In other words, many dimensions cause a greater risk of _overfitting_.

Although a possible solution is to increase the number of instances, having a data set with, for example, 100 features would require training instances of about the number close to the number of atoms in the universe.



<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->
# Main Approaches for Dimensionality Reduction

In the following subchapters two forms of dimensionality reduction will be briefly discussed: **Projection** and **Manifold Learning**.

<!------------------------------------------------------->
<!------------------------------------------------------->
## Projection

Given the fact that in real problems the data is usually distributed unevenly across dimensions, as seen in sets like MNIST where many features are nearly constant, while others are highly correlated. Because of this, the training instances tend to be within or close to a **subspace** of much lower dimension than in the high-dimensional space, as can be seen in the figure below, where the 3D data can be represented by the 2D hyperplane:

[Images]

It is possible to observe in the image on the left a 2D subspace of lower dimension with a high-dimensional 3D space where the data actually is. In the image on the right you can see the perpendicular projection of each instance in this subspace, obtaining a 2D data set with a lower dimensionality, now with axes $z_{1}$ and $z_{2}$ representing the coordinates of the projection in the plane.



<!------------------------------------------------------->
<!------------------------------------------------------->

## Manifold Learning


To illustrate the **Manifold**, it is useful to think of the _Swiss Roll_. A 2D _Manifold_ is a shape that can be folded and twisted in a higher dimension.

[Images]

Many dimension reduction algorithms operate by modeling the _Manifold_, called **Manifold Learning**. This is based on the **Manifold Hypothesis** (or **Manifold Assumption**), which states that most high-dimensional data sets in the real world are similar to a _Manifold_ of much lower dimension.

An example of this would be the generation of digits for the MNIST data set. If you randomly generate images, you would very rarely see anything that looks like handwritten digits. By applying constraints and reducing the degrees of freedom, this becomes dramatically more likely. So, as mentioned in the previous paragraph, constraints tend to compress the data set into a _Manifold_ of lower dimension.


>[!NOTE]
> The _Manifold Assumption_ is accompanied by the implicit assumption that the task to be performed (regression or classification) will be simpler on a non-lower dimension of the **Manifold**. The task will certainly be faster, but its performance will depend on the training set.




<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->
# PCA


**PCA** or **Principal Component Analysis** is the most popular dimensionality reduction algorithm. In simple and summarized terms, the hyperplane that is closest to the data is identified and then the data is projected onto the hyperplane.





<!------------------------------------------------------->
<!------------------------------------------------------->
## Preserving the Variance                                              

Before projecting the training set onto the lower-dimensional hyperplane, the best hyperplane is first selected. This is usually chosen based on the one that preserves the greatest variance of the original set. In this way, it loses less information than the other projections, and also minimizes the mean squared distance between the original data set and the projection onto this axis. This is the basic idea behind PCA.




<!------------------------------------------------------->
<!------------------------------------------------------->  
## Principal Components                                                 

PCA identifies the axis with the largest amount of variance. If applied to a 2D graph, the result would be a line with points on top in a 1D dimension. In the case of a 2D graph, there would be a second axis that would contain the remaining amount of variance (the sum of the variance of the axes would not necessarily be 100%, but it would be close to it). The same goes for higher dimensions, where the first PCA would have the largest amount of variance, the second PCA would have the second largest amount, and so on.

The unit vector that defines the i-th axis is called the _Principal Component_ (PC), where the first PC is $c_{1}$, the second is $c_{2}$, and so on up to $c_{n}$.
The principal components of the training set can be obtained using the SVD (_Singular Value Decomposition_) technique, which decomposes the matrix $X$ as $X = U \cdot \sigma \cdot V^{T}$, where V contains the principal components. This can be obtained using the numpy `svd()` function:

```python
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
```


<!------------------------------------------------------->
<!------------------------------------------------------->  
## Projecting Down to d Dimensions                                      


Having the PCs, it is possible to reduce the training set to $d$ dimensions by projecting it onto the hyperplane that preserves the most variance. For example, a 3D dataset would be reduced to a 2D dimension by the hyperplane defined by the two main PCs.

To perform this projection of a $d$ dimensional dataset $X_{d-proj}$ it is necessary to calculate the matrix multiplication of the set $X$ multiplied by the matrix $W_{d}$, which contains the first $d$ columns of the matrix $V$:

```math
X_{d-proj} = X \cdot W_{d}
```

The code below demonstrates the equation for the first two PCs:
```python
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
```

<!------------------------------------------------------->
<!------------------------------------------------------->  
## Using Scikit-Learn                                                   

It is possible to use Scikit-Learn to implement what was demonstrated in the subchapters above through the `PCA` class:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
```

<!------------------------------------------------------->
<!------------------------------------------------------->  
## Explained Variance Ratio                                             

To obtain the **Explained Variance Ratio** you can use the `explained_variance_ratio_` function. This is intended to demonstrate how much variance is maintained by each PC. It is common for the selected PCs to have a good part of the variance maintained, while the other unselected dimensions maintain the rest.

<!------------------------------------------------------->
<!------------------------------------------------------->  
## Choosing the Right Number of Dimensions                              

It is usually chosen to preserve about 95% of the variance, with exceptions such as using it only for visualization or when there are few dimensions. It will be exemplified using the MNIST dataset:

```python
from sklearn.datasets import fetch_openml
# [...]
pca = PCA()
pca.fit(X_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 # d equals 154

# pca = PCA(n_components=d) # It's one option, but not the best

```
When using `n_components` as a `float` between 0.0 and 1.0, it indicates the variance rate that you want to preserve:

```python
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

pca.n_components_ # Will return 154
```

Another way to obtain isos is to plot the variance as a function of the number of dimensions, where an elbow is observed in the curve, where the variance stops growing rapidly.

[Image]


Another way is when doing preprocessing for supervised learning such as classifications, it is possible to adjust the number of dimensions like any other hyperparameter (it will be better discussed in the code [Link]).

<!------------------------------------------------------->
<!------------------------------------------------------->  
## PCA for Compression                                                  

Comparing the training set before and after dimension reduction for MNIST, 95% of the variance was preserved while using only 154 features, instead of the original 784, having less than 20% of the original size.

It is possible to perform decompression by applying an inverse transform, which will not return the original data due to the 5% of information lost by variance. The mean squared distance between the original and reconstructed data is called _Reconstruction Error_. The code and equation below demonstrate how it is done:

```python
X_recovered = pca.inverse_transform(X_reduced)
```

```math
X_{recovered} = X_{d-proj} \cdot W_{d}^{T}
```

<!------------------------------------------------------->
<!------------------------------------------------------->  
## Randomized PCA                                                       

If the hyperparameter `svd_solver` is set to "randomized", a stochastic algorithm is used to perform _Randomized PCA_, which quickly finds an approximation of the first PC.

```python
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)

X_reduced = rnd_pca.fit_transform(X_train)
```

<!------------------------------------------------------->
<!------------------------------------------------------->  
## Incremental PCA                                                       



The implementation of PCA requires that the training set be trained, which causes problems in cases where online training is required, for example, or for sets that **do not fit in memory**. The **Incremental PCA** (IPCA) algorithm divides the training set into mini-batches and increases the algorithm with one mini-batch at a time.

Two ways of applying it to the same data set will be demonstrated as examples, being MNIST using 100 mini-batches. The first uses the Numpy library function `array_split()` to reduce the dimensions to 154, and it is necessary to use the `partial_fit()` method instead of `fit()`:

```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):
inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
```

A different way is to use the `memmap` class, which allows you to manipulate log arrays stored in a binary file on disk. Since the `IncrementalPCA` class uses only a small part of the array at any given time, memory is kept under control and the `fit()` method can be used:

```python
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)

inc_pca.fit(X_mm)
```







<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->
# Random Projection                                                     


The method uses a random projection into a lower-dimensional space, causing similar instances to remain similar and very different instances to remain very different. As with all forms of reduction, the more dimensions are discarded, the more information is lost and the more instances are distorted.

Choose the number of dimensions using the equation below, which determines the minimum number of dimensions to be preserved, ensuring with high probability that the distances do not change more than the given tolerance. The equation is known as the Johnson Lindenstrauss equation.




```math
d \geq \frac{4 log(m)}{\frac{1}{2} \varepsilon^{2} - \frac{1}{3} \varepsilon^{3}}
```


<details>

<summary>Equation terms</summary>

- $d$: Is the number of dimensions;
- $m$: Is the number of instances;
- $n$: Is the number of features;
- $\varepsilon$: Is the Squared Distance or the tolerance.

</details>


After obtaining the value of $d$, it is possible to create the matrix **P** of format [d,n], with Guasian distribution and mean 0 and variance $1/d$ and used to project a dataset of $n$ dimensions in $d$ dimensions. Below you can see the code to obtain the reduction:

```python
from sklearn.random_projection import johnson_lindenstrauss_min_dim
m, ε = 5_000, 0.1
d = johnson_lindenstrauss_min_dim(m, eps=ε)

n = 20_000
np.random.seed(42)
P = np.random.randn(d, n) / np.sqrt(d) # std dev = square root of variance
X = np.random.randn(m, n) # generate a fake dataset
X_reduced = X @ P.T
```


It is possible to do this through the Scikit-Learn library through the `GaussianRandomProjection` class, generating the random matrix. Using the `fit()` and `transform()` methods it is possible to perform the projection:

```python
from sklearn.random_projection import GaussianRandomProjection

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced = gaussian_rnd_proj.fit_transform(X) # same result as above
```


There is also a second transformer in this library, `SparseRandomProjection`, which uses less memory, is faster and has comparable quality. It is preferable to use this one on large data sets. It has something called **Rate r** which shows the rate of non-zero items in the random matrix called density. By default it is $1/\sqrt{2}$, and can be changed by the hyperparameter r.

In short, **Random Projection** is a simple, fast and memory-efficient method useful for dimensionality reduction.

<!------------------------------------------------------->
<!------------------------------------------------------->
<!-------------------------------------------------------> 
# LLE                                                                   









<!------------------------------------------------------->
<!------------------------------------------------------->
<!-------------------------------------------------------> 
# Other Dimensionality Reduction Techniques                              

Other dimensionality reduction methods in the Scikit-Learn library are:

- **Multidimensional Scaling** (MDS): Tries to preserve distance between instances. Comparable to _Random Projection_, but when this is better for many data, MDS is better for low-dimensional data. ` sklearn.manifold.MDS`.

- **Isomap**: Operates by creating a graph that connects instances to their closest neighbors and performs the reduction by trying to reduce the _Geodesic Discance_ of the instances (shortest path between two _nodes_). `sklearn.manifold.Isomap`

- **T-Distributed Stochastic Neighbor Embedding** (t-SNE): Tries to keep similar instances close and dissimilar instances far apart. Operates best for visualizing clusters. `sklearn.manifold.TSNE`

- **Linear Discriminat Analysis** (LDA): Operates with linear classification, where it learns the most discriminative axes of the classes to determine the hyperplanes. It distances the classes from each other and is useful before executing classification algorithms.  `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`

The image below shows the comparison between some of these methods:

[Image]


