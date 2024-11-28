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



<!------------------------------------------------------->
<!------------------------------------------------------->  
## Choosing the Right Number of Dimensions                              



<!------------------------------------------------------->
<!------------------------------------------------------->  
## PCA for Compression                                                  



<!------------------------------------------------------->
<!------------------------------------------------------->  
## Randomized PCA                                                       



<!------------------------------------------------------->
<!------------------------------------------------------->  
## Incremental PCA                                                       











<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->
# Random Projection                                                     









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


