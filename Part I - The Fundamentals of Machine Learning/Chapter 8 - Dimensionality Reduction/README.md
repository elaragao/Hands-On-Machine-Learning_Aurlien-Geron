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







<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->
# PCA


**PCA** or **Principal Component Analysis** is the most popular dimensionality reduction algorithm. In simple and summarized terms, the hyperplane that is closest to the data is identified and then the data is projected onto the hyperplane.





<!------------------------------------------------------->
<!------------------------------------------------------->
## Preserving the Variance                                              



<!------------------------------------------------------->
<!------------------------------------------------------->  
## Principal Components                                                 



<!------------------------------------------------------->
<!------------------------------------------------------->  
## Projecting Down to d Dimensions                                      



<!------------------------------------------------------->
<!------------------------------------------------------->  
## Using Scikit-Learn                                                   



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




