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

<!------------------------------------------------------->
<!------------------------------------------------------->

## Manifold Learning











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




