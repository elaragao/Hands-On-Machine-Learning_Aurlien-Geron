Support Vector Machines (SVMs) are a versatile machine learning model capable of performing linear and nonlinear classifications, regression, and even outlier detection. They perform best on small to medium-sized sets of complex data.

# Linear SVM Classification

One way to explain the SVM is from the image below. The image on the right shows three lines representing random predictions. The solid lines demonstrate that they can separate the two groups, but the dashed line demonstrates a poor performance in this regard.

The image on the right represents an **SVM classifier**. The central solid line represents the **Decision Boundary**, separating the classes and staying as far away as possible from the closest training instances. You can see dashed lines separating the points of the closest training instances from each other, and the space between these lines is called _Street_. The SVM classifier, the solid line, fits into the widest possible street between the classes, which is a _large margin classification_.

[Image]

It is important to note that adding more training instances that are not on this street does not affect the SVM. It is also important to note that SVM is sensitive to scaling.

## Soft Margin Classification

Strict margin enforcement is called _hard margin classification_, and has two major problems: it only works with linearly separable data and it is sensitive to outliers. It ends up being sensitive to problems such as outliers in places that prevent correct classification (for example, an outlier between different instances) or prevent generalization (the outlier being too close to another instance).

[image]

A more flexible model is called _soft margin classification_, and its goal is to maintain a balance between a large street and limit _margin violations_ (instances that are either in the middle of the street or on the wrong side). When creating these models using Scikit-Learn, it is possible to specify hyperparameters such as the regularization hyperparameter called `C`. Reducing `C` makes the street wider, but more prone to margin violations, and increasing it makes the street have fewer violations, but the street is smaller.


# Nonlinear SVM Classification
In many cases, Linear SVM is not enough. Adding some new features is an alternative, which in some cases can allow linear separation. The image below shows the feature $x_{1}$ on the left, and on the right the addition of the feature $x_{2} = x_{1} ^{2}$, which results in something linearly separable.

[Image]

The Scikit-Learn library has classes capable of both assisting in the application of SVC and in data generation, in this case, the `make_moons` class, a dataset for training binary classifications. The pipeline below demonstrates one way to use this dataset.


```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = make_pipeline(
	PolynomialFeatures(degree=3),
	StandardScaler(),
	LinearSVC(C=10, max_iter=10_000, random_state=42)
)

polynomial_svm_clf.fit(X, y)

```

## Polynomial Kernel
For the application of polynomial methods in SVM, a useful mathematical technique is the _Kernel Trick_, which allows the same result as a high degree polynomial without implementing a high degree. It is worth noting that low degree polynomials may not be good for complex data, and high degree polynomials may make the algorithm very slow. The code below uses a third degree polynomial kernel:

```python
from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(StandardScaler(),
	SVC(kernel="poly", degree=3, coef0=1, C=5))

poly_kernel_svm_clf.fit(X, y)
```
 The hyperparameter `coef0` controls how much the model is influenced by high-degree terms versus low-degree terms.
 
> [!NOTE]
> Overfitting can be caused by a high polynomial degree, and it is necessary to reduce the degree. The same goes for when Underfitting occurs due to low polynomial degree.

## Similarity Features

This technique consists of adding features using a similarity function, which measures how similar instances are to a specific _landmark_. It works in a similar way to the geographic partitioning seen in Chapter 2.

The simplest way to choose landmarks is to create a reference point in each instance of the data set. Because of the many dimensions created, it tends to be possible to partition all the data linearly. The disadvantage is that if there are a very large number of instances, there will be too many features, and therefore, it will be a slow process.


## Gaussian RBF Kernel

Operates similarly to _Kernel Trick_ for polynomials, but for Similarity Features. For this, the _Gaussian RBF Kernel_ is used, as demonstrated in the code below:

```python
rbf_kernel_svm_clf = make_pipeline(StandardScaler(),
SVC(kernel="rbf", gamma=5, C=0.001))
rbf_kernel_svm_clf.fit(X, y)
```

The model has two hyperparameters, gamma ($\gamma$) and C. The hyperparameter $\gamma$ controls the range of influence of each data point, that is, the larger it is, the shorter ... [Image]

There are other kernels, but they are more specific, such as some used for strings, or Levenshtein distance, for example.

> [!TIP]
> How to choose which kernel to use? In case of a large training set, `LinearSVC` due to its speed. If the training set is not large, it is valid to use Kernelized SVMs, starting with Gaussian RBF Kernel, and then, polynomial. In cases of specific sets, it is valid to try more specific models.


## SVM Classes and Computational Complexity

For this brief analysis, consider the number of training instances as $m$ and the number of features as $n$.

The `LinearSVC` class operates at a time complexity of approximately O(m × n).

The `SVC` class operates at a time complexity between O(m$^{2}$ × n) and O(m$^{3}$ × n), i.e., it becomes much slower as the number of instances increases.

It is worth mentioning that the `SGDClassifier` class can produce similar data to `LinearSVC` with a good time scale.



# SVM Regression






<!-- https://www.analyticsvidhya.com/blog/2021/06/support-vector-machine-better-understanding/#h-svm-regression -->
# Under the Hood of Linear SVM Classifiers





# The Dual Problem


## Kernelized SVMs
