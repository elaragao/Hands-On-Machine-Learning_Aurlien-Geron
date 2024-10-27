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
## Similarity Features
## Gaussian RBF Kernel
## SVM Classes and Computational Complexity





# SVM Regression





# Under the Hood of Linear SVM Classifiers





# The Dual Problem


## Kernelized SVMs
