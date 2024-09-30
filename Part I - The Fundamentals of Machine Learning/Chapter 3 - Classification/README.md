# Classification



# MNIST

In this chapter, we will use a dataset called MNIST, which consists of 70,000 images with handwritten labels. Scikit-Learn provides several functions for downloading popular datasets, including MNIST, through the `sklearn.datasets` package. The `sklearn.datasets` package contains three types of functions:

- `fetch_*`: such as `fetch_openml()` for downloading real-world datasets. `fetch_openml()` returns input as DataFrames.

- `load_*`: for loading small data.

- `make_*`: for generating fake datasets useful for testing. These are usually generated as tuples (X,y) containing data for input and targets.

Other datasets are returned as `sklearn.utils.Bunch` objects, which are dictionaries:

- DESCR: Description of the dataset.

- data: Input data, usually as a 2D NumPy array.
- target: labels, usually as a 1D NumPy array.

The way to access the dataset is demonstrated in the code below:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
```

> [!TIP]
> When dealing with images, it is preferable not to use DataFrames. To do this, set `as_frame=False`.


In this set there are 70,000 images, each with 784 features, because each image has a dimensionality of 28×28 pixels, and these have intensities from 0 (white) to 255 (black). To more successfully analyze the images, it is necessary to reshape them into a 28×28 matrix, using `cmap="binary"` to obtain a color map:

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
	image = image_data.reshape(28, 28)
	plt.imshow(image, cmap="binary")
	plt.axis("off")
```

The MNSIT dataset already contains a separation between training and testing data, with the first 60,000 for training, which is already shuffled.

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

# Training a Binary Classifier
To facilitate the initial understanding of the classification, a binary classifier will be trained first, only identifying whether or not the values ​​correspond to the digit "5". It is then necessary to separate the labels only for values ​​that correspond to the number 5:

```python
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
```

An interesting classifier to start with is the _Stochastic Gradient Descent_ (SGD) using the `SGDClassifier` class from SciKit-Learn.

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])
```





# Performance Measures

## Measuring Accuracy Using Cross-Validation

## Confusion Matrices

## Precision and Recall
	
## The Precision/Recall Trade-Off

## The ROC Curve




# Multiclass Classification




# Error Analysis




# Multilabel Classification



# Multioutput Classification
