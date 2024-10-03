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

Compared to the previous chapter, evaluating a classifier is a little more laborious than a regressor. Some ways to evaluate this will be presented.



## Measuring Accuracy Using Cross-Validation

Just as we saw in [Chapter 2], using Cross-Validation will separate the training set into k-folds, in the current case, 3 folds, training the model k times and holding a different fold for evaluations.

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

When observing the accuracy, you can observe a very high value in all folds. This occurs because, even with the errors, the minority of values ​​are different from 5, which makes the evaluation model give the false impression that the model is excellent. To demonstrate this, it is possible to use the `DummyClassifier`, which classifies all values ​​with the most frequent data, in this case, different from 5, and will still obtain a high accuracy (in this case, above 90%).

Thus, it is possible to observe that accuracy is not the best way to evaluate classifiers, especially when dealing with _skewed datasets_ (when one value is much more frequent than others).


## Confusion Matrices

A confusion matrix counts the number of times instances A and B were classified for all possible A/B pairs. To calculate the confusion matrix, it is first necessary to have a set of predictions so that it can be compared to the correct labels. It is important to note that for now the test set should be ignored.

In addition to the scores, k-fold validation can return the predictions made for each fold. This is ideal for analysis at this stage of the project.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)
```

This will return a 2×2 matrix. Each row represents a real class, while columns represent the _predicted class_




<div class="block-language-tx"><table>
<thead>
	<tr>
		<th style="text-align:center" colspan="2">Confusion Matrix</th>
	</tr>
</thead>
	
<tbody>
	<tr>
		<td>
			<b>Negative</b>
		</td>
		<td>
			<b>False Positive</b>
		</td>	
	</tr>
</tbody>

<tbody>
	<tr>
		<td>
			<b>False Negative</b>
		</td>
		<td>
			<b>Positive</b>
		</td>
	</tr>
</tbody>

</table>
</div>





- **Negatives**: Values ​​correctly classified as negative.
- **False Positive**: Or type I error, where a negative value is classified as positive.
- **False Negative**: Or type II error, where a positive value is classified as negative.
- **Positive**: Values ​​correctly classified as positive.

In addition to the confusion matrix, it is necessary to obtain more concise metrics, which are **Precision** and **Recall**.

Precision: Indicates the proportion of correct positive predictions in relation to the total positive predictions made by the model.

```math
Precision = \frac{TP}{TP + FP}
```
Recall: Also called sensitivity or true positive rate (TPR), it measures the model's ability to find all positive samples in the dataset.

```math
Recall = \frac{TP}{TP + FN}
```



## Precision and Recall

Scikit-Learn provides function to compute the values ​​of Precision and Recall:

```python
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)   # == TP / (TP + FP)
recall_score(y_train_5, y_train_pred)      # == TP / (TP + FN)
```

One way to use Precision and Recall for analysis is to use the F1 metric, which consists of a _harmonic mean_ between them, giving better results for F1 only when both Precision and Recall are high and favors when they are similar.

```python
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
```

```math
F1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}} = 2 × \frac{Precision × Recall}{Precision + Recall} = \frac{TP}{TP + \frac{FN + FP}{2}}
```

It is important to highlight for future analyses that, because it favors when Precision and Recall are similar, it is not ideal for certain cases, in metrics that favor one and neglect the other. For example:

- **Precision more important**: In the case of detecting safe videos for children, it is preferable to reject videos that are safe (**Low Recall**), but keep the safe videos (**High Precision**).
- **Recall more important**: When training something to detect shoplifters, it is preferable to have a higher false alarm rate (**Low Precision**), but to detect almost all cases where a shoplifting occurred (**High Recall**).

It is not possible to have a high rate for both, increasing Precision reduces Recall, and vice versa. This is called the _Precision/Recall Trade-off_


 
## The Precision/Recall Trade-Off
In order to understand how _Trade-Off_ works, it is important to first understand how the `SGDClassifier` operates. For each instance, a score is calculated based on the `Decision Function`. If the chosen digit is **greater** than the _threshold_, it will be categorized as a positive instance, and if it is **less** than the _threshold_, it will be categorized as a negative instance. Below we will compare three positions of the **Decision Threshold**:


![tradeoff_precisionrecall](https://github.com/user-attachments/assets/0d639c83-13a6-4f15-b7b2-6f147a2a2780)


- Position on the Left: The position between the number "9" and "5" contains 6 TP numbers, and 2 FP numbers.
- Position in the Center: The position between two "5" numbers, contains 4 TP numbers (real "5") on the right, and one FP (a "6" number), returning a Precision of 80% and a Recall of 67%.
- Position on the Right: The position between "6" and "5" contains 3 TP numbers and 3 FN numbers.

> [!IMPORTANT]
> It is then possible to observe that increasing the threshold makes Precision greater and Recall smaller, and vice versa.

Although it is not possible to define the threshold directly, it is possible, through the decision scores, to change this value, calling the `decision_function()` method instead of the `predict()` method, which will return the score of the instances and then set the threshold:


```python
y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0
y_some_digit_pred = (y_scores > threshold)
```

To decide which threshold to use, assuming that we have cross-validation, we must define the return of the scores and not the predictions. From there, it is possible to use the `precision_recall_curve()` function to then calculate the possible limits between Precision and Recall.

```python
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

```

Assuming that you decide to opt for a threshold where you have 90% Precision, an alternative is to use the NumPy `argmax()` function to obtain the lowest limit that this is found, obtaining the first index of the maximum value. Then, with this, it is possible to make predictions with the `predict()` method.

```python
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]

y_train_pred_90 = (y_scores >= threshold_for_90_precision)
```

> [!CAUTION]
> Even though there is a high Precision, a low Recall, such as the one obtained at 48%, is not a good thing.



## The ROC Curve
The _Receiver Operating Characteristic_ (ROC) curve is another method for binary classifiers. The ROC curve compares the Recall (or _True Positive Rate_) with _False Positive Rate_ (FPR), also called _fall-out_, which is the rate of negative instances classified as positive. To plot the curve, you can obtain the values ​​through the code:

> [!NOTE]
> _True Negative Rate_ (TNR), or _specificity_ can also explicitly specify the FPR value, where $FPR = 1 - TNR$

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

> [!IMPORTANT]
> There is also a trade-off in the relationship between TPR (Recall) and FPR. The higher one, the lower the other.

To estimate the classifiers, one way is to compare the _Area Under the Curve_ (AUC). A perfect classifier will have an area of ​​1, and a random one will have an area of ​​about 0.5. This can be estimated from the code:

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
```

>[!IMPORTANT]
> How to choose between the ROC curve and the Precision/Recall (PR) curve?
>- The PR would be chosen in situations where there are **few positive classes**, or when you care more about **avoiding false positives**. Example: Credit card fraud detection, where there are few cases of fraud (positive) and many cases of non-fraud (negative).
>- ROC is chosen in more generic situations, where the values ​​are more balanced, **considering both false positives and false negatives**, but it does not detect the problem of false positives in rare classes well.

After using the `SGDClassifier` model, it is important to try other models, such as `RandomForestClassifier`, where their PR curves and F1 score will be compared. The `precision_recall_curve()` function, which was used from the _scores_ in the SGD model, which cannot be obtained in a _RandomForestClassificer_ because it does not have a `decision_function()` method, instead it has the `predict_proba()` method, which returns the class probabilities of each instance.

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
```

To make Score predictions, codes such as:

```python
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba ≥ 50%

f1_score(y_train_5, y_pred_forest)
roc_auc_score(y_train_5, y_scores_forest)
```


# Multiclass Classification

_Multiclass Classifiers_ or _Multinomial Classifiers_ are capable of classifying into more than two classes. Some examples of classifiers are:
- Binary: `SGDClassifier`,`SVC`
- Multiclass: `LogisticRegression`,`RandomForestCLassifier`,`GaussianNB`

Although binary classifiers are not capable of distinguishing between more than two classes, it is possible to use them for this purpose. For this, there are two most popular strategies that will be exemplified using the dataset we are currently studying:
- _One Versus The Rest_ (OvR) or _One Versus All_ (OvA): Individually classifies each digit (i.e., 0-detector, 1-detector, 2-detector...) and from the classifier that has the highest score obtains its decision.
- _One Versus One_ (OvO): These are classifiers that individually distinguish each digit from the other (e.g. "0" and "1", "0" and "2"...). Trains a total of $N \times (N - 1)/2$ classifiers, in our case, 45.

>[!CAUTION]
> Looking at the two strategies for binary classifiers to operate as multiclass, you might be wondering "In which case would it be advantageous to use _One Versus One_. In the case of some algorithms, such as SVM (Support Vector Machines), which scale poorly with increasing training set, this strategy provides the advantage of being a large set of small training sets, and not a large training set, allowing better use of this algorithm. But for most binary classification, the _One Versus The Rest_ method is still preferred.

Scikit-Learn detects whether it is trying to use a binary classification algorithm, and selects _OvO_ or _OvR_ depending on which algorithm. Assuming you want to test the SVM algorithm for classification:

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])  # y_train, not y_train_5

svm_clf.predict([some_digit]) # Predict the value of variable some_digit

some_digit_scores = svm_clf.decision_function([some_digit]) # Return the scores of some_digit

class_id = some_digit_scores.argmax() # Return ID of highest score.
```

When the classifier is trained, the `classes_` attribute stores a list of the target classes. In most cases, it is necessary to compare the ID obtained in the previous code with the target label.

```python
svm_clf.classes_

svm_clf.classes_[class_id]
```

If you wish, you can force the use of the strategy you want, by importing the test class, be it `OneVsOneClassifier` or `OneVsRestClassifier`.

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])

len(ovr_clf.estimators_) # Return number of estimators
```

It is possible, in an analogous way, to do the same for SGD, which will be explained better in the .ipybn files





# Error Analysis

Assuming that, at this stage, the promising model has been found, it is then necessary to analyze the errors. Initially, it is useful to view a _Confusion Matrix_ using the `confusion_matrix()` function. The `ConfusionMatrixDisplay.from_predictions()` function will plot the graph to aid visualization. Since, on several occasions, the data for one value is in greater number than others, it is useful to normalize the matrix using the `normalize = True` argument.

> [!NOTE]
> Unlike the matrix seen previously, this one will have dimensions of 10 × 10, it is worth noting that the most relevant data is in the main diagonal.

```python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3) # Predict Values

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred) # Plot Confusion Matrix

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
 normalize="true", values_format=".0%") # Plot normalized Confusion Matrix

plt.show() 
```

>[!CAUTION]
> Be careful, because matrices are not symmetric. For example, assuming that the number "5" was classified x times as number "8", this does not mean that the number "8" was classified x times as number "5".

It is also possible to analyze the weights of the errors. These matrices can be categorized by row or by column. By categorizing by row, we see the percentage of errors where the images identified as the value of the **x** axis were actually the **y** axis. By categorizing by column, we indicate the percentage of the values ​​classified in **x** that are actually **y**. The code below demonstrates the plot of the matrix:

```python
sample_weight = (y_train_pred !=y_train)

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")

plt.show()
```

> [!WARNING]
> This type of matrix indicates the percentages of the **TOTAL ERRORS**, and not of all the values. It is observable since, in the main matrix, all the values ​​that correspond to themselves are 0%.

Performing the analyses of these types of confusion matrices is useful for reducing possible errors. Some possible options when detecting errors in a specific class are:
- Obtaining more training data (which resembles the value giving the error, an FP) for the classifier to distinguish from the real ones.
- Improving the algorithm to have closed loops in the class giving the error.
- Pre-processing the images so that the errors stand out more and are correctly detected.
- Analyzing individual errors.


# Multilabel Classification



# Multioutput Classification
