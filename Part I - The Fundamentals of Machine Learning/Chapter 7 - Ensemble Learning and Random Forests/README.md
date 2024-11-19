#  Ensemble Learning and Random Forests 

**Ensemble Learning** is the technique of aggregating predictions from a group of predictors. It usually gives a better result than using the best individual predictor. An Ensemble Learning algorithm is called an **Ensemble Method**.

An example of this method is what happens when training a group of _Decision Trees_ classifiers, where the set gives the name of the **Random Forest** method.


<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->

# Voting Classifiers

Assuming that you have trained a few classifiers for a data set, one way to create an even better classifier is to use the **Hard Voting Classifier**, where you have the classifier with the most votes for the ensemble prediction.

Even though each classifier is a _Weak Learner_ (where it does only slightly better than a random guess), it can still be a _Strong Learner_ (high accuracy) if there are enough weak learners. This is true by the _law of large numbers_.



<details>
<summary>Law of large numbers Explanation</summary>



The law of large numbers states that, as the sample N increases, the proportion of results tends to approach the true probability.

Thinking about a game of "heads or tails", with an unfair coin, where there is a 51% probability of "heads". From this, assuming there are 100 tosses, we have a binomial distribution $B(n,p)$, where, $n = 100$, being the sample number, and $p = 0.51$, being the probability of the phenomenon occurring, that is, getting "heads".

So, we have to have the probability of getting the majority as "heads", that is, calculate $P(X>50)$, with $X$ being the number of "heads" for the binomial distribution $B(100, 0.51)$. It is possible to approximate a binomial distribution by a normal one due to the number of trials $n$. Considering the mean ($\mu$) and the standard deviation ($\sigma$) for a binomial as:



  $$\mu = n \cdot p = 100 \cdot 0.51 = 51$$



  $\sigma = \sqrt{n \cdot p \cdot (1 - p)} = \sqrt{100 \cdot 0.51 \cdot 0.49} \approx 4.99$


Using the normal approximation, we have the equation for $X > 50$:

  $P(X > 50) \approx P(Z > \frac{50 - \mu}{\sigma}) \approx P(Z > \frac{50 - 51}{4.99}) \approx P(Z > -0.2)$

By converting using the normal table, we see that $P(Z > -0.2) \approx 0.588$, or 58.8%. Interestingly, by this same logic, when changing the sample n to 1000 the probability approaches 75% and when changing the n to 10,000 the probability approaches 97%.

In the same way, assuming that there are 1000 classifiers that are minimally better than random probability, with about 51% accuracy, when predicting the most voted class one can aim for about 75% accuracy.


</details>

>[!NOTE]
> _Ensemble_ methods work best when the predictors are as independent of each other as possible. Using different classifiers with different algorithms increases the chance of improving their accuracy.

You can use the Scikit-Learn `VotingClassifier` class for this, using a list with the name-predictor pairs as shown in the code below:

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
	estimators=[	
		('lr', LogisticRegression(random_state=42)),
		('rf', RandomForestClassifier(random_state=42)),
		('svc', SVC(random_state=42))
	]
)

voting_clf.fit(X_train, y_train)

```

The original value of the estimators can be analyzed using the `estimators_` attribute, and these can be analyzed using the `named_estimators` attribute as a list or `named_estimators_` as a dictionary.

The `predict()` method performs _hard voting_ and returns the class in which there were the most votes.

If all classifiers can estimate the class probabilities (i.e., have the `predict_proba()` method), it is possible to predict the class with the highest probability with the average of the individual classifiers, which is **Soft Voting**. It usually performs better than **hard** because it gives more weight to the best results. To select _soft_, simply set `voting` to "soft".

>[!TIP]
> If the classifier used cannot estimate the probability, the probability hyperparameter should be set to "True", and then a `predict_proba()` method can be added.
<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->


# Bagging and Pasting

In addition to the voting approach, **Bagging** and **Pasting** can also be analyzed. These work using the **same algorithm**, but use **different subsets** of the _training set_. The differences between these consist of:

- **Bagging**: Among the training subsets, there may be repetition of data that appears in another training subset;
- **Pasting**: If a piece of data appears in a training subset, it will not appear in another.

Once all predictors are trained, the ensemble can predict the new instance by aggregating the predictions. In the case of classification, the _statistical_ mode is typically used (usually the most frequent prediction, like a _hard voting_), and in the case of regression, the mean is used. Individual predictors have **higher bias** compared to the original training ensemble, but aggregation tends to **reduce bias and variance**. The total, or network, result has a result with a **similar bias** but **lower variance** compared to a single predictor trained on the original training set.

>[!NOTE]
> It is possible to train in parallel with different CPU cores or different servers, making the model very scalable.
>
<!------------------------------------------------------->  
## Bagging and Pasting in Scikit-Learn

It is possible to use Scikit-Learn for both Bagging and Pasting. The `BaggingClassifier` class is for classification and `BagginRegressor` for regression, and can be changed to _Pasting_ by changing the hyperparameter `bootstrap=False`.

>[!TIP]
> The `n_jobs` parameter informs the number of CPU cores that will be used. `-1` is the command to use all available ones.

The code below demonstrates an example of Bagging with 500 Decision Tree predictors (`n_estimators`) with 100 training instances (`max_samples`). It is interesting to note that _Soft Voting_ is applied if the `BaggingClassifiers` classification predictors have the attribute to estimate the probability of the class `predict_proba()`.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
max_samples=100, n_jobs=-1, random_state=42)

bag_clf.fit(X_train, y_train)
```

It is interesting to see the comparison between decision trees and the use of Bagging. The ensemble has a **comparable Bias** and a **Lower Variance** (same number of errors and less irregular decision boundary).

[Image]

>[!NOTE]
> When comparing Bagging with Pasting, Bagging has a greater diversity of subsets, thus having a higher bias and causing less correlation between them, reducing the variance of the ensemble. Usually, Bagging has better results, but if possible, it is better to test both.

<!------------------------------------------------------->
## Out-of-Bag Evaluation
Using Bagging typically results in approximately 63% of the training instances being used, with the other 37% not participating in the training. This 37% portion is called _Out-Of-Bag_ (OOB).

Often, the OOB is used as a validation set and evaluates the accuracy of the model. This can be done from Scikit-Learn using the `oob_score_` class. In addition, it is possible to use the decision function for each training instance from `oob_decision_function_`, which can return the class probabilities if there is the `predict_proba()` method.



<!------------------------------------------------------->
## Random Patches and Random Subspaces
Feature sampling is something that proves useful when dealing with high-dimensional inputs such as images, and is called the _Random Patches Method_. Two hyperparameters can control this, `max_features` and `bootstrap_features`. By keeping all training instances (e.g. `bootstrap` = False and `max_samples` = 1.0) except for the sampled features, it is called the _Random Subspaces Method_. This results in more diverse predictors, having **higher bias** and **lower variance**.





<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->

# Random Forest

A Random Forest can be seen as an ensemble of Decision Trees, usually by the Bagging method. It has the class `RandomForestClassifier` for classification or `RandomForestRegressor` for regression. Given a few exceptions, `RandomForestClassifier` has all the hyperparameters of `DecisionTreeClassifiers` and `BaggingClassifier`. The code below demonstrates how the code operates a classifier with 500 trees (`n_estimators`) and 16 nodes at most (`max_leaf_nodes`):

```python

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

```


>[!NOTE]
> **Random Forest** introduces extra randomness: instead of choosing the best feature when splitting a node from the training set, it chooses the **best feature of the subset of the tree being trained**. Compared to _Decision Trees_, which choose the _best feature_, **Random Forest** chooses _random features_. It trades off higher bias for lower variance, often producing a better model.



## Extra-Trees

To understand **Extra Trees** (_Extremely Randomized Trees_) it is important to remember the two precursors. **Decision Trees** operate on the complete training set looking for the best threshold. **Random Forest**, unlike **Decision Trees**, looks for the best threshold in **each subset it operates on**.

**Extra Trees** operate using the training subset using all the data from the training set (unlike _bagging_, which uses about 67%) using **Random Thresholds**. This trades a **higher bias** for a **lower variance**. Compared to **Random Forest**, its performance is faster, and the decision between choosing one of the two can be discriminated only by the accuracy in each case.

It has classes in the Scikit-Learn library. For classification, `ExtraTreesClassifier` and for regression `ExtraTreesRegressor`.


## Feature Importance



Something interesting that **Random Trees** allow is the _measurement of feature importance_. It is possible to obtain a score for each feature before training and scale the results, so that the sum of these importances is 1, being accessed through `feature_importances_`.

>[!NOTE]
> It is possible to do this for regressions, classifications and images (with the pixels with more or less importance being represented in a heat graph).


<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->

# Boosting

_Hypothesys Boosting_ or **Boosting** is using Ensemble methods that combine weak learners into a strong one, sequentially training predictors to correct the former.

<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->
## AdaBoost

**AdaBoost** (_Adaptive Boosting_) works by correcting instances that were _underfitted_ by their predecessor, focusing increasingly on difficult cases. The predictors have different weights depending on the accuracy in the weighted training set. You can visualize this type of predictor as **Stumps**, which consist of decision trees with __one node__ and __two leafes__.

>[!NOTE]
> As these corrections are made, the learning rate decreases, that is, the weight of incorrectly classified instances increases less with each iteration.

>[!WARNING]
> It is worth keeping in mind that, given the fact that these trainings are sequential, it is not possible to do the training in parallel, and they do not scale as well as _Bagging_ or _Pasting_.

The equation below shows the **Weighted Error Rate of the $j^{th}$ predictor**, where for each instance the weight $w^{(i)}$ is initially set to $1/m$. For each trained predictor, the generated error rate $r_{1}$ is calculated, and so on according to the equation:


```math
r_{j} = \sum^{m}_{i=1} w^{(i)} \: \: \: where \: \: \: \hat{y}^{(i)}_{j} \neq y^{(i)}
```


<details>

<summary>Equation terms</summary>

- $r_{j}$: Is the Weighted Error Rate of the $j^{th}$ predictor.
- $w^{(i)}$: Is the calculed weight of the $i^{th}$ instance.
- $\hat{y}^{(i)}_{j}$: Is the prediction of the instance $i^{th}$ of the $j^{th}$ predictor.
- $y^{(i)}$: Is the correct label of the $i^{th}$ instance.


</details>



So, the equation below indicates _Predictor's Weight_ ($\alpha _{j}$). The more accurate the predictor, the higher the weight it will have. If it is "guessing" randomly, the weight will be close to zero. If the error rate is very wrong (less accurate than random) the weight will be negative.

```math
\alpha_{j} = \eta log \frac{1 - r_{j}}{r_{j}}
```
<details>
<summary>Equation terms</summary>

- $\alpha_{j}$: Is the Predictor's Weight.
- $r_{j}$: Is the Weighted Error Rate of the $j^{th}$ predictor.
- $\eta$: Is the learning rate hyperparameter (default 1).

</details>


Having these values, the AdaBoost algorithm updates the weight of the instances according to the equation below, which is **Weighted Update Rule**:


```math
w^{(i)} \leftarrow \left\{  \begin{matrix} 
w^{(i)} \: \: \: if \: \: \: \hat{y}^{(i)}_{j} = y^{(i)} \\
w^{(i)} exp(\alpha_{j}) \: \: \: if \: \: \: \hat{y}^{(i)}_{j} \neq y^{(i)}
\end{matrix} \right.
```

With this, the prediction is then made, and the cycle repeats until there is either a perfect predictor or the number of set predictors reaches the limit. The class ($k$) that receives the highest number of votes will be the chosen predictor, which is demonstrated by the equation below:

```math

\hat{y}(x) = argmax_{k} \sum^{N}_{j=1} \alpha_{j} \: \: \: where  \: \: \: \hat{y}_{j}(x) = k

```



<details>
<summary>Equation terms</summary>

- $\alpha_{j}$: Is the Predictor's Weight.
- $N$: Is the number of predictors.
- $k$: Is the possible classes.

</details>


Below is an example code containing 30 _stumps_ using the SciKit-Learn `AdaBoostClassifier` class:

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
 	n_estimators=30,learning_rate=0.5, random_state=42)

ada_clf.fit(X_train, y_train)
```
<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->


## Gradient Boosting

The **Gradient Boosting** method, unlike _AdaBoost_, which adjusts the weight of each instance at each iteration, adjusts the **Residual Error** for each previous predictor. A more in-depth explanation can be found at [this link](https://pages.github.com/](https://www.datacamp.com/tutorial/guide-to-the-gradient-boosting-algorithm) <!--[add link: https://www.datacamp.com/tutorial/guide-to-the-gradient-boosting-algorithm].-->

The code below exemplifies one way to use the algorithm, in this case, using _Decision Trees_ as base predictors for _Regression_, a technique called **Gradient Tree Boosting** or **Gradient Boosted Regression Trees** (GBRT). The example given will use three trees. The _first_ tree will perform the regression on the existing data. The _second_ tree will perform the regression using the residual error committed by the first predictor (`y2`). The _third_ tree will perform the regression using the residual error committed by the second predictor (`y3`).


```python
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X) # Residual errors made by first predictor
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X) # Residual errors made by second predictor
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)

```



## Histogram-Based Gradient Boosting




<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->

# Stacking



<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->
