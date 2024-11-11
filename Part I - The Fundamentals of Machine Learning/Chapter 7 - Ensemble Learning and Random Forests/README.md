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



## Extra-Trees



## Feature Importance






<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->

# Boosting



## AdaBoost




## Gradient Boosting




## Histogram-Based Gradient Boosting




<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->

# Stacking



<!------------------------------------------------------->
<!------------------------------------------------------->
<!------------------------------------------------------->
