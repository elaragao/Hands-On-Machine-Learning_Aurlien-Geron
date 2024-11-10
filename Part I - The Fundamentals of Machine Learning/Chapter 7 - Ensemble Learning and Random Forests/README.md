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

In addition to the voting approach, bagging and pasting can also be analyzed:

- **Bagging**:
- **Pasting**:

  
## Bagging and Pasting in Scikit-Learn


## Out-of-Bag Evaluation


## Random Patches and Random Subspaces



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
