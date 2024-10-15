# Training Models
This chapter will cover in more depth how the algorithms themselves work, rather than just using them in practice. This will be beneficial for improving analyses and correcting errors more effectively. With mathematics as the main focus in this section, the following topics will be studied:

- Linear regression and ways to train it
- Direct calculation to fit the set, "closed form"
- Iterative optimization, which is Gradient Descent (GD), and some variables
- Polynomial Regression
- Logistic Regression
- Softmax Regression



<!--
$\hat{x}$, $\tilde{x}$, $\vec{x}$

A, *A*,**A**, $\textbf{A}$ , $A$



```math
\mathbf{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}
```


```math
\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
```
-->

# Linear Regression
The simplified _Linear Regression_ equation can be seen as:

```math
\hat{y} = \alpha + \beta X
```
<details>

<summary>Equation terms</summary>

- $\hat{y}$: Is the predicted value.
- $\alpha$: Is the constant value that represents the intercept of the line with the vertical axis.
- $\beta$: Is the slope, the **angular coefficient**.
- $X$: Is the independent variable.
  
</details>


In general terms, the linear model makes predictions by computing a weighted sum of the input features, plus a constant called the bias term (also called the intercept term). A more formal way of writing the equation:


```math
\hat{y} = \theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{n} x_{n} 
```
<details>
<summary>Equation terms</summary>

- $\hat{y}$: Is the predicted value.
- $\theta_{n}$: Is the constant value that represents the intercept of the line with the vertical axis.
- $x_{n}$: Is the slope, the **angular coefficient**.
- $X$: Is the independent variable.
  
</details>

An even more concise way of representing this equation is through the vectorized form:

```math
\hat{y} = h_{\theta}(x) = \theta \cdot \textbf{x}

```


<details>
<summary>Equation terms</summary>

- $h_{\theta}$: is the hypothesis function, using the model parameters $\theta$.
- $\theta$: is the **model’s parameter vector**, containing the bias term $\theta_{0}$ and the feature weights $\theta_{1}$ to $\theta_{n}$ 
- $x$: is the **instance’s feature vector**, containing $x_{0}$ to $x_{n}$ , with $x_{0}$ always equal to 1.


  

  
</details>



> [!NOTE]
> $\theta \cdot x$: is the **dot product** of the vectors $\theta$ and $x$, which is equal to $\theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{n} x_{n}$


Furthermore, it is necessary to use a metric to evaluate how well the model performed, and for simplicity, the chosen metric will be MSE. The equation below demonstrates a linear regression using the hypothesis $h_{\theta}$ on the training set $X$:


```math

MSE(X, h_{\theta}) = \frac{1}{m} \sum^{m}_{i = 1} (\theta^{T}x^{i} - y^{i})^{2}

```


## The Normal Equation
One way to find the value of $\theta$ that minimizes the MSE is through the Ordinary Least Squares, or, _Normal Equation_.

```math

\hat{\theta} = (X^{T}X)^{-1} X^{T} y

```
<details>
<summary>Equation terms</summary>

- $\hat{\theta}$: is the value of $\theta$ that minimizes the _cost function_
- $y$: is the **vector of target** containing $y^{(1)}$ to $y^{(m)}$

</details>



Written in code form:

```python
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X) # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

> [!TIP]
> In case A and B are both NumPy Arrays, the operation `A @ B` is equivalent to `np.matmul(A,B)`.

Therefore, the way to predict a value using these equations would be:

```python
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new) # add x0 = 1 to each instance
y_predict = X_new_b @ theta_best
y_predict

```

Similarly, it can be calculated using the SciKit-Learn library

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)
```


<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->
<!-- Man, I got stuck here. I need to come back later because there was a lot of information in a few lines. -->

## Computational Complexity
When using the training method and parameter adjustments mentioned, the computational complexity and time taken are linear, that is, having twice as many resources would take twice as long. It is necessary to analyze other ways of adjusting the parameters, such as those that will be seen in later sections.






# Gradient Descent

Gradient Descent is a very general optimization algorithm capable of iteratively adjusting parameters to minimize a cost function.

The algorithm measures the local gradient of the error function with respect to the parameter vector $\theta$, and goes in the direction of gradient descent, and when it reaches zero it becomes the minimum. The image below will be used to explain the gradient. In a normal case, the method starts with a random $\theta$ (called _Random Initialization_), and then gradually improves. The image below demonstrates how _Gradient Descent_ operates.


![GradDesc_3](https://github.com/user-attachments/assets/eb202878-ed1b-4fdc-93a2-f6f2afbea50e)


> [!NOTE]
> It is interesting to note that usually the size of the learning step is proportional to the slope of the cost function, so the steps gradually get smaller as the cost approaches the minimum.


An important parameter is the step size, which is determined by the hyperparameter _Learning Rate_. Some cases can be determined or drastically affected by these


![GradDesc_2](https://github.com/user-attachments/assets/dc7af7e7-d143-4fdf-8aa8-3a1c40b83c0f)


<!-- Lista pra ver as parada -->

- The image on the left shows when the _Learning Rate_ is **too small**, which makes the algorithm need many iterations, and therefore, takes longer.
- The central image shows when the _Learning Rate_ is **too high**, which makes it go from side to side of the valley, and there is a possibility of ending up higher than when it started, usually failing to find a good solution.
- The image on the right shows the possibility of having **Local Minimum**, which tends to be no better than the **Global Minimum**. The image shows cases of starting on the left, which would make it stay at the **Local Minimum**, and if it started on the right, it would take a long time to reach the **Global Minimum**










<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->

## Batch Gradient Descent

In order to implement _Gradient Descent_, it is necessary to calculate the gradient of the cost functions of each parameter of the model $\theta_{j}$, that is, calculate for small changes in $\theta_{j}$. To calculate this function, the equation below is used:


```math
\frac{\partial}{\partial \theta _{j}}MSE(\theta) = 
\frac{2}{m}\sum_{i=1}^{m}(\theta^{T} x^{(i)} - y^{(i)}) x_{j}^{(i)}
```

In simpler terms, it is possible to calculate the gradient vector described as $\nabla MSE(\theta)$, which contains all the partial derivatives of the cost function (one for each parameter of the model). This is why it is called _Bach Gradient Descent_, since it calculates the entire batch of data at each step.

```math
\nabla _{\theta} MSE(\theta) =
\begin{pmatrix}
  \frac{\partial}{\partial \theta_{0}}MSE(\theta) \\
  \frac{\partial}{\partial \theta_{1}}MSE(\theta) \\
  ... \\
  \frac{\partial}{\partial \theta_{n}}MSE(\theta)
\end{pmatrix}
= \frac{2}{m} X^{T} \cdot (X \cdot \theta - y)
```

Then, obtaining the value, it is necessary to subtract the value $\theta$ from the result obtained by the gradient $\nabla MSE(\theta)$, which is multiplied by a constant called _Learning Rate_, represented by the letter $\eta$, which operates determining the step size.

```math

\theta ^{(next \: step)} = \theta - \eta \nabla _{\theta} MSE(\theta)

```

The code below exemplifies how this equation works:

```python

eta = 0.1  # learning rate
n_epochs = 1000 # number of iterations
m = len(X_b)  # number of instances

np.random.seed(42)
theta = np.random.randn(2, 1)  # randomly initialized model parameters

for epoch in range(n_epochs):
	gradients = 2 / m * X_b.T @ (X_b @ theta - y)
	theta = theta - eta * gradients


```

The image below demonstrates how different learning rate values ​​operate.


![GradDesc_5](https://github.com/user-attachments/assets/e3c47a34-6876-45f7-8edf-8b1aea07b2e0)




- **On the left** there is a very low **Learning Rate**, it will be possible to reach the solution, but it will take a long time.
- **In the center** the **Learning Rate is good**, converging to the solution in a few iterations
- **On the right** there is a very high **Learning Rate**, causing the algorithm to diverge, getting further away from the solution at each step.

> [!NOTE]
> It is possible to use `GridSearch` to find a good learning rate, but it is important to limit the number of iterations to avoid slow models.


> [!TIP]
> How to set the number of iterations? If it is too low, it may be far from the ideal solution, if it is too high, it will take a long time. One possible solution is to **set a large number of iterations** and stop the algorithm when the gradient vector becomes too small, when the norm becomes smaller than the number $\epsilon$ (the _tolerance_), which occurs when the Gradient Descent reaches **Near Minimum**.

<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
## Stochastic Gradient Descent
To understand _Stochastic Gradient Descent_ (**SGD**), it is important to reinforce what _Batch Gradient Descent_ is.
- **Batch Gradient Descent** uses **the entire training set** to calculate the gradients at each step, and if the training set is too large, it will slow down the algorithm.
- **Stochastic Gradient Descent** chooses **a random instance** (a single sample, like, $x^{(42)}$ and $y^{(42)}$) in the training set at each step and calculates the gradients based on this chosen random instance, making the algorithm faster.

The stochastic nature or randomness of SGD makes the algorithm much less regular compared to _Batch Gradient Descent_, which smoothly reduces to a minimum, SGD **approaches the minimum** and will keep "bouncing", but not stabilizing.


> [!NOTE]
> In short, randomness is **good** because it avoids local minima, and **bad** because it does not establish a minimum.

One way to reduce the problem caused by randomness is to gradually decrease the _learning rate_, starting with large values ​​to avoid local minima and decreasing it until a global minimum is established, a process called _simulated annealing_. The name of the function that determines the _learning rate_ at each iteration is _learning scheadule_.

> [!CAUTION]
> If the learning rate decreases **quickly** it can get stuck in local minima, if it decreases **slowly** it can hover around the minimum for a long time.


```python

n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters


def learning_schedule(t):
	return t0 / (t + t1)
 

np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization


for epoch in range(n_epochs):
	for iteration in range(m):
		random_index = np.random.randint(m)
		xi = X_b[random_index : random_index + 1]
		yi = y[random_index : random_index + 1]
		gradients = 2 * xi.T @ (xi @ theta - yi)  # for SGD, do not divide by m
		eta = learning_schedule(epoch * m + iteration)
		theta = theta - eta * gradients

```



It is possible to perform linear regression using SGD through the Scikit-Learn `SGDRegressor` library. The function operates until it reaches the maximum number of epochs as 1000 (_max_iter_) **or** until it decays less than the set value of $10^{-5}$ (_tol_) in an interval of 100 epochs (_n_iter_no_change_). It starts with a _learning rate_ of 0.01 (_eta0_).


```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,n_iter_no_change=100, random_state=42)

sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
```


<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
## Mini-Batch Gradient Descent



<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
# Polynomial Regression

_Polynomial Regression_ is a way to perform regression on datasets that do not follow a linear structure using a linear-like model. A simple way to do this is to add the necessary powers to fit the model. The Scikit-Learn `PolynomialFeatures` class allows you to perform the process:




```python
from sklearn.preprocessing import PolynomialFeatures


poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)


lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
```






<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
# Learning Curves
There are cases in which we do not know the degree of the polynomial to apply to the data fit. If a small degree is chosen, _underfitting_ will tend to occur, if a very large degree is chosen, there will be _overfitting_. It is common practice to use _cross validation_ to evaluate performance.

> [!NOTE]
> It is important to remember that if the model performs well on the training set and generalizes poorly, the case tends to be **Overfitting**. If the model performs poorly on both, it tends to be **Underfitting**.

In addition to this method, it is also possible to use _Learning Curves_, which consist of graphs comparing the model on the training and validation sets as functions of the size of the training set (or training iteration). They are trained several times on subsets of different sizes in the training set, and then a function is defined for the plot of the curves. The error naturally starts at 0, and stabilizes until it reaches a plateau. An example can be seen in the code below:


```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
 LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5, scoring="neg_root_mean_squared_error")

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")

plt.show()

```

> [!NOTE]
> By observing the graphs, behaviors can be observed to see whether or not the function would work for the purpose. In the case of **Underfitting**, the training and validation lines will be very close, but with a high error. In the case of **Overfitting**, a considerable distance will be observed between the training and validation sets, with the training set having the lowest error.


<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
# Regularized Linear Models
One way to reduce overfitting is to regularize the model. The fewer degrees of freedom a model has, the more difficult it will be to overfit, and a simple way to regularize a polynomial model is to reduce the number of degrees. In more linear models, it is common for regularization to be done by adjusting and restricting the model's weights.

## Ridge Regression


<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
## Lasso Regression


<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
## Elastic Net Regression


<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
## Early Stopping







<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
# Logistic Regression


<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
## Estimating Probabilities

<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
## Training and Cost Function


<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
## Decision Boundaries



<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
<!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção --> <!-- Nova Seção -->
## Softmax Regression


