# Training Models
This chapter will cover in more depth how the algorithms themselves work, rather than just using them in practice. This will be beneficial for improving analyses and correcting errors more effectively. With mathematics as the main focus in this section, the following topics will be studied:

- Linear regression and ways to train it
- Direct calculation to fit the set, "closed form"
- Iterative optimization, which is Gradient Descent (GD), and some variables
- Polynomial Regression
- Logistic Regression
- Softmax Regression


$\hat{x}$, $\tilde{x}$, $\vec{x}$




A, *A*,**A**, $\textbf{A}$ , $A$



```math
\mathbf{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}
```


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

_Gradient Descent_ is a very general optimization algorithm capable of iteratively adjusting parameters to minimize a cost function.

The algorithm measures the local gradient of the error function with respect to the parameter vector $\theta$, and goes in the direction of gradient descent, and when it reaches zero it becomes the minimum. The image below will be used to explain the gradient.

![GradDesc](https://github.com/user-attachments/assets/84eccbeb-c249-40f3-878e-eb71beed5c8c)

<!-- Lista pra ver as parada -->
- f
- f
- f
  


## Batch Gradient Descent
## Stochastic Gradient Descent
## Mini-Batch Gradient Descent




# Polynomial Regression







# Learning Curves







# Regularized Linear Models


## Ridge Regression
## Lasso Regression
## Elastic Net Regression
## Early Stopping








# Logistic Regression


## Estimating Probabilities
## Training and Cost Function
## Decision Boundaries
## Softmax Regression


