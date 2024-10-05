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
\text{\boldmath$\theta$}

<summary>Equation terms</summary>

- $\hat{y}$: Is the predicted value.
- $\theta_{n}$: Is the constant value that represents the intercept of the line with the vertical axis.
- $x_{n}$: Is the slope, the **angular coefficient**.
- $X$: Is the independent variable.
  
</details>





## The Normal Equation




$$\textsf{\textbf{A}}$$

$$\textbf{A}$$

$$\textsf{A}$$


## Computational Complexity







# Gradient Descent
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


