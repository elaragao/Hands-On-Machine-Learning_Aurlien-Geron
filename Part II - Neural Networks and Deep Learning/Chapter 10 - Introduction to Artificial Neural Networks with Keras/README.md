# Introduction to Artificial Neural Networks with Keras

**Artificial Neural Networks** (ANNs) were inspired by the functioning of neurons in the brain. The functioning of this tool is the core of _Deep Learning_, being versatile, scalable and powerful for complex Machine Learning problems.

This chapter will present the functioning and architectures of ANNs, Perceptrons and Multilayer Perceptrons. Then, implementations using TensorFlow Keras API will be seen.

<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>

# From Biological to Artificial Neurons

Briefly speaking of history, ANNs had their first "mention" around the year 1940, where it was explained mainly as _Propositional Logic_ of computers to perform activities similar to the brain. Later, this concept was revisited in the 1960s, until in the 1980s better architectures were explored and a prototype of what we have today was filament, until, in 1990.
<!------------------------------------------------------>
<!------------------------------------------------------>
## Biological Neurons

<!------------------------------------------------------>
<!------------------------------------------------------>
## Logical Computations with Neurons
The first model proposed for _Artificial Neurons_ operated basically as a binary on/off operation coordinated from _logic gates_. The image below demonstrates some cases:

[Image]

- $C = A$: If neuron A is activated, C will be activated;
- $C = A \wedge B$: Performs a logical AND. Neuron C is activated only when A and B are activated;
- $C = A \vee B$: Performs a logical OR. Neuron C is activated when A is activated, B is activated, or A and B are activated;
- $C = A \wedge \neg B$: Performs a logical AND and NOT (in B). C is activated when A is activated and B is deactivated.



<!-- https://gitmind.com/pt/faq/latex-math-symbols.html -->


<!------------------------------------------------------>
<!------------------------------------------------------>
## The Perceptron

**Perceptron** is one of the simplest forms of ANNs, based on artificial neuron models called _Threshold Logic Unit_ (TLU) or _Linear Threshold Unit_ (LTU), where the inputs and outputs are **numbers** instead of binary values, and their connections are associated with **Weights**. The algorithm, in basic form, operates in two steps, starting with linear functions for the inputs followed by a _Step Function_:

- $1^{st}$: $z = w_{1}x_{1} + w_{2}x_{2} + \cdot \cdot \cdot + w_{n}x_{n} + b = w^{T}x + b$
- $2^{nd}$: $h_{w}(x) = step(z)$

<details>
<summary>Equation terms</summary>
  
- $x$: Inputs;
- $w$: Weights of values;
- $b$: Bias;
- $z$: Linear Function;
- $h$: Step Function.
  
</details>
  
The _Step Function_ can be either _Heaviside Step Function_ or _Sign Function_, less commonly:

```math

heaviside(z) = 

\left\{  \begin{matrix} 

0 \: \: \: if \: \: \: z < 0\\
1 \: \: \: if \: \: \: z \geq 0

\end{matrix} \right.


\: \: \: \: \: \: \: \: \: \: \: \:

sgn(z) =

\left\{  \begin{matrix} 

-1 \: \: \: if \: \: \: z < 0\\
0 \: \: \: if \: \: \: z = 0\\
+1 \: \: \: if \: \: \: z > 0

\end{matrix} \right.
```

The schematic below basically demonstrates how a Perceptron works based on TLUs, which can be divided into three main layers:

- _Input Layer_: Layer containing inputs;
- _Fully Connected Layer_ or _Dense Layer_: Single layer where every TLU is connected to each Input;
- _Output Layer_: Layer containing outputs.

[Image]

>[!NOTE]
> The Perceptron consists of a single layer of TLUs.

The Image Perceptron can classify instances into three different binary classes. The equation below shows the simplified equation:

```math

h_{W,b}(X) = \Phi (XW + b)

```

<details>
<summary>Equation terms</summary>
  
- $X$: Input Matrix. Has one row per instance and one column per feature;
- $W$: Weight Matrix. Has one row per input and one column per neuron;
- $b$: Bias Vector. Contains one bias term per neuron;
- $\Phi$: _Activation Function_. In the case of TLUs, _Step Function_

</details>



The way the Perceptron is trained is similar to `SGD`. It operates on an idea similar to *Hebb's Rule* or *Hebbian Learning*, which states that neurons tend to increase their connections when they are fired simultaneously. For each output neuron that produced a wrong prediction, it reinforces the connection weights of the inputs that would have contributed to the correct prediction. The equation that governs this is called _Perceptron Learning Rule_, or _Weight Update_.

```math

w_{(i,j)}^{next \: step} = w_{(i,j)} + \eta (y_{j} - \hat{y_{j}})x_{i}

```

<details>
<summary>Equation terms</summary>
  
- $w_{(i,j)}$: Connection weight of i^{th} input and the j^{th} neuron;
- $y_{j}$: j^{th} ouput of training instance;
- $\hat{y_{j}}$: j^{th} output target;
- $x_{i}$: The i^{th} input of training instance;
- $\eta$: Learning Rate.

</details>


> [!NOTE]
> **Perceptron Learning Algorithm** can be the equivalent of using `SGDClassifier` with the following hyperparameters: `loss = "perceptron"`, `learning_rate = "constant"`, `eta0 = 1` (learning rate), `penalty = None` (no regularization).



Although it is not possible to perform very complex patterns because they have linear outputs, the algorithm tends to converge to a solution due to the _Perceptron Convergence Theorem_. It is possible to import the `Perceptron` class from the Scikit-Learn library.

Despite this, the algorithm is unable to handle complex problems, such as XOR logic gates. This limitation is overcome by *MultiLayer Perceptron* (MLP).

<!------------------------------------------------------>
<!------------------------------------------------------>
## The Multilayer Perceptron and Backpropagation

MLP consists of more than one layer of TLUs. The layers closest to the inputs are called _Lower Layers_, those closest to the outputs _Upper Layers_, the final layer _Output Layers_ and the additional layers in general, _Hidden Layers_.

[Image]

When ANNs contain many _Hidden Layers_ they are called _Deep Neural Networks_, and the field that usually studies this is Deep Learning.


To train MLP algorithms, a method called **Reverse-Mode Automatic Differentiation** (or **Reverse-Mode Autodiff**) is used, calculating the gradients automatically and efficiently. The algorithm makes two passes through the neural network (front and back) and is able to discover how the weight and bias of each neuron connection should be adjusted to reduce the error. Through this calculation, it is then possible to apply _Gradient Descent_ to train the algorithm. This combination of techniques is called **Backpropagation**.





The algorithm for this basically operates in 5 steps:

- 1st Step: The algorithm operates one _mini-batch_ at a time and passes through the entire training set several times. Each pass through the set is called an **Epoch**;

- 2nd Step: The _mini-batch_ enters the network, passes through the _Input Layer_, and then passes to the first _Hidden Layer_, its output is computed, which is used as input for the next _Hidden Layer_ and so on. This is called the **Forward Pass**, and the intermediate results are preserved for the **Backward Pass**;

- 3rd Step: The _Output Error_ is measured;

- 4th Step: The amount of each output in each layer contributes to the error is calculated analytically by the _Chain Rule_;

- 5th Step: The **Backward Pass** is performed. The algorithm measures how much error contributions came from the lower layers until reaching the beginning using the _Chain Rule_ again. This measures the error of all the weights and biases of the connection;

- 6th Step: The _Gradient Descent_ is used.

>[!CAUTION]
> It is important that all layers are initialized with random weights and biases.

>[!NOTE]
> Summary of **Backpropagation**: Predictions are made for mini-batches (forward pass), the error is measured, the error is passed through layers backwards to measure the error contributions of the parameters (backward pass) and then the weights and biases of each connection are adjusted (gradient descent).

Unlike how it is done in Perceptrons, a different _Activation Function_ is required in MLPs. This became necessary because the _Step Function_ does not have gradients to work with (zero derivative). Some activation functions used are:

- Sigmoid Function: $\sigma (z) = 1/(1 + exp(-z))$
- Hyperbolic Tangent Function: $tanh(z) = 2 \sigma (2z) -1$
- Rectified Liner Unit (ReLU) Function: $ReLU(z) = max(0,z)$

<!------------------------------------------------------>
<!------------------------------------------------------>
## Regression MLPs
In cases of Regression, to use MLPs, each prediction is made in an output neuron, or _Output_. For example, in a simple regression, there will be one output neuron, in a multivariate regression, there will be one output neuron per dimension.

It is possible to use the `MLPRegressor` class from Scikit-Learn, and the simplified dataset used in Chapter 2 will be used as an example (add Link). The data will then undergo processing and will be standardized so that there are no problems with Gradient Descent. The ReLU activation function will be applied to the _Hidden Layers_ and the _Adam_ variant of Gradient Descent and L2 regularization will be used.

>[!NOTE]
> Scikit-Learn does not provide an _Activation Function_ for the _Output Layer_ and only supports MSE.


Abaixo se tem o funcionamento básico de MLPs para regressão:

| Hyperparameter | Typical Value |
| :---         |     :---:      |
| # Hidden Layers | Usually 1-5|
| # Neurons Per Hidden Layer | Usually 10-100|
| # Output Neurons | 1 per prediction dimension |
| Hidden Activation | ReLU |
| Output Activation | None or ReLU/Softplus (positive outputs) or sigmoid/tanh (bounded outputs)|
| Loss Function |  MSE or Hubber if outliers|

<!------------------------------------------------------>
<!------------------------------------------------------>
## Classification MLPs


<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>

# Implementing MLPs with Keras

<!------------------------------------------------------>
<!------------------------------------------------------>
## Building an Image Classifier Using the Sequential API

<!------------------------------------------------------>
<!------------------------------------------------------>
## Building a Regression MLP Using the Sequential API

<!------------------------------------------------------>
<!------------------------------------------------------>
## Building Complex Models Using the Functional API

<!------------------------------------------------------>
<!------------------------------------------------------>
## Using the Subclassing API to Build Dynamic Models

<!------------------------------------------------------>
<!------------------------------------------------------>
## Saving and Restoring a Model

<!------------------------------------------------------>
<!------------------------------------------------------>
## Using Callbacks

<!------------------------------------------------------>
<!------------------------------------------------------>

## Using TensorBoard for Visualization

<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>

# Fine-Tuning Neural Network Hyperparameters

<!------------------------------------------------------>
<!------------------------------------------------------>
## Number of Hidden Layers

<!------------------------------------------------------>
<!------------------------------------------------------>
## Number of Neurons per Hidden Layer

<!------------------------------------------------------>
<!------------------------------------------------------>
## Learning Rate, Batch Size, and Other Hyperparameters
