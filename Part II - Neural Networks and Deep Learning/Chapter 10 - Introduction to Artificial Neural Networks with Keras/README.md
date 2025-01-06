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

- $1^{st}$: $z = w_{1}x_{1} + w_{2}x_{2} + \cdot \cdot \cdot w_{n}x_{n} + b = w^{T}x + b$
- $2^{nd}$: $h_{w}(x) = step(z)$

- $x$: Inputs;
- $w$: Weights of values; - $b$: Bias;
- $z$: Linear Function;
- $h$: Step Function.

The _Step Function_ can be either _Heaviside Step Function_ or _Sign Function_, less commonly:

```math

heaviside(z) = \: \: \: \: \: \: \: \: sgn(z) =

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

- $X$: Input Matrix. Has one row per instance and one column per feature;
- $W$: Weight Matrix. Has one row per input and one column per neuron;
- $b$: Bias Vector. Contains one bias term per neuron;
- $\Phi$: _Activation Function_. In the case of TLUs, _Step Function_


<!------------------------------------------------------>
<!------------------------------------------------------>
## The Multilayer Perceptron and Backpropagation

<!------------------------------------------------------>
<!------------------------------------------------------>
## Regression MLPs

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
