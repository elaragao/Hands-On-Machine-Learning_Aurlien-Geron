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
MLPs, in cases of classification, can be divided into 3 main cases:

- **Binary Classification**: They can have only one _Output Layer_, with the return being a sigmoid with values ​​0 or 1;
- **Multilabel Binary Classification**: In cases, for example, of classifying _Spam_ and _Ham_ emails that can be _Urgent_ and _Non-Urgent_, there would be 2 Outputs, one for Spam/Ham and the other for Urgent/Non-Urgent, and it is also possible to use a sigmoid function;
- **Multiclass Classification**: where classifications can be multiple options, but only one at a time, as in the case of digits 0 and 9 there would be 10 outputs, one per class, and in this case the softmax function is usually used where there is a probability of being part of each class between 0 and 1, with the total value added to 1.

For classifications in MLPs, in general, the _Loss Function_ of **Cross-Entropy Loss** is used. Scikit-Learn uses the `MLPClassifier` class for these tasks, and operates in a very similar way to `MLPRegressor`, differing in that it uses the minimization of _Cross Entropy_ instead of MSE. The table that can represent the classification using MLPs:



| Hyperparameter | Binary Classification | Multilabel Binary Classification | Multiclass Classification |
| :--- | :---: | :---: | :---: |
| # Hidden Layers | Usually 1-5| | |
| # Output Neurons | 1 | 1 per binary label | 1 per class|
| Output Layer Activation | Sigmoid | Sigmoid | Softmax |
| Loss Function | X-Entropy | X-Entropy | X-Entropy |

<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>

# Implementing MLPs with Keras

Keras is TensorFlow’s high-level deep learning API.


<!------------------------------------------------------>
<!------------------------------------------------------>
## Building an Image Classifier Using the Sequential API


To illustrate these cases, we will use the MNIST dataset containing 70,000 grayscale images in dimensions of 28 × 28 pixels with 10 classes representing clothing items, called Fashion MNIST. It is worth noting that linear models achieve an accuracy of 83% in this set.

<!------------------------------------------------------>
<!------------------------------------------------------>

### Using Keras to Load the Dataset

Para se exemplificar esets casos, será utilizada o conjutno de dados do MNIST contenod 70000 imagnes em grayscale em dimensão 28 × 28 pixels com 10 classes representando peças de roupa, denominado Fashion MNIST.É válido ressaltar que, modelos lineares conseguem uma acurácia de 83% neste conjunto.


<!------------------------------------------------------>

### Using Keras to Load the Dataset
The Fashion MNSIT dataset is already separated into training and testing sets, and is also shuffled, but a portion will be kept for the testing set. Unlike the MNSIT set, the Fashion MNSIT has dimensions of 28×28 instead of a 1D array of size 784, and the intensity is represented by integers from 0 to 255.

<!------------------------------------------------------>


### Creating the model using the sequential API

Below we see a neural network with two _Hidden Layers_:

```python
tf.random.set_seed(42) # 1
model = tf.keras.Sequential() # 2
model.add(tf.keras.layers.Input(shape=[28, 28])) # 3
model.add(tf.keras.layers.Flatten()) # 4
model.add(tf.keras.layers.Dense(300, activation="relu")) # 5
model.add(tf.keras.layers.Dense(100, activation="relu")) # 6
model.add(tf.keras.layers.Dense(10, activation="softmax")) # 7
```

The explanation of the following lines is:

- **1**: Random seed to make reproducible results, randomizing the initial weights of the _Hidden Layers_ and _Output Layers_;
- **2**: Creates a _Sequential Model_, where there are individually connected stacks;
- **3**: _Input Layer_ created and added to the model, specifying the format of the instances;
- **4**: _Flatten Layer_. Converts images into a 1D array. Does only a simple preprocessing, in our case, from [32, 28, 28] to [32,784];
- **5**: Added a _Dense Hidden Layer_ containing 300 neurons, using the ReLU activation function. Each _Dense Layer_ contains its own weight matrix and each neuron contains a bias;
- **6**: Second _Dense Hidden Layer_ with 100 neurons;
- **7**: _Dense Output Layer_ with 10 neurons (1 per class) using the Softmax activation function.

>[!TIP]
> You can use the `.summary()` function to obtain information about each layer, including the name, output format, and parameter numbers.

A simpler way to write the above function is:

```python
model = tf.keras.Sequential(
	[
		tf.keras.layers.Flatten(input_shape=[28, 28]),
		tf.keras.layers.Dense(300, activation="relu"),
		tf.keras.layers.Dense(100, activation="relu"),
		tf.keras.layers.Dense(10, activation="softmax")
	]
)
```


If desired, it is possible to change the names of each layer, but Keras gives unique names to each layer. The `get_layer()` method shows the names of the model layers, just as the `get_weight()` method returns the weights, and `set_weight()` allows them to be changed.

>[!NOTE]
> The weights are initialized randomly, and the biases can be initialized to zero. If you want a different initialization, you can use `kernel_initializeer` for weights or `bias_initializer` for bias.


<!------------------------------------------------------>


### Compiling the Model

After creating the model, it is necessary to compile it using `compile()`, specifying the _Loss Function_ and _Optimizer_, and other optional arguments.

```python
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
```

The reason for using this _Loss Function_ is because there are sparse labels, that is, there is only one possible result per class, and the classes are exclusive.
<!------------------------------------------------------>

### Training and Evaluating the Model

To do the training, it is necessary to call the `fit()` function, using the chosen inputs (X_train), the target classes (y_train), the number of epochs for training and the validation data (optional).

```python
history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid))
```



>[!NOTE]
> Keras shows the number of _mini-batches_ being processed during each _epoch_. The default size of the _batches_ is 32, and since we are training 55000 images, there are 1719 _batches_ per _epoch_.

>[!TIP]
> If there are classes with a larger number of representatives than others, it is possible to adjust their weight using the `class_weight()` argument.

The `fit()` method returns the _History_ object, containing the training parameters (`history.params`), the list of epochs (`history.epoch`) and the dictionary containing the _loss_ and other extra metrics (`history.history`).



>[!NOTE]
> It is interesting to note that the _training error_ is computed using a moving average during each epoch, while the _validation error_ is computed at the **end** of each epoch.
> Also, the performance of the _training set_ tends to outperform the _validation set_ when trained for long enough (_training set_ usually approaches 100%).



If the performance is still not what is desired, the hyperparameters can be adjusted in the following order:
- Learning rate;
- Try another optimizer;
- Model hyperparameters (like number of neurons per layer, number of layers, activation functions, `batch_size`)

Once the accuracy that satisfies the model is obtained, it is evaluated in the _test set_ using the `evaluate()` method:

```python
model.evaluate(X_test, y_test)
```


<!------------------------------------------------------>

### Using the Model to Make Predictions

To make the model prediction, use the `predict()` method

```python
y_proba = model.predict(X)
```

Using the NumPy method, `argmax()`, the class with the highest probability is returned:

```python
import numpy as np

y_pred = y_proba.argmax(axis = -1) # Return class with greater probability

np.array(class_names)[y_pred] # Return class name with greater probability
```



<!------------------------------------------------------>
<!------------------------------------------------------>
## Building a Regression MLP Using the Sequential API



For regression cases, we will use the California Housing dataset again. It will consist of 3 _Hidden Layers_ composed of 50 neurons.

The API construction will be similar to that used for regression. The difference is that the _Output Layer_ is only one neuron, the _Activation Function_ will not be used and the _Loss Function_ is MSE, the metric is RMSE and the _Adam Optimizer_ will be used as _Optimizer_. Furthermore, instead of the _Flatten Layer_, a _Normalization Layer_ is used, which is adjusted in the model **before** `fit()` using the `adapt()` method.


```python
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
```


<!------------------------------------------------------>
<!------------------------------------------------------>
## Building Complex Models Using the Functional API


An example of a non-sequential neural network is the so-called _Wild & Deep_ (shown in the image below), which connects the input (or part of it) to the output, making it possible to learn deep patterns (through the deep path) and simple rules (through the short path). It differs from the usual MLPs, where everyone starts from and through the same path.

[Image]


The code below shows how the image above would be written.
Initially, the first 5 lines create the _Layers_, being _Normalization_, 2 _Hidden Layers_ with 30 neurons and _ReLU_ activation function, a _Concatenate Layer_ and an _Output Layer_ with one neuron. The following lines are used to make the "path" between the _input_ and the _output_.

The `normalization_layer` is used as a function, passing the `input_` as an object, and this is done consecutively with the layers until reaching the output, connecting them sequentially. The `concat_layer` connects both sequentially (for the _Deep_ case) and directly from the `normalization_layer` (for the _Wide_ case).



```python
normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])
```



An alternative way is if you want to use both a _Deep_ and a _Wide_ case. You can do this by declaring the functions as follows:

```python
input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat) # Remember that part for the next code example
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])
```



They will behave as shown in the image below:

[Image]

To compile models like this, it is done in a very similar way to the usual way, however, passing two inputs (`X_train_deep` and `X_train_wide`) instead of the `X_train` matrix:

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit((X_train_wide, X_train_deep), y_train, epochs=20,
                    validation_data=((X_valid_wide, X_valid_deep), y_valid))
mse_test = model.evaluate((X_test_wide, X_test_deep), y_test)
y_pred = model.predict((X_new_wide, X_new_deep))
```

>[!NOTE]
> This can be used for tasks such as locating and classifying an object in an image.



In addition, it is also possible to use a different _Output_ to perform regularization, this being an _Auxiliary Output_, to see if the underlying part of the network learns something useful on its own, without depending on the rest of the network. Each one should have its own _Loss Function_, with different weights (giving more weight to the main output, and less to the auxiliary). The image below demonstrates how this works:

[Image]

The code for these steps is:

```python
# [...] Same as above, up to the main output layer
output = tf.keras.layers.Dense(1)(concat)
aux_output = tf.keras.layers.Dense(1)(hidden2)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output, aux_output])


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=("mse", "mse"), loss_weights=(0.9, 0.1),
	optimizer=optimizer, metrics=["RootMeanSquaredError"])
```

To train the model, there must be different labels for different outputs. therefore, there must be one output for the main output, and another for the auxiliary output. It is also necessary to clarify that the sums of the Loss weights must be treated as individual metrics:


```python
norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit((X_train_wide, X_train_deep), (y_train, y_train), epochs=20,validation_data=((X_valid_wide,

eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results

y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
```

<!------------------------------------------------------>
<!------------------------------------------------------>
## Using the Subclassing API to Build Dynamic Models


Both sequential and functional APIs are declarative. You start by declaring which layers you want to use, how to connect them, and then feeding them as a model. Despite its many advantages, the model is static, meaning it cannot operate on models that involve things like loops, conditionals, branching, or other dynamics. Subclasses are used for this:


```python
class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # needed to support naming the model
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output


model = WideAndDeepModel(30, activation="relu", name="my_cool_model")
```


The above class does the same thing as what was done in the previous subchapter, where the construction is done in `__init__`, and the subsequent operations in `call()`.


<!------------------------------------------------------>
<!------------------------------------------------------>
## Saving and Restoring a Model


To save a model, you can use the simple command:

```python
model.save("my_keras_model", save_format="tf")
```
Note that the format `save_format="tf"` refers to the TensorFlow Saved Model Format.

You can also load models with:

```python
model = tf.keras.models.load_model("my_keras_model")
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
```

>[!NOTE]
> You can save only the weights using `save_weights()` and `load_weights()` to obtain only the parameters. If you train models lasting days or weeks, you need to constantly save "checkpoints" to avoid future problems.


<!------------------------------------------------------>
<!------------------------------------------------------>
## Using Callbacks

**Callbacks** are arguments called during the `fit()` method and are used to customize model behaviors. _callbacks_ are called before and after each _training_, _epoch_ and processing of each _batch_. A common example is the use in `ModelCheckpoint`, where checkpoints are saved without regular intervals:

```python
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints", save_weights_only=True)
history = model.fit([...], callbacks=[checkpoint_cb])
```

>[!NOTE]
> In cases where you have a _validation set_, you can use the argument `save_best_only = True` to save only the best performance so far. It is used both to prevent overfitting and as an alternative to using Early Stopping.

Another example of a callback is `EarlyStopping`, which stops training when there is no more progress in the validation set for some epochs. The `restore_best_weights = True` argument causes the best model from the training to be returned after the training is complete. Ideally, both `ModelCheckpoint` and `EarlyStopping` can be used together.


```python
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit([...], callbacks=[checkpoint_cb, early_stopping_cb])
```

Additionally, it is possible to create custom _callbacks_, in addition to several others available through `TensorFlow` itself.
<!------------------------------------------------------>
<!------------------------------------------------------>

## Using TensorBoard for Visualization


**TensorBoard** is a tool for visualizing TensorFlow, allowing you to view learning curves, training statistics, images generated by the model, and other creations to visualize the profile of the created network.

>[!NOTE]
> In the case of Jupyter, the interface is shown on a previously configured localhost.



<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>

# Fine-Tuning Neural Network Hyperparameters


Due to the great flexibility of neural networks, many adjustments to many existing hyperparameters are necessary. A less common option is to convert the Keras model to a Scikit-Learn model using `KerasRegressor` or `KerasClassifier`. A more common way is to use the **Keras Tuner** library, which offers several solutions and has excellent integration with TensorBoard.



This is done using the `keras_tuner` library, which operates through functions adjusting hyperparameters such as integers, floats, strings, etc. over a range of values. An example of this can be seen in the code below:


```python
import keras_tuner as kt

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model
```



Still in the MNIST Fashion set, in the first part of the hyperparameter adjustment function, 4 are set (they will be explained later). Then, in the second part of the function, a `Sequential` is created starting with a _Flatten Layer_, defining the number of _Hidden Layers_, and then defining the _Output Layer_ with 10 neurons (one per class). The 4 hyperparameters set at the beginning of the function are:

- number of hidden layers (`n_hidden`): In the function, in order of what is presented, it first analyzes whether the name "n_hidden" exists in the `HyperParameters` hp object, and if not, it registers it with values ​​between 0 and 8, and returns a default value of 2 if it has not been set before;
- number of neurons per layer(`n_neurons`): acts in a similar way to the previous function;
- learning rate (`learning_rate`): unlike the two previous functions, it has the `sampling` argument, which basically defines how the steps will operate between the minimum and maximum values.
- type of optimizer (`optmizer`): decides between SGD and Adam.




After defining the `build_model` function, a common way is to `kt.RandomSearch`, and then call the `search()` method:

```python
random_search_tuner = kt.RandomSearch(
build_model, objective="val_accuracy", max_trials=5, overwrite=True,
directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)

random_search_tuner.search(X_train, y_train, epochs=10,
valid_data=(X_valid, y_valid))
```

In this code, 5 trials (`max_trials`) are run with the highest validation accuracy objective (`objective`) containing 10 epochs (`epoch`) with a directory (`directory`) "my_fashion_mnist" and a subdirectory (`project_name`) "my_rnd_search".



<!------------------------------------------------------>
<!------------------------------------------------------>
## Number of Hidden Layers


In many cases it is possible to start with a _Hidden Layer_. When you start to approach more complex cases, it is necessary to use deeper networks, using fewer neurons than shallow networks and with better performance with the same amount of data.


One way to understand why this is the case is to think about the real world. Much of the world also works in a hierarchical way. Thinking about it in a basic way, in drawings, we would have as a lower _Hidden Layer_ low-level structures like lines, in different shapes and orientations. When combining these, we would have intermediate _Hidden Layers_, containing shapes like circles, squares, triangles, etc. And then, in higher _Hidden Layers_, there would be the combination of these intermediate structures, giving shapes to things like faces, scenes. This provides the ability to **Generalize Datasets**.

Using examples of trained models, if you have a model trained to recognize faces in photos, you can use this as an intermediate layer to recognize hairstyles or eye colors in other training, also using weights and biases from the previous layers. This is called **Transfer Learning**.


<!------------------------------------------------------>
<!------------------------------------------------------>
## Number of Neurons per Hidden Layer

The first and last layers (Input and Output Layer) depend exclusively on the input and output data. In the case of MNIST Fashion, for the _Input Layer_, it requires 784 inputs (pixels 28 × 28) and in the _Output Layer_ it has 10 neurons (10 different classes).

In the case of _Hidden Layers_, the most common is to structure them in the form of a pyramid, with more layers at the base and fewer as they approach the _Output Layer_. For example, if there were 3 _Hidden Layers_ in the MNIST set, the first would have 300, the second 200 and the third 100. This was no longer used when it was discovered that using the **same number of neurons** in the layers yields almost the same result, sometimes even better.

An approach used for both the number of layers and the number of neurons is called "_stretch pants_". It consists of using a greater number of layers and neurons, and applying an _Early Stop_ to them.


<!------------------------------------------------------>
<!------------------------------------------------------>
## Learning Rate, Batch Size, and Other Hyperparameters
