# Training Deep Neural Networks

The previous chapter presented solutions to somewhat simpler problems. Now, if one wants to address more complex problems, such as detecting multiple objects in images, more elaborate methods will be necessary.

Problems such as slow training, little data, _overfitting_ of thousands of parameters, gradient adjustments, etc. This chapter will address these and other possible problems, as well as techniques to overcome and solve them.

<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>
# The Vanishing/Exploding Gradients Problems

In the backpropagation algorithm, after calculating the error gradient for each network parameter, this gradient is propagated from the output layer to the input layer, updating the parameters at each step to minimize the cost function. However, as the gradient is transmitted to the initial layers of the network, it tends to decrease (or, in some cases, increase too much), which generates two main problems:

_Evanescent Gradients (Vanishing Gradients)_:
As the gradient propagates to the lower layers, it becomes very small. This causes the weights of these layers to be practically not updated, preventing the training from converging to a good solution. In short, if the gradients "disappear", the initial layers cannot learn effectively.

_Explosive Gradients_:
In some situations – more common in recurrent neural networks – the gradients can increase uncontrollably as they pass through the layers, resulting in very large updates to the weights and causing the training to diverge completely.

Glorot and Bengio (2010) published a paper that pointed out that the combination of the sigmoid activation function and the weight initialization scheme of the time (using a normal distribution with mean 0 and standard deviation 1) led to a progressive increase in the variance of the values ​​as the data passed through each layer. With the sigmoid, this greater variance causes the function to saturate – that is, for very large values ​​(positive or negative) the sigmoid output approaches 0 or 1, where its derivative is almost zero. Thus, during backpropagation, there is not enough gradient to update the weights of the lower layers, impairing the network's learning.
<!------------------------------------------------------>
<!------------------------------------------------------>
## Glorot and He Initialization



<!------------------------------------------------------>
<!------------------------------------------------------>
## Better Activation Functions



<!------------------------------------------------------>
<!------------------------------------------------------>
## Batch Normalization



<!------------------------------------------------------>
<!------------------------------------------------------>
## Gradient Clipping





<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>
# Reusing Pretrained Layers



<!------------------------------------------------------>
<!------------------------------------------------------>
## Transfer Learning with Keras



<!------------------------------------------------------>
<!------------------------------------------------------>
## Unsupervised Pretraining



<!------------------------------------------------------>
<!------------------------------------------------------>
## Pretraining on an Auxiliary Task


<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>
# Faster Optimizers



<!------------------------------------------------------>
<!------------------------------------------------------>
## Momentum



<!------------------------------------------------------>
<!------------------------------------------------------>
## Nesterov Accelerated Gradient



<!------------------------------------------------------>
<!------------------------------------------------------>
## AdaGrad



<!------------------------------------------------------>
<!------------------------------------------------------>
## RMSProp



<!------------------------------------------------------>
<!------------------------------------------------------>
## Adam



<!------------------------------------------------------>
<!------------------------------------------------------>
## AdaMax



<!------------------------------------------------------>
<!------------------------------------------------------>
## Nadam



<!------------------------------------------------------>
<!------------------------------------------------------>
## AdamW



<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>
# Learning Rate Scheduling



<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>
# Avoiding Overfitting Through Regularization



<!------------------------------------------------------>
<!------------------------------------------------------>
## ℓ1 and ℓ2 Regularization



<!------------------------------------------------------>
<!------------------------------------------------------>
## Dropout



<!------------------------------------------------------>
<!------------------------------------------------------>
## Monte Carlo (MC) Dropout



<!------------------------------------------------------>
<!------------------------------------------------------>
## Max-Norm Regularization



<!------------------------------------------------------>
<!------------------------------------------------------>
<!------------------------------------------------------>
# Summary and Practical Guidelines
