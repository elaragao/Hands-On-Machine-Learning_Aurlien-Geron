# Decision Trees

Decision trees are capable of performing both classification and regression, as well as multi-output tasks. They are capable of handling complex datasets.

The chapter will briefly discuss their functionality, how to train them, and how to visualize the predictions made.

# Training and Visualizing a Decision Tree

The .ipybn file will demonstrate how the decision tree operates on the `load_iris` dataset, using the `DecisioTreeClassifier` class. The output can be viewed using the `export_graphviz` class.

# Making Predictions

For decision trees, it is important to understand the structure of their output, as shown in the figure below:

[Image]

- Root Node: Initial part where the first values ​​to be differentiated are defined. Depth 0, located at the top. Asks if the length of the flower is less than or equal to 2.45cm.

- Path: The path that moves indicates whether what was promised in the previous node is **True** or **False**. If it went to the **Left** path, it corresponds to **True**, that is, in our case, the size was equal to or less than 2.45cm. If it went to the **Right** path, it indicates that it is **False**, that is, it is not equal to or less than 2.45cm.

- Leaf Node: Node that has no more branches, no more child nodes, therefore, it does not ask any more questions;

- Split Node: Node that culminates in more branches (depth n).

In addition, a tree node has the following attributes:

- `samples`: Counts how many training instances are in this node;
- `value`: Indicates how many instances of each class are in this node;
- `gini`: Measures the degree of impurity of the nodes (pure == 0).

The equation indicates the Gini Scores equation at the $i^{th}$ node:

```math
G_{i} = 1 - \sum_{k=1}^{n} p_{i,k^{2}}
```

# Estimating Class Probabilities

It is possible to use decision trees to estimate the probabilities that instances belong to. For example, suppose you want to analyze a flower with petals that are 5 cm long and 1.5 cm wide:

```python
tree_clf.predict_proba([[5, 1.5]]).round(3)
```

This will return an array indicating the values ​​corresponding to the classes in order. The code below will return as output the class with the highest probability, _Iris versicolor_ (class 1).

```python
tree_clf.predict([[5, 1.5]])
```

# The CART Training Algorithm

```math
J(k, t_{k}) = \frac{m_{left}}{m} G_{left} + \frac{m_{right}}{m} G_{right}  
```


# Computational Complexity

The training complexity is O(n × m log$_{2}$(m)).

The log$_{2}$(m) corresponds to the binary logarithm of m. The algorithm compares all features (or fewer if `max_features` is changed) in all samples at each node, then indicates the values ​​of n × m, resulting in O(n × m log$_{2}$(m)), making it a very fast algorithm.


# Gini Impurity or Entropy?

Like Gini Impurity, Entropy is also a criterion that can be used to measure impurity, being set in the hyperparameter `criterion = 'entropy'`. The higher the entropy, the greater the disorder and impurity, and the lowest possible entropy is zero. It is calculated at the $i^{th}$ node as:

```math
H_{i} = - \sum ^{n} _{k=1}_{p_{i},k \neq 0} p_{i},k log_{2}(p_{i},k)
```

In comparison, both Gini and Entropy produce similar trees. Gini tends to be faster in most cases, but where the measures differ is that Gini tends to isolate classes into their own branches, while Entropy produces more balanced branches.

# Regularization Hyperparameters



# Regression



# Sensitivity to Axis Orientation



# Decision Trees Have a High Variance
