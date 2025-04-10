# Decision Trees

Decision trees are capable of performing both classification and regression, as well as multi-output tasks. They are capable of handling complex datasets.

The chapter will briefly discuss their functionality, how to train them, and how to visualize the predictions made.


<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->

# Training and Visualizing a Decision Tree

The .ipybn file will demonstrate how the decision tree operates on the `load_iris` dataset, using the `DecisioTreeClassifier` class. The output can be viewed using the `export_graphviz` class.

<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
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

<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
# Estimating Class Probabilities

It is possible to use decision trees to estimate the probabilities that instances belong to. For example, suppose you want to analyze a flower with petals that are 5 cm long and 1.5 cm wide:

```python
tree_clf.predict_proba([[5, 1.5]]).round(3)
```

This will return an array indicating the values ​​corresponding to the classes in order. The code below will return as output the class with the highest probability, _Iris versicolor_ (class 1).

```python
tree_clf.predict([[5, 1.5]])
```

<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
# The CART Training Algorithm

_Classification and Regression Tree_ (CART) trains decision trees by first splitting the training instances into two parts using the feature $k$ and the threshold $t_{k}$ (petal length $\leq$ 2.45cm). The equation below gives the cost function that the algorithm tries to minimize:

```math
J(k, t_{k}) = \frac{m_{left}}{m} G_{left} + \frac{m_{right}}{m} G_{right}
```

- $G_{left/right}$: impurity of the left/right subsets.

- $m_{left/right}$: number of instances in left/right.

If hyperparameters are not used to limit the tree growth (`min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, and `max_leaf_nodes`), the algorithm will continue until it reaches maximum depth, or there are no more ways to split to reduce impurity.

<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->


# Computational Complexity

The training complexity is O(n × m $log_{2}$(m)).

The $log _{2}$ (m) corresponds to the binary logarithm of m. The algorithm compares all features (or fewer if `max_features` is changed) in all samples at each node, then indicates the values ​​of n × m, resulting in O(n × m $log _{2}$ (m)), making it a very fast algorithm.



<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->

# Gini Impurity or Entropy?

Like Gini Impurity, Entropy is also a criterion that can be used to measure impurity, being set in the hyperparameter `criterion = 'entropy'`. The higher the entropy, the greater the disorder and impurity, and the lowest possible entropy is zero. It is calculated at the $i^{th}$ node as:

```math
H_{i} = - \sum ^{n} _{k=1} p_{i,k} log_{2}(p_{i,k})
```

In comparison, both Gini and Entropy produce similar trees. Gini tends to be faster in most cases, but where the measures differ is that Gini tends to isolate classes into their own branches, while Entropy produces more balanced branches.


<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->

# Regularization Hyperparameters

Decision trees make few assumptions when training, if left without restrictions the structure will possibly overfit. This type of model is called a _Nonparametric Model_ because the number of parameters is not determined before training, so the model is free to perform.

The _Parametric Model_ has a predetermined number of parameters, therefore, the degree of freedom is limited, reducing the risk of overfitting, but increasing underfitting. The process of applying restrictions to avoid overfitting is called **Regularization**. What is usually regularized is the **Decision Tree Depth**. The most common hyperparameters are:

- `max_depth`: Maximum depth that the tree will go;
- `max_features`: Maximum number of features evaluated for divisions in the nodes;
- `max_leaf_nodes`: Maximum number of leaves;
- `min_samples_split`: Minimum number of samples that a node must have to be split;
- `min_samples_leaf`: Minimum number of samples for a leaf to be created;
- `min_weight_fraction_leaf`: Similar to the previous one, but expressed as a fraction of the number of weighted instances.
  

<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
# Regression

It is also possible to perform regression using decision trees using the Scikit-Learn `DecisionTreeRegressor` class. The image below represents the result of the tree:

```math

J(k,t_{k}) = \frac{m_{left}}{m} MSE_{left} + \frac{m_{right}}{m} MSE_{right} 

```

Where:

```math

MSE_{node} = \frac{\sum_{i \: \in \: node} (\hat{y}_{node} - y^{(i)})^{2}}{m_{node}}

\: \: \: \: \: \: \:
and
\: \: \: \: \: \: \:

\hat{y}_{node} = \frac{\sum_{i \: \in \: node} y^{(i)}}{m_{node}}

```


The algorithm works, instead of trying to minimize the impurity, to minimize the MSE. They are also prone to overfitting because they do not have regularization. Below is an example of a comparison between an analysis with restrictions of `min_samples_leaf=10` and without:

[Image]

<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->

# Sensitivity to Axis Orientation

One drawback of decision trees is that they tend to use orthogonal decision boundaries (splits are always perpendicular to the axis), which makes views at different angles, such as 45º, less straightforward.

One way to scale is to apply Principal Coefficient Analysis (PCA), which reduces the correlation between features, which is usually better for trees. An example can be seen in the code below:

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(StandardScaler(), PCA())
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(X_iris_rotated, y_iris)
```

<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->
<!------------------------------------------------------------------------------------------------------------------------------->

# Decision Trees Have a High Variance

Decision trees have a high variance, meaning that with small changes in data or hyperparameters, they can produce very different models. Since the Scikit-Learn model is stochastic, even by modifying the same decision tree with the same data and hyperparameters, it is possible to produce a different model. The way to avoid this problem is by calculating the average of the predictions, using the **Random Trees** model.
