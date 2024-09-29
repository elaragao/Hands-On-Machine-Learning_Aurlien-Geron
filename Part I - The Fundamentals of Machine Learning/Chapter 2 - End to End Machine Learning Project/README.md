In this chapter, a simple and complete example will be executed to illustrate the step-by-step process of a machine learning project. The project will illustrate a real-state company. The following steps will be:

1. Observe the complete scope
2. Obtain data
3. Analyze and visualize the data
4. Prepare the data for the ML model
5. Select and train the model
6. Improve the model
7. Present the solution
8. Send, monitor and maintain the system
   

> [!NOTE]
> It is worth highlighting that, for the study of this chapter, notes were made in the READ.ME file, in the code provided by Aurélien Géron and in the summarized code that will be made to study the chapter in question.

# Working with Real Data
The best way to learn about Machine Learning is to train with real data sets. These can be found in abundance on the internet. Some examples of these are:

+ Popular open data repositories:
 	+ [OpenML](OpenML.org)
 	+ [Kaggle](Kaggle.com) 
	+ [PapersWithCode](PapersWithCode.com)
 	+ [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml)
 	+ [Amazon’s AWS datasets](https://registry.opendata.aws/)
	+ [TensorFlow datasets](https://tensorflow.org/datasets)

+ Meta portals (they list open data repositories):
	+ [DataPortals](dataportals.org)
	+ [OpenDataMonitor](OpenDataMonitor.eu)


+ Other pages listing many popular open data repositories:
	+ [Wikipedia’s list of machine learning datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
	+ [Quora](quora.com)
	+ [The datasets subreddit](https://www.reddit.com/r/datasets/)

The data used will be California Housing Prices from the StatLib repository [add link], based on the 1990 census. Likewise, the discussions that took place in order to exemplify each point of a real project, will also follow according to this data, so, imagine this as a real project from its beginning to its delivery.


# Look at the Bit Picture
In order to analyze the problem, it is important to know what data we have. In this case, to model the price of houses in this state, we obtained metrics such as population, median income, and average price of houses for each group of blocks (with populations between 600-3000) in California, also called districts.


## Frame the Problem
The first question is: what is the business objective? How does the company benefit from this model? These questions are crucial to understanding what will be used to solve the problem, how it will be evaluated, and the time available for this task.

The answer to the objective will be that the model's output will be used to feed another Machine Learning system to then analyze whether or not it is advantageous to invest in this area. The problem arises because at this stage of stipulating the average price for subsequent surveys, it is done manually by experts, which takes a lot of time and reduces the time for the company to invest in these places, which makes the process expensive, time-consuming, and inefficient, in addition to, in this case, having an error of around 30%. Therefore, the data that was considered good for this estimate is the census data.

Therefore, it is necessary to define how the system will be designed, as was analyzed in the previous chapter.

Will the learning be supervised, unsupervised, semi-supervised, self-supervised, or reinforcement? The learning in question will be **supervised**, given the presence of labeled data (if in our case, average housing price data).

Does the task consist of regression, classification or another category? The task consists of **regression**, since it requires predicting a value, and in this case, multiple regression, since there are several resources.

Will batch or online learning techniques be used? If there is enough data, there are no abrupt changes and there is enough memory space, the **simple batch** system should be effective.


## Select Performance Measure
A typical measure is the Root Mean Squared Error (RMSE), which gives an idea of ​​the error during predictions, giving greater weight to larger errors.

```math
RMSE(\mathbf{X},h) = \sqrt{
\frac{1}{m}
\sum_{i=1}^{m}
(h(x^{(i)})-y^{(i)})^{2}
}
```
Where:
- **h** is the system’s prediction function, also called a _hypothesis_.
- **X** is a matrix containing all the feature values (excluding labels) of all dataset.
- **m** is the number of instances in the dataset.
- **x** is a vector of all the feature values (excluding the label) of the i<sup>th</sup> instance.
- **y** is the labels of the dataset of the i<sup>th</sup> instance.



Although RMSE is the most widely used measure because it highlights outliers, the Mean Absolute Error (MAE) is also used.

```math
MAE(\mathbf{X},h) = 
\frac{1}{m}
\sum_{i=1}^{m}
|
h(x^{(i)})-y^{(i)}
|
```


## Check the Assumptions
Finally, it is important to check with everyone involved whether the system designed for this problem will meet the need. For example, thinking about a system in which there is no need for exact prices, classifying prices as "cheap", "medium" and "expensive" would be enough if they did not require a close approximation of the real values. Now, having confirmed that we have chosen the correct model, we will move on to the next sections.


# Get the Data
Links to the codes for this book are available [online](https://github.com/ageron/handson-ml3), and the template for the current problem will be followed for this chapter.


## Running the Code Examples Using Google Colab
The current instructions, although they are for implementation in Google Colab, will be implemented in Jupyter. These instructions will be briefly explained in topics in this chapter.
 - Edit the first cell, leaving it as the title (set the cell as "Markdown" and add the "#" sign before the sentence).
 - Create a new cell by clicking on the "+" option in the toolbar, then print "Hello World".



## Saving Your Code Changes and your Data
To save the progress in Google Colab, do the following: File → "Save a copy in Drive". If you prefer, you can also download the file, for example: File → Download → "Download.ipynb". The file can be accessed through Drive or by uploading the file from the computer.

## The Power and Danger of Interactivity
On Jupyter-style platforms, where you can run individual cells, you have the huge benefit of being able to work individually with code snippets, but the problem is that you sometimes forget to run specific cells for the program to work.


## Book Code vs Notebook Code
The codes in the Notebook are more up-to-date due to library changes, and, obviously, because it is not possible to change what is written in the book. In addition, the book also has extra explanations, such as how to handle some graphs, for example.



## Download the Data
For this case, the compressed file "*housing.tgz*" will be used, a **csv** (comma separated values) file. Instead of downloading the file and decompressing it, it is more recommended to write a function for this. The reason is both for practice and so that, in real cases, the data can be updated, and the automation of collecting and using the files in a more dynamic way is essential.



## Take a Quick Look at the Data Structure
The **`head()`** method allows you to see the first 5 rows of data. In our current case, these are: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `edian_house_value`, and `ocean_proximity`.

The **`info()`** method provides a quick description of the data, which is the number of rows, the attributes, and the number of non-null values. By following the code along with these instructions, you will notice that the total_bedrooms column has 20,433 non-null values, meaning that there are 207 districts without this data.

You can see that the file types are all numeric (floats, in our case), with the exception of ocean_proximity, which in our case, is a categorical attribute, which we can see through the **`value_counts()`** method.

The **`describe()`** method provides a summary of the numeric attributes of the data. We see parts that are self-explanatory, such as The count, mean, min, and max. The "std" stands for *standard deviation*, which measures the dispersion of values. The values ​​25%, 50% (or median) and 75% represent the percentiles, or quartiles; these indicate the value below which a certain percentage of observations in a group of observations fall. For example, 25% of districts have a housing_median_age less than 18, while 50% are less than 29 and 75% are less than 37.

Another way to analyze numerical values ​​is to use the **`hist()`** method, which will show histograms of the numerical data.


## Create a Test Set
When separating training and test sets, it is necessary to apply a **`random.seed`** in order to always make the data "shuffled in the same way", otherwise, this "shuffling" would be done randomly every time the program was run. Remember that shuffling is important to improve the generalization of the data to give a fairer division and avoid unwanted patterns that the initial order could give, and it is important to always have an **index** in this data, and whenever new data is included in the analysis, that these are added below the last values ​​that existed. It is also important to emphasize again that the instance of the divisions for training and testing are usually **80% - 20%**.

>[!NOTE]
>**`random.seed`** is a way of defining a pseudo randomization of the data, which will always be shuffled in the same way according to the number defined for this operation, for example, **`np.random.seed(42)`**.

The most common way to separate data is to use **`train_test_split()`**, provided by Scikit-Learn.

When using purely random sampling, it is possible to also fall into a bias, and for this it is important to pay attention to _stratified sampling_, which is to ensure that the test set is representative of the general population for more significant factors, as in our case, the median income (column `median_income`) being the most determining factor.

With this information, stratifications are created so that there is an income category attribute, and it is necessary that each of the strata is significant both in its quantity and its size. To do this, the **`pd.cut()`** function is used to label this data.

If it is then possible to perform a stratified division, the **`model_selection`** class is used, which provides ways to divide the data set for training and testing. It is worth highlighting that it is important to check whether the proportions of the most significant value, these being the labels given the average incomes, follow the same proportion.




# Explore and Visualize the Data to Gain Insights
At this point, we will take a deeper look at the data. If the set is large enough, we can only look at the training data. It is important to have a copy of the original set separate from the data to avoid any problems.

## Visualizing Geographical Data
Since the data is directly related to the location, we will analyze the geographic information of the place, trying to analyze where the population is most concentrated. [code]

One way to deepen the analysis is to combine the scatter plot with the heat map, to analyze not only the population, but also the average value of the houses.

```python
housing.plot(
	kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100, label="population",
	c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10, 7)
)
 plt.show()
```

## Look for Correlations
One of the most common ways to analyze the data initially is to analyze its correlations. A correlation is the influence of the value of one variable on another, and in this case, the relationship between the available variables and the variable of interest will be analyzed. 

```python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```


If there are many columns, only the most relevant ones will be analyzed in more detail. It is important to note that correlation analysis only analyzes linear relationships between values.


## Experiment with Attribute Combinations
Another way to perform data analysis is to see how correlations behave with combined attributes of the data. A visible example in the data set we are analyzing would be data such as number of bedrooms and total number of rooms in a district. More interesting values ​​for analysis would be the number of rooms per household, as well as the number of bedrooms per room.

```python
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
```


# Prepare the Data for Machine Learning
In order to prepare data for the Machine Learning process, it is best to set up functions so that this can be done in a simpler and more replicable way. It is also necessary, again, to create a copy of the data that was previously created to avoid any possible damage to its integrity.

## Clean the Data
Many datasets have missing data, and this tends to harm the algorithms. To do this, one of the three options is usually chosen:

1. Delete the rows that have missing data.

2. Delete the columns that have missing data.

3. Set the missing values ​​to something (zero, most frequent value, mean, median, constant value, etc.). This is called **imputation**

For the cases mentioned, the following lines are usually used:

1. `df.dropna(subset=["col_name"], inplace = True)`

2. `df.drop("col_name", axis = 1)`

3. `df["col_name"].fillna(median, inplace = True)`

It is preferable to choose the third option because it is the least destructive. The Scikit-Learn library has the **`SimpleImputer`** class, which replaces the missing values ​​not only in the training set, but in all sets. In the example, the median will be used as the parameter, so only numeric attributes will be used initially, eliminating the ocean_proximity column.

To import the imputer, after selecting it in the library, the usual command for this is:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")
```


In order to eliminate non-numeric columns, the following is used:

```python
df_num = df.select_dtypes(include=[np.number])
```
And then, the median is applied, which will calculate the median for each of the numeric attributes:
```python
imputer.fit(df_num)
```



## Handling Text and Categorical Attributes
In some cases, it is also necessary to deal with non-numeric cases. In the case of the dataset used in this study, the ocean_proximity column requires this type of treatment because it is a categorical attribute. Initially, the categorical attribute is separated:

```python
df_cat = df[["cat_column"]]
```

One way to deal with this is by using the OrdinalEncoder class from the Scikit-Learn library, as in the excerpt shown below:

```python
ordinal_encoder = OrdinalEncoder()
df_cat_encoder = ordinal_encoder.fit_transform(df_cat)
```

This will return a 1-dimensional array, where the categorical data will now be represented by numbers corresponding to each category. In Machine Learning analyses, algorithms can understand that close values ​​are more similar than more distant ones (for example, 0 and 1 are more similar than 0 and 4, which does not translate into reality in many cases).

One way to get around this problem is through binary classifications, which would be the representation of 0 and 1 for each of the categorical values, with 0 indicating that it does not have the attribute, and 1 indicating that it has the attribute. This is called "**one-hot encoding**", with attribute 1 being the "hot" attribute, and 0 being the "cold" attribute. It can be used by importing the **`OneHotEncoder`** class as shown in the code below:

```python
cat_encoder = OneHotEncoder()
df_cat_1hot = cat_encoder.fit_transform(df_cat)
```

This will return a **sparse matrix** instead of a 1D array. This model can also be useful when there are hundreds or thousands of different categories in the same column, as it saves memory.

Similarly, Pandas' **`get_dummies()`** function also allows you to perform this binary classification operation:

```python
df_test = pd.DataFrame({"cat_column": ["col1","col2"]})
pd.get_dummies(df_test)
```

In terms of comparison between the two forms of binary classification, get_dummies() sees only two categories, so it produces two columns, while OneHotEncoder produced one column per learned category, in the correct order, and also has the ability to detect unknown categories and generate exceptions.




## Feature Scaling and Transformation
Column values ​​with very different scale values ​​can lead Machine Learning algorithms to error. Scale scaling is necessary to reduce bias in analyses. The two most common methods are min-max scaling and standardization. Both operations can be obtained through SciKit Learn using the preprocessing class.

**Min-Max scaling**, or **normalization**, returns column values ​​between 0 and 1 (or -1 to 1 depending on the request)

```math
Normalization = \frac{x - x_{min}}{x_{max} - x_{min}}
```
Where:
- **x** is the current value.
- $x_{min}$ is the min value of the column.
- $x_{max}$ is the max value of the column.

```python
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
df_num_min_max_scaled = min_max_scaler.fit_transform(df_num)
```

**Standardization** scaling does not restrict to specific ranges like normalization, and is less affected by outliers. 
```math
Standarization = \frac{x - \overline{x}}{\sigma}
```
Where:
- **x** is the current value.
- $\overline{x}$ is the mean of the column.
- $\sigma$ is the standart deviation of the column.

```python
std_scaler = StandardScaler()
df_num_std_scaled = std_scaler.fit_transform(df_num)
```

In cases where the distributions of a feature have a **heavy tail** (when values ​​far from the mean are not rare), scaling methods will not be as effective, so before scaling it is important to make the distribution approximately symmetric. Two common ways to do this are to replace the value by its **square root**, or, in case the values ​​follow the power laws, by its **logarithmic value**.

Another way to deal with this is through **bucketizing**, which consists of dividing the data into groups of approximately equal sizes, and instead of using the data value, using something like its percentile, in order to have a more balanced distribution.

In multi-peaked or multi-modal distributions, **bucketizing** can be done by treating the group IDs as categories, rather than numeric values.

Another approach for multimodal distributions is to add features for each of the modes (or just the main ones). The similarity measure is usually calculated using a **radial basis function** (RBF) — any function that depends only on the distance between the input value and a fixed point, the most commonly used being the Gaussian. Usually, the `rbf_kernel()` function from Scikit-Learn is used.

Moving away from just looking at the input values, we also need to look at the target values. If the target has a heavy-tailed distribution, it is also possible to replace the value with the logarithm, but in this case, the logarithm will be predicted. For these cases, there is a method in Scikit-Learn called **`inverse_transform()`**, which makes these transformations easier. For example, assuming that the StandardScaler method was used and the model is trained, the resulting data can be transformed to the original scale with `inverse_transform()`.

Another simpler approach to this is to use TransformedTargetRegressor. From the regression inputs and the required transformation, it will automatically perform the inverse transform to produce the prediction.


## Custom Transformers
You can also customize the transformations you want to make. From the SciKit-Learn library, using the FunctionTransformer method, it is possible to see an example of a logarithmic function transform together with the inverse transform:

```python
from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(df[["column"]])
```

Another example is with the use of hyperparameters, using as an example the Gaussian RBF for property ages similar to 35 years:

```python
rbf_transformer = FunctionTransformer(rbf_kernel,kw_args=dict(Y = [[35.]], gamma = 0.1))
age_simil_35 = rbf_transformer.transform(df[["df_median_age"]])
```

If it is necessary for the In order for the transformer to be trainable, it is necessary to create custom classes, requiring three methods: `fit()`, `transform()` and `fit_trasnform()`. This is better explained in the .ipybn file.


## Transformation Pipelines
Because several steps need to be executed to transform the data in the correct order, a more effective way to do this is through **Pipelines**, which is also a class provided by Scikit-Learn. A basic example is shown below:

```python
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
	("impute", SimpleImputer(strategy="median")),
	("standardize", StandardScaler()),
])
```

The Pipeline constructor takes a list of name/estimator pairs (2-tuples) defining a sequence of steps. The names can be anything you want, as long as they are unique and do not contain double underscores (__). The estimators must all be transformers (i.e., they must have a fit_transform() method), except the last one, which can be anything: a transformer, a predictor, or any other type of estimator.

If you do not want to define names, you can use the **`make_pipeline`** class, as seen below:

```python
from sklearn.pipeline import make_pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
```

In a Pipeline, when calling a `fit()` method, it automatically returns a `fit_transform()` sequentially on all transformers until the final estimator, which calls only the `fit()` method. If there is a predictor at the end, it would pass the result to the `predict()` method. It is interesting to note that the behavior of Pipelines can also behave like a list, for example, when declaring pipeline[1], it will return the second values ​​of the index, calling pipeline[:-1], it will call all values ​​except the last one, and so on.

More complex uses will be seen in the .ipybn files.


 
# Select and Train a model
After the data exploration, data cleaning and preprocessing steps, it is possible to try to apply the models for training.


## Train and Evaluate on the Training Set
At first, it is ideal to try more basic and simple models to evaluate the results, so we will start a test with the basic linear regression model:
```python
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(df, df_labels)
```

After that, it is possible to both check the data by comparing it with the values ​​of the labels and see the performance measure, in the case of this project, RMSE:
```python
from sklearn.metrics import mean_squared_error

lin_rmse = mean_squared_error(df_labels, df_predictions, squared=False)
```

Assuming that it was not effective due to underfitting, some of the solutions are to add more resources to the algorithm or try a more powerful one. At the moment, the simplest option is to opt for the second option.

If you then choose to use a DecisionTreeRegressor, you will use the following code:
```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(df, df_labels)
```

Assuming that, from the RMSE test, the model made no errors, clearly demonstrating overfitting. This problem will be further explored by analyzing this model in more depth.



## Better Evaluation Using Cross-Validation
An alternative to better train the model is using **cross-validation**, which is possible through the Scikit-Learn library in the` k_fold cross-validation` class, allowing the training sets to be divided into subsets called folds. Then, it trains and evaluates the decision tree model 10 times, choosing a different fold for evaluation each time and using the other 9 folds for training. The result is a matrix containing the 10 evaluation scores.

```python
from sklearn.model_selection import cross_val_score
tree_rmses = -cross_val_score(tree_reg, df, df_labels,
scoring="neg_root_mean_squared_error", cv=10)
```

A common occurrence is to have a very low error in the training sets and in the validation sets demonstrate a relatively high error due to overfitting.

The best alternative, then, is to use a different training model, in this case, Random Forest:

```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(
	preprocessing,
	RandomForestRegressor(random_state=42)
)

forest_rmses = -cross_val_score(
	forest_reg, df, df_labels,
	scoring="neg_root_mean_squared_error", cv=10
)
```
Assuming that this model, even though it performed better, still had a large discrepancy between the RMSE of the training and validation, still indicating the presence of overfitting. Possible solutions that do not involve delving into the chosen model are to use others, focusing on selecting a few promising models (somewhere between two and five). But to continue the project, we will decide that this is the chosen model and we will try to optimize the hyperparameters.


# Fine-Tune Your Model
This stage begins to be carried out from the moment a series of promising models for training is decided upon.

## Grid Search
Grid Search works by optimizing hyperparameters, listing them and testing them to see which one improves the model's performance.

```python
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline(
	[
		("preprocessing", preprocessing),
		("random_forest", RandomForestRegressor(random_state=42)),
	]
)

param_grid = [
	{
		'preprocessing__geo__n_clusters': [5, 8, 10],
	 	'random_forest__max_features': [4, 6, 8]
	},
	{
		'preprocessing__geo__n_clusters': [10, 15],
	 	'random_forest__max_features': [6, 8, 10]
	},
]

grid_search = GridSearchCV(
	full_pipeline,
	param_grid, cv=3,
	scoring='neg_root_mean_squared_error'
)

grid_search.fit(df, df_labels)
```

Note in the function above that it is not only possible to adjust the hyperparameters within the model, but it is also possible to adjust them in the estimator. The hyperparameter is detected from the double underscore (__), for example, `preprocessing__geo__n_clusters`; the first signal is identified by locating the estimator "`preprocessing`" and searching for it in the `ColumnTransformer`, then searching for the transformer, in this case identified as "geo", the `ClusterSimilarity` transformer.

In order to identify the number of trainings that occurred, we initially observe the two dictionaries. The first dictionary has two lists with three attributes each, totaling 3×3=9; while the second has a list with three attributes and another with two, therefore, 2×3=6. The total number of analyses performed will then be the sum of these, 9 + 6 = 15. Finally, the pipeline will be trained 3 times per training round due to 3-fold cross-validation, therefore, 15 × 3 = 45 total training rounds.


## Randomized Search
Randomized Search operates in a similar way to Grid Search, but instead of trying all possible combinations, it evaluates a fixed number of combinations, selecting a random value for each hyperparameter in each iteration. It has advantages over Grid Search, such as not being affected in processing time by irrelevant hyperparameters, and often requiring fewer attempts for better fits.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
	'preprocessing__geo__n_clusters': 
	  randint(low=3,  high=50),
	 'random_forest__max_features': randint(low=2, high=20)
}

rnd_search = RandomizedSearchCV(
	 full_pipeline,
	param_distributions=param_distribs,
	n_iter=10, cv=3,
	scoring='neg_root_mean_squared_error',
	random_state=42
)
 
rnd_search.fit(housing, housing_labels)
```

## Ensemble Methods
This describes the combination of groups, or ensembles, of models that perform best. This will be discussed further in later chapters.


## Analyzint the Best Models and Their Errors
Analyzing the models allows us to see the relative importance of each attribute. This allows us to make more accurate predictions, focusing on those that have the greatest weight and ignoring or removing those that have less relevance.

```python
final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
sorted(
	zip(feature_importances,
	final_model["preprocessing"].get_feature_names_out()),
	reverse=True
)
```

## Evalue Your System on the Test Set
After performing all the steps, it is important to evaluate the final model using the test set. The way to do this is demonstrated in the codes below:

```python
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
```

Something useful for this type of analysis is to apply a **confidence interval**, usually 95%. This confidence interval approximately predicts a range of error for 95% of the cases. It is a way of expressing the uncertainty around this point estimate, which was obtained, in our case, by the RMSE.

```python
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(
	stats.t.interval(confidence, len(squared_errors) - 1,
	loc=squared_errors.mean(),
	scale=stats.sem(squared_errors))
)
```


# Launch, Monitor and Maintain Your System
Once the model is finally ready, it must be sent to the production environment. One of the basic ways to save the model is by using the **`joblib`** library:
```pyton
import joblib
joblib.dump(final_model, "my_california_housing_model.pkl")
```

It is worth noting that, in a production environment, it is not only Python that is used, but other languages ​​to program an interface so that it is intuitive to set the desired values ​​for the analysis and facilitate the user's operation. This is usually done through a REST API.

Platforms such as Google Cloud allow **joblib** to be loaded into it. In addition, it is also important to write codes to monitor and check the system's performance and send alerts when errors occur. For the system we designed, it is important to have subsequent information, such as real sales prices of the houses and value updates, so that everything can be updated and optimized when necessary. It is worth noting that it is important to have backups during updates to prevent major future damage.

