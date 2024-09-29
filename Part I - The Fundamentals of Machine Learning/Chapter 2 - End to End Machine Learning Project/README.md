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

When using purely random sampling, it is possible to also fall into a bias, and for this it is important to pay attention to "stratified sampling", which is to ensure that the test set is representative of the general population for more significant factors, as in our case, the average income being the most determining factor.

With this information, stratifications are created so that there is an income category attribute, and it is necessary that each of the strata is significant both in its quantity and its size. To do this, the **`pd.cut()`** function is used to label this data.

If it is then possible to perform a stratified division, the **`model_selection`** class is used, which provides ways to divide the data set for training and testing. It is worth highlighting that it is important to check whether the proportions of the most significant value, these being the labels given the average incomes, follow the same proportion.




# Explore and VIsualize the Data to Gain Insights
## Visualizing Geographical Data
## Look for Correlations
## Experiment with Attribute Combinations



# Prepare the Data for Machine Learning
## Clean the Data
## Handling Text and Categorical Attributes
## Feature Scaling and Transformation
## Custom Transformers
## Transformation Pipelines

 
# Select and Train a model
## Train and Evaluate on the Training Set
## Better Evaluation Using Cross-Validation



# Fine-Tune Your Model
## Grid Search
## Randomized Search
## Ensemble Methods
## Analyzint the Best Models and Their Errors
## Evalue Your System on the Test Set



# Launch, Monitor and Maintain Your System


