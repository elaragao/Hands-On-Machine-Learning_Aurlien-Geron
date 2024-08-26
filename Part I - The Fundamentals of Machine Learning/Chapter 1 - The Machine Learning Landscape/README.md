# What is Machine Learning?

Machine Learning is the science of programming computers to learn from data without being explicitly programmed.

A commonly seen example is spam filtering, where the algorithm learns to flag emails as either spam or ham (non-spam).

The examples the system uses are called the *training set*. Each of these examples is referred to as a training instance or sample. The part of the system that makes predictions is called the *model*. To evaluate models, performance metrics such as *accuracy*, for example, are used.


# Why Use Machine Learning?

Using the spam example again, if a traditional programming approach were used, the programmer would need to individually identify each subject commonly found in spam emails, store them, and then apply a code to filter them, updating it again each time the spam format changes.
By using Machine Learning algorithms, with a set of user-identified email examples, the model would recognize patterns and classify the emails as either ham or spam.

In summary, Machine Learning can be used in cases where:

 * Problems have solutions that require very fine adjustments or involve lengthy rules.
 * Data is dynamic (frequent updates and new data emerge rapidly).
 * Insights are needed in complex problems with large amounts of data.


# Application Examples

 * Analyzing images on production lines and classifying them
 * Detecting tumors
 * Automatically classifying articles
 * Automatically flagging offensive comments in forum discussions
 * Summarizing long documents
 * Predicting future profits based on past profits
 * Making an app respond to voice commands
 * Detecting credit card fraud
 * Segmenting customers based on purchases to design personalized marketing strategies for segments
 * Representing a complex, high-dimensional dataset in a clear and insightful diagram
 * Recommending products to customers based on previous purchases
 * Building bots for games


# Types of Machine Learning Systems

Learning systems can be classified as follows: how they are supervised during training, whether they can learn incrementally in real-time, and whether they work by comparing new data points or by detecting patterns in the training data and building predictive models. These classifications are not exclusive and can be combined.

## Training Supervision
This classification refers to the type and amount of supervision involved.

### Supervised Learning
In supervised learning, the algorithm has access to the desired output values in the data, also known as *labels*. There are many examples, but we will focus on two. The first is **classification**, such as classifying emails as either spam or ham. The second is **linear regression**, which is used to predict numerical values, or targets, such as predicting car prices based on data like year, mileage, and brand.


### Unsupervised Learning
Unsupervised learning involves data that is *unlabeled*. Some examples of these algorithms are briefly discussed below.

**Clustering** works by trying to detect groups of similar data, seeking to find connections between them. For example, in an analysis of website visitors, it might identify the age groups that most frequently view content, as well as the times they do so (considering a site for stories, one could also consider which age groups tend to read which types of stories).

**Visualization algorithms** are another example, as they produce graphical representations of data to observe how the data is organized and to identify patterns that were not previously apparent.

**Dimensionality reduction** also allows for pattern identification, such as correlating the age of cars with their mileage, as mentioned earlier in linear regression. This is known as feature extraction.

**Anomaly detection** is another commonly used algorithm, often used to detect potential credit card fraud. Similarly, **novelty detection** is also employed.
Finally, **association rule algorithms** are common in this category. They aim to find correlations within large data sets, such as identifying correlations between products purchased together in supermarkets.


### Semi-supervised Learning
This approach uses data sets where some data points are labeled while the majority are not. It combines unsupervised and supervised algorithms.
A common example is the organization of photos stored on smartphones, such as in Google Photos. If it is known that a person appearing in a few photos is a family member of the user, the system will then classify the same person in other photos as a family member. In terms of algorithms, it's as if clustering is used to assign labels to data, which then allows for supervised learning.


### Self-supervised Learning
The self-supervised approach involves generating a fully labeled dataset from an unlabeled one.

An example is having a set of unlabeled images. In this set, a small portion is randomly masked to create and train a model to recover the original image, with the masked images serving as the model's inputs. This technique is useful for restoring damaged images or removing unwanted objects from them, but it is usually treated as an intermediate step for a subsequent classification task, such as identifying the animal present in the image, thereby distinguishing between species.


### Reinforcement Learning
Reinforcement learning systems operate by analyzing a situation, selecting and executing actions that result in either penalties or rewards, ultimately delivering the best strategy (also called a *policy*) for the desired outcome.

A common example is observing the algorithm through thousands of Go games to develop the best strategy and create a robot capable of playing the game.


## Batch Versus Online Learning

This distinction classifies systems based on whether they can learn incrementally from incoming data streams.

### Batch Learning
Batch learning involves a system that is incapable of learning incrementally and needs to utilize all the available data, typically performed offline (also known as *offline learning*).

This type of model tends to experience a decline in performance over time because it does not evolve along with the data or the environment being analyzed, a phenomenon known as *model rot* or *data drift* (the impact varies depending on the analysis; for example, the damage is less severe for image classification than for market analysis). If new data becomes available, the model must be retrained from scratch. Although this task can be automated and executed with relative ease, it is computationally expensive and is usually performed once a week or a few times during that week.

In general, it is preferable to operate systems that can use algorithms capable of learning incrementally.

### Online Learning
Online learning trains the system by continuously feeding it data sequentially, either individually or in small groups called *mini-batches*. The steps are quick and cost-effective.

It is useful because it adapts to extremely rapid changes, such as those in stock market systems, and allows training even on mobile devices. It can also be used for large datasets that do not fit into main memory (referred to as *out-of-core* learning).

An important parameter for this is the speed at which it adapts to changes, known as the *learning rate*, which can be adjusted. Slowing it down, for example, makes the system less sensitive to new noise.

The data may be affected by poor new data, and often requires constant monitoring to prevent these issues.


## Instance-Based Versus Model-Based Learning

Another way to categorize learning is by how it *generalizes.* Having a good performance measure on the training data is good, but not sufficient; the real goal is to perform well on new instances.

### Instance-Based Learning
In instance-based learning, the system learns from the examples, memorizes them, and then generalizes to new ones by creating a *similarity measure.* The most basic form of learning is essentially memorization. For example, in creating spam filters in this mode, the filter will flag emails that are very similar to known spam by analyzing common words between them.

### Model-Based Learning and a Typical Machine Learning Workflow
Another way to generalize a set of examples is to build a *model* from these examples and then make *predictions.* A common example of this is the *linear regression* model, based on available data, such as correlating data from the *Better Life Index* with *Life Satisfaction* and *GDP,* as seen in the example [add example to the repository]. From this, it is possible to estimate the *Life Satisfaction* of a country whose data is not included based on its *GDP.*



# Main Challenges of Machine Learning
When choosing a model and training it, there are two major problems: "Bad Data" and "Bad Model".

## Insufficient Quantity of Training Data
For machine learning, a considerable amount of data is needed to work. A basic example would be the need for hundreds or thousands of images of apples for the model to recognize.


## Nonrepresentative Training Data
This occurs when, even though there is a relevant volume of data, it is not satisfactorily representative. This type of error sometimes results in biased results. Using the previous example of GPD and Life Satisfaction, suppose there is no data from countries that are not in the range between $20,000 and $60,000. This will cause the understanding that there is linear behavior and will not consider the *noise*, when in fact, it is observable that sometimes, moderately rich countries have higher Life Satisfaction than richer countries. A brief summary to compare the two steps is that, for a small amount of data, what is called *sampling noise* occurs, and for non-representative data there is *sampling bias*.


## Poor-Quality Data
This occurs when the collected data has a lot of outliers and noise (usually due to *poor-quality measures*), which will cause performance failures in the model, and will make the person responsible for the analysis spend more time cleaning and treating the data than analyzing it itself.

In some cases, when there are some outliers, it is simpler to just discard them. In other cases, where only a few instances are lost, such as 5% of the specific age of consumers of a certain product is missing, several strategies can be adopted, such as ignoring this data, covering these values ​​with averages or training and adapting a model.


## Irrelevant Features
For the model to be effective, it must have a minimally significant amount of useful data, and the process is called *Feature Engineering*. The process consists of three parts: Feature Selection, Feature Extraction, and Creation of new features by gathering new data.


## Overfitting the Training Data
Now moving on to problems that involve modeling and not data, the first one analyzed will be *Overfitting*, which in simple terms, is the complete *overgeneralization* of the model. In practice, this means that the model works exceptionally well with the trained data, but when using new data, the model performs poorly.

With this type of error, when there is new noise, the model will detect a pattern in it, if it is used again for training, or it will fail if it is used for analysis. In other words, the model is unable to generalize new data.

The problem of Overfitting occurs when the model is too complex for the amount of data and noise available. This problem is usually remedied by simplifying the model, obtaining more data, or reducing the amount of noise. This process is called *regularization*. The amount of regularization applied to the model is controlled by something called the algorithm's *hyperparameter*. The hyperparameter does not affect the algorithm itself, but is set before training and kept constant during training.


## Underfitting the Training Data
This is basically the opposite of overfitting. It occurs when the model is too simple for the proposed data and analysis. To avoid this problem, some options are: Selecting more powerful models with more parameters, having better features for the algorithm (feature engineering), or reducing the model's constraints (hyperparameter changes).

To briefly summarize the two problems with the model, the model cannot be too complex (overfitting) nor too simple (underfitting).


# Testing and Validating
The usual way to test a model is to divide the data into a *training set* and a *test set*. The error rate in new cases is called *generalization error* (or *out-of-sample error*), and this value indicates how well the model will perform in new instances.
If the training error is low, but the generalization error is high, this indicates overfitting in the data. It is worth noting that the most common division is 80% for the training set and 20% for the test set, but the number can vary depending on how large the total data set is. If the set is very large, it is not necessary to follow this pattern, and a larger training set can be used.

## Hyperparameter Tuning and Model Selection
In model selection, such as comparing linear and polynomial regression, the way to decide which is the best is by comparing how they generalize the tests.

Once the best model is obtained, regularization should be performed to avoid overfitting. So, to define the regularizations of the hyperparameters, 100 different models are tested using 100 different hyperparameters, and then the best model would be found, in theory. Even though this model has a small *generalization error*, suppose 5%, when tested on a production model, the generalization error value could be higher, like 15%. This occurs because everything was tested with the same testing sets, making the model only good for the proposed test set.

A common solution is called *holdout validation*, which basically consists of saving part of the training data (training set) for a validation set (*validation set*, *development set* or *dev set*). The steps for this are basically the Initial Division, where the training and test data are divided; the Training Set Division, where the training and validation data are divided; the Training and Validation of the data; the Final Training of the data; and finally, the Final Evaluation.

Using an example where you have a set of 10,000 data. In the Initial Division, 80% would be used for the training set (8,000 data), and 20% for the test set (2,000). In the Training Set Division stage, the training set and the dev set can be divided in the same way in the proportion 80-20, with 6,400 data for the training set and 1,600 data for the dev set. In Training and Validation, different hyperparameters are trained for a model using the data from the training set, and are tested on the dev set, and the hyperparameter that has the best performance is selected. Then, in the Final Training, the model is trained completely using all 8,000 data (6,400 from the training set + 2,000 from the dev set), providing an estimate for generalization of the new data. Finally, the Final Evaluation is performed using the data from the test set, to then estimate how the model generalized and the generalization errors of the new data were. One problem to avoid during this process is the amount of data in the training set and the dev set. If the data set in the dev set is too small, it is possible to select a suboptimal model, and if it is too large, the training set will be too small. One way to avoid this is *cross validation*, which divides several dev sets. Each model is evaluated once for each dev set, and averaging them gives a more accurate measure of performance, but the training time is multiplied by the number of validation sets.


## Data Mismatch
The data mismatch problem occurs when there is a significant difference between the characteristics of the data used to train a model and the data that the model encounters during validation, testing, or in the real world after training. To explain this error, and how to identify it, we will use an example of an application to identify flowers through photos, which has 1000 representative photos taken with the application, and another part collected from the web.

After dividing these sets of photos from the web and the application into training set, dev set, and test set, we are faced with the problem of the result being below the desired one. In this case, we will not know if the problem in question consists of overfitting or if there was a mismatch between the images, a *Data Mismatch*.

A solution proposed by Andrew Ng is to keep a train-dev set containing more imprecise images, in this case, from the web, now having a total of 4 sets: train set, train-dev set, dev set, and test set. With this, the training set will be used for training and evaluated as a trian-dev set. In this case, if the performance is urim, then it will be an overfitting problem. If the model performs well, the dev set will be evaluated. If in this case the performance is poor, the problem in question will be *Data Mismatch*, and for this problem the images can be preprocessed so that they look more like the photos taken by the application, and then the model can be retrained. Once the model performs well on the train-dev set and on the dev set, it will be possible to evaluate it on the test set.
