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



