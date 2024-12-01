# Unsupervised Learning Techniques

Although most of the efforts made for Machine Learning are in supervised learning, most of the data that exists does not have captions, which highlights the importance of **unsupervised learning**. The following topics will be covered in the following chapter:

- **Clustering**: Groups similar instances into _Clusters_. It is useful for data analysis, customer segmentation, recommendation systems, search systems, image sizing, etc.

- **Anomaly Detection (or Outlier Detection)**: Its objective is to identify what is "normal" for the system and detect abnormal instances. Instances considered normal are called **inliers** and those that are abnormal are called **anomalies** or **outliers**. Widely used in fraud detection, defective products, time series analysis, etc.

- **Density Estimation**: Estimates the Probability Density Function (PDF) of what generated the data set. Used for anomaly detection such as instances in very low density regions, and also useful for data visualization.
  
<!---------------------------------------------------->
<!---------------------------------------------------->
<!---------------------------------------------------->


# Clustering Algorithms: k-means and DBSCAN 

**Clustering** consists of grouping instances within a group with other similar instances, called a **Cluster**. It is an unsupervised process, different from classification. Some of the applications of **Clustering** are:

- **Customer Segmentation**: Useful for _Recommender Systems_, when identifying patterns of purchases and activities on websites.

- **Data Analysis**: Helps in visualization;

- **Dimensionality Reduction**: After clustering, it is possible to see the _Affinity_ of the instances, which measures how well an instance is in a cluster.

- **Feature Engineering**: As shown in chapter 2, the geographic cluster of the California Housing dataset.

- **Anomaly Detection**: Detects based on the _Low Affinity_ of the instances in the clusters.

- **Semi-Supervised Learning**: If there are some instances, it is possible to cluster and propagate the captions to other instances in the same cluster.

- **Search Engines**: An example is image searches, finding one that is close to the cluster of the image used to perform the search.

- **Image Segmentation**: Clusters pixels according to colors, replacing and reducing the number of different colors, used for object detection and tracking systems, in addition to detecting contours.

There is no single way to define a cluster, it depends on the context and the algorithm. Some search for the center around a defined point called _Centroid_, others search for denser regions. Two algorithms for this called **k-means** and **DBSCAN** will be discussed shortly.
<!---------------------------------------------------->
<!---------------------------------------------------->
<!---------------------------------------------------->


## k-means 



<!---------------------------------------------------->
              
## Limits of k-means


<!---------------------------------------------------->
                           
## Using Clustering for Image Segmentation 



<!---------------------------------------------------->

## Using Clustering for Semi-Supervised Learning  


<!---------------------------------------------------->

## DBSCAN 


<!---------------------------------------------------->
                                        
## Other Clustering Algorithms   



<!---------------------------------------------------->
<!---------------------------------------------------->
<!---------------------------------------------------->


# Gaussian Mixtures


<!---------------------------------------------------->
                              
## Using Gaussian Mixtures for Anomaly Detection


<!---------------------------------------------------->

## Selecting the Number of Clusters


<!---------------------------------------------------->
 
## Bayesian Gaussian Mixture Models 


<!---------------------------------------------------->

## Other Algorithms for Anomaly and Novelty Detection 



<!---------------------------------------------------->
