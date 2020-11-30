################################################################################
# Introduction
# 
# Author: Nitesh Kumar Gupta
# Date: 25/11/2020
#
# This is a source code (R Script) for, Edx HarvardX Data Science Professonal 
# certificate, second capstone project. 
# In this project, we will implement customer segmentation. Customer segmentation 
# is one the most important applications of unsupervised learning. We will make 
# use of K-means clustering which is the essential algorithm for clustering 
# unlabeled dataset.
################################################################################

################################################################################
# Install required packages if not installed already
# Note: this process could take a couple of minutes
################################################################################

if(!require(readr)) install.packages("readr")
if(!require(purrr)) install.packages("purrr")
if(!require(cluster)) install.packages("cluster")
if(!require(gridExtra)) install.packages("gridExtra")
if(!require(grid)) install.packages("grid")
if(!require(NbClust)) install.packages("NbClust")
if(!require(factoextra)) install.packages("factoextra")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(dplyr)) install.packages("dplyr")

################################################################################
# Load required libraries
################################################################################

library (readr)
library(purrr)
library(cluster) 
library(gridExtra)
library(grid)
library(NbClust)
library(factoextra)
library(tidyverse)
library(dplyr)

################################################################################
# For better data accessibility by script, I have uploaded customer data into  
# my github repository which is publicly available. We are going to use github 
# raw url to fetch and load data instead of reading data from local path.
################################################################################

# Customer data file url
customer_file_url="https://raw.githubusercontent.com/niteshn99/Edx-HarvardX-CustomerSegment/main/data/Mall_Customers.csv"
# read csv file into dataframe 
customer_data <- read_csv(url(customer_file_url))


################################################################################
# EDA - Exploratory data analysis
################################################################################

# Basic understaning of data
# Lets have a look at first six rows of customer data
head(customer_data)

# Lets have a look at structure of customer data
str(customer_data)

# Columns/Features names
names(customer_data)

# Customer Gender Distribution
a=table(customer_data$Gender)
barplot(a,main="Using BarPlot to display Gender Comparision",
        ylab="Count",
        xlab="Gender",
        col=rainbow(2),
        legend=rownames(a))
# Above bar plot shows gender distribution and it is clear from graph is that 
# number of female shopper is higher than male shopper. Lets have a look at that 
# in ratio proportion. 
# Ratio of male and female distribution

pct=round(a/sum(a)*100)
lbs=paste(c("Female","Male")," ",pct,"%",sep=" ")
library(plotrix)
pie3D(a,labels=lbs,
      main="Pie Chart Depicting Ratio of Female and Male")

# From the above graph, we conclude that percentage of females is 56% whereas 
# percentage of male in customer dataset is 44%

# Customer Age Distribution

sd(customer_data$Age)
summary(customer_data$Age)

hist(customer_data$Age,
     col="blue",
     main="Histogram to Show Count of Age Class",
     xlab="Age Class",
     ylab="Frequency",
     labels=TRUE)

boxplot(customer_data$Age,
        col="#ff0066",
        main="Boxplot for Descriptive Analysis of Age")

# From above two graphs, we conclude that maximum number of customers are in age 
# group between 30 and 50 years. Minimum age of customer is 18 whereas maximum 
# age is 70.

# Customer Annual Income Distribution

sd(customer_data$`Annual Income (k$)`)
summary(customer_data$Annual.Income..k..)

hist(customer_data$`Annual Income (k$)`,
     col="#660033",
     main="Histogram for Annual Income",
     xlab="Annual Income Class",
     ylab="Frequency",
     labels=TRUE)

## Annual income density plot
plot(density(customer_data$`Annual Income (k$)`),
     col="yellow",
     main="Density Plot for Annual Income",
     xlab="Annual Income Class",
     ylab="Density")
polygon(density(customer_data$`Annual Income (k$)`),
        col="#ccff66")

# From the above descriptive analysis, we conclude that the minimum annual income of the customer is 15
# and maximum income is approximately 140. Average income having highest frequency count based histogram 
# distribution is 70; however, average average income of all customer is approximately equal to 60 or 61.
# Finally, based on Kernel distribution plot we can conclude that annual income has a normal distribution.


# Customer Spending Score 

sd(customer_data$`Spending Score (1-100)`)
summary(customer_data$`Spending Score (1-100)`)

boxplot(customer_data$`Spending Score (1-100)`,
        horizontal=TRUE,
        col="#990000",
        main="BoxPlot for Descriptive Analysis of Spending Score")

hist(customer_data$`Spending Score (1-100)`,
     main="HistoGram for Spending Score",
     xlab="Spending Score Class",
     ylab="Frequency",
     col="#6600cc",
     labels=TRUE)

# Descriptive analysis on spending score shows that minimum score is 1, maximum is 99, and average score is
# 50.20. From the histogram plot, we can conclude that customers between class 40 and 50 have highest spending
# score among all the classes.


################################################################################
# Customer Segmentation using K-means Algorithm
################################################################################

# Determining optimal cluster

## Elbow Method

set.seed(123, sample.kind = "Rounding")
# Function to calculate total intra-cluster sum of square 
iss <- function(k) {
  kmeans(customer_data[,3:5],k,iter.max=100,nstart=100,algorithm="Lloyd" )$tot.withinss
}
# Range of K values out of which we want to determine best optimal k value
k.values <- 1:10
#Intra-cluster sum of square for each K values
iss_values <- map_dbl(k.values, iss)
# Elbow curve plot to determine best optimal K value
plot(k.values, iss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total intra-clusters sum of squares")

# From above plot, we can conclude that K = 4 is the appropriate number of clusters since it seems to be 
# appearing at the bend in the elbow curve plot.


## Average Silhouette Method

# Using the silhouette function in the cluster package, we can compute 
# the average silhouette width using the kmean function. With the optimal number of k clusters, 
# we can maximize the average silhouette over significant values for k clusters.

# K = 2 clusters
k2<-kmeans(customer_data[,3:5],2,iter.max=100,nstart=50,algorithm="Lloyd")
s2<-plot(silhouette(k2$cluster,dist(customer_data[,3:5],"euclidean")))
# K = 3 clusters
k3<-kmeans(customer_data[,3:5],3,iter.max=100,nstart=50,algorithm="Lloyd")
s3<-plot(silhouette(k3$cluster,dist(customer_data[,3:5],"euclidean")))
# K = 4 clusters
k4<-kmeans(customer_data[,3:5],4,iter.max=100,nstart=50,algorithm="Lloyd")
s4<-plot(silhouette(k4$cluster,dist(customer_data[,3:5],"euclidean")))
# K = 5 clusters
k5<-kmeans(customer_data[,3:5],5,iter.max=100,nstart=50,algorithm="Lloyd")
s5<-plot(silhouette(k5$cluster,dist(customer_data[,3:5],"euclidean")))
# K = 6 clusters
k6<-kmeans(customer_data[,3:5],6,iter.max=100,nstart=50,algorithm="Lloyd")
s6<-plot(silhouette(k6$cluster,dist(customer_data[,3:5],"euclidean")))
# K = 7 clusters
k7<-kmeans(customer_data[,3:5],7,iter.max=100,nstart=50,algorithm="Lloyd")
s7<-plot(silhouette(k7$cluster,dist(customer_data[,3:5],"euclidean")))
# K = 8 clusters
k8<-kmeans(customer_data[,3:5],8,iter.max=100,nstart=50,algorithm="Lloyd")
s8<-plot(silhouette(k8$cluster,dist(customer_data[,3:5],"euclidean")))
# K = 9 clusters
k9<-kmeans(customer_data[,3:5],9,iter.max=100,nstart=50,algorithm="Lloyd")
s9<-plot(silhouette(k9$cluster,dist(customer_data[,3:5],"euclidean")))
# K = 10 clusters
k10<-kmeans(customer_data[,3:5],10,iter.max=100,nstart=50,algorithm="Lloyd")
s10<-plot(silhouette(k10$cluster,dist(customer_data[,3:5],"euclidean")))

# Maximum average silhouette can be seen for K = 6

# We can make use of the fviz_nbclust() function to determine and visualize the optimal number of clusters
fviz_nbclust(customer_data[,3:5], kmeans, method = "silhouette")

# From above plot, we can conclude that K = 7 is the appropriate number of clusters for our customer data. 
# However, maximum average silhouette value can be seen for K = 6 when we computed individually.

## Gap statistic method

# For computing the gap statistics method we can utilize the clusGap function for providing 
# gap statistic as well as standard error for a given output.

set.seed(125, sample.kind = "Rounding")
stat_gap <- clusGap(customer_data[,3:5], FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)
fviz_gap_stat(stat_gap)

# From above plot, we can conclude that K = 6 is optimal number of cluster for our customer data. 
# However, in our previous analysis we concluded that 4 is optimal based on elbow curve and 7 based on
# Silhouette method. Looking at all together, 6 and 7 have very minor difference and 4 is far away from optimal.

## Lets us take k = 6 as our optimal cluster

k6<-kmeans(customer_data[,3:5],6,iter.max=100,nstart=50,algorithm="Lloyd")
k6

# From above output of our kmeans operation, we observe a list with several key information. 
# From this, we conclude the useful information being –

# cluster – This is a vector of several integers that denote the cluster which has an allocation of each point.
# totss – This represents the total sum of squares.
# centers – Matrix comprising of several cluster centers
# withinss – This is a vector representing the intra-cluster sum of squares having one component per cluster.
# tot.withinss – This denotes the total intra-cluster sum of squares.
# betweenss – This is the sum of between-cluster squares.
# size – The total number of points that each cluster holds.


## Lets inspect first Two Principle Components

pcclust=prcomp(customer_data[,3:5],scale=FALSE) #principal component analysis
summary(pcclust)
pcclust$rotation[,1:2]

# Visualizing the Clustering Results using the Annual Income, Spending Score, Age, and First Two Principle Components

set.seed(1, sample.kind = "Rounding")

# Visualizing the clustering using Annual income and Spending Score
ggplot(customer_data, aes(x = `Annual Income (k$)`, y = `Spending Score (1-100)`)) + 
  geom_point(stat = "identity", aes(color = as.factor(k6$cluster))) +
  scale_color_discrete(name=" ",
                       breaks=c("1", "2", "3", "4", "5","6"),
                       labels=c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5","Cluster 6")) +
  ggtitle("Segments of Mall Customers", subtitle = "Using K-means Clustering")

# From above scatter plot, we can conclude that
# Cluster 1 and 2 - These two cluster consist of customers with medium annual income and medium annual spend of income.
# Cluster 3 - These are the customers having low annual income but high annual spend of income.
# Cluster 4 - These are the customers having low annual income and low annual spend of income.
# Cluster 5 - These are the customers having high annual income and high annual spend of income.
# Cluster 6 - These are the customers having high annual income but low annual spend of income. 


# Visualizing the cluster using Spending Score and Age
ggplot(customer_data, aes(x = `Spending Score (1-100)`, y = Age)) + 
  geom_point(stat = "identity", aes(color = as.factor(k6$cluster))) +
  scale_color_discrete(name=" ",
                       breaks=c("1", "2", "3", "4", "5","6"),
                       labels=c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5","Cluster 6")) +
  ggtitle("Segments of Mall Customers", subtitle = "Using K-means Clustering")

# From above scatter plot, we can conclude that
# Cluster 1 - These are the customers with lower age group having medium annual spend of income.
# Cluster 2 - These are the customers with higher age group having medium annual spend of income.
# Cluster 3 - These are the customers with lower age group having high annual spend of income.
# Cluster 4 - These are the customers with higher age group having low annual spend of income.
# Cluster 5 - These are the customers with medium age group but high annual spend of income.
# Cluster 6 - These are the customers with medium age group but low annual spend of income.

# Visualizing customers into 6 cluster based on first two principal component
kCols = function(vec) {
  cols=rainbow (length (unique (vec)))
  return (cols[as.numeric(as.factor(vec))])
  }

digCluster<-k6$cluster 
dignm<-as.character(digCluster) # K-means clusters
plot(pcclust$x[,1:2], col = kCols(digCluster), pch = 19, xlab = "K-means",ylab = "classes")
legend("bottomright",unique(dignm),fill=unique(kCols(digCluster)))

# Based on above cluster plot, we can conclude that
# Cluster 1 and 2 – These two clusters consist of customers with medium PCA1 and medium PCA2 score.
# Cluster 3 – This comprises of customers with a high PCA2 and a medium PCA1.
# Cluster 4 – This cluster comprises of customers with a high PCA1 score and a high PCA2.
# Cluster 5 – This cluster represents customers having a high PCA2 and a low PCA1.
# Cluster 6 – In this cluster, there are customers with a medium PCA1 and a low PCA2 score.

# Results
# Below is complete list of customer data with cluster they belong too.
customer_data_with_cluster <- as.data.frame(k6$cluster) %>% 
  mutate(CustomerID = row_number()) %>%
  left_join(customer_data, by = 'CustomerID')

customer_data_with_cluster
