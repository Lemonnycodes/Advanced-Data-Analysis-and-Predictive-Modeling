
# Advanced Data Mining and Predictive Modeling

This repository contains a collection focused on advanced data mining and predictive modeling techniques. Each project leverages different algorithms and approaches to tackle complex problems in classification, clustering, and predictive analytics. The projects demonstrate skills in feature selection, dealing with imbalanced data, implementing custom algorithms, and optimizing performance metrics.

## Table of Contents
- [Project 1: Predictive Modeling for Drug Activity](#project-1-predictive-modeling-for-drug-activity)
- [Project 2: Ensemble Classifier with Boosting for Drug Activity Prediction](#project-2-ensemble-classifier-with-boosting-for-drug-activity-prediction)
- [Project 3: K-Means Clustering on Iris Dataset](#project-3-k-means-clustering-on-iris-dataset)
- [Project 4: K-Means Clustering on Handwritten Digits](#project-4-k-means-clustering-on-handwritten-digits)

## Project 1: Predictive Modeling for Drug Activity

**Objective:**  
Develop predictive models to determine whether a given compound is active (1) or not (0) based on its molecular features, which is crucial in the initial stages of drug discovery for identifying promising compounds.

**Description:**  
Drugs are small organic molecules that achieve their desired activity by binding to a target site on a receptor. The first step in drug discovery is identifying compounds that can bind to a specific receptor. This project involves developing a binary classification model to distinguish active compounds from inactive ones. Given an imbalanced dataset with thousands of binary features representing the topological shapes and characteristics of molecules, the goal is to optimize the model using the F1-score.

**Key Techniques:**
- Feature selection and engineering
- Handling imbalanced data
- Binary classification using various algorithms
- Performance evaluation using F1-score
- Cosine similarity determination using the spatial matrix module
- Data splitting into 70% training and 30% testing sets
- Evaluation of top k elements from sorted similarity tuples with accuracies ranging from 50% to 80%

**Technologies:**  
Python, scikit-learn, pandas, NumPy

**Results:**  
The model achieved an accuracy of 77.6% for 18,000 datasets. Cosine similarity was determined using the spatial matrix module, and the train and test data were split into a 70:30 ratio. The top k elements (k = 3, 20, 50, 72) were selected and returned as tuples, with accuracies ranging from 50% to 80%. Positive and negative reviews/labels were considered, and the majority value from the most similar tuple was assigned to the test vector. The estimated time for determining the accuracy was 1 second per review. The final formatted document with accurate analysis is stored as "format file.csv". A confusion matrix was also determined to evaluate the performance of the KNN classification.

![KNN Confusion Matrix](https://github.com/Lemonnycodes/Advanced-Data-Mining-and-Predictive-Modeling/blob/main/1knn.jpeg)


## Project 2: Ensemble Classifier with Boosting for Drug Activity Prediction

**Objective:**  
Implement a custom ensemble classifier using Boosting to improve the prediction of drug activity, thereby enhancing the reliability of identifying potential drug candidates.

**Description:**  
Building on the previous assignment, this project focuses on implementing an ensemble classifier using Boosting techniques, such as AdaBoost. The dataset remains the same, with the task of predicting the activity of compounds. The challenge includes writing custom code for the Boosting algorithm while using libraries for data pre-processing and base classifiers. The F1-score will be used as the performance metric.

**Key Techniques:**
- Feature selection/reduction
- Handling imbalanced data
- Implementing custom Boosting algorithms
- Ensemble learning
- Performance evaluation using F1-score

**Technologies:**  
Python, scikit-learn (for pre-processing), custom Boosting implementation

## Project 3: K-Means Clustering on Iris Dataset

**Objective:**  
Test and evaluate a custom K-Means algorithm on the Iris dataset to understand the fundamental principles of clustering and benchmark algorithm performance on a classic dataset.

**Description:**  
The Iris dataset is a well-known benchmark in data mining and machine learning. This assignment involves implementing the K-Means clustering algorithm from scratch and testing it on the Iris dataset, which includes four features: sepal length, sepal width, petal length, and petal width, across 150 instances. The goal is to assign each instance to one of three clusters and evaluate the clustering performance using the V-measure.

**Key Techniques:**
- Implementing K-Means algorithm
- Clustering evaluation using V-measure
- Benchmarking with a known dataset

**Technologies:**  
Python, NumPy

## Project 4: K-Means Clustering on Handwritten Digits

**Objective:**  
Implement the K-Means algorithm to cluster 10,000 images of handwritten digits, exploring the application of clustering techniques in high-dimensional image data for pattern recognition.

**Description:**  
This assignment requires implementing the K-Means clustering algorithm from scratch to cluster a dataset of 10,000 images of handwritten digits (0-9). Each image is represented as a 28x28 pixel matrix, which is flattened into a 1x784 vector. The goal is to assign each instance to one of ten clusters and evaluate the clustering performance using the V-measure. The project highlights the application of K-Means to high-dimensional data and the challenges associated with clustering image data.

**Key Techniques:**
- Implementing K-Means algorithm
- Clustering high-dimensional data
- Evaluation using V-measure
- Handling large datasets

**Technologies:**  
Python, NumPy, scikit-learn (for evaluation)
