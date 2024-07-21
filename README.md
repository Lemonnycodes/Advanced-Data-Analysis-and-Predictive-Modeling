
# Advanced Data Mining and Predictive Modeling

This repository contains a collection focused on advanced data mining and predictive modeling techniques. Each one is about an algorithm and approaches to tackle complex problems in classification, clustering, and predictive analytics. This demonstrate skills in feature selection, dealing with imbalanced data, implementing custom algorithms, and optimizing performance metrics.

## Table of Contents
- [ Predictive Modeling for Drug Activity](#project-1-predictive-modeling-for-drug-activity)
- [Predictive Modeling for Drug Activity Using Dimensionality Reduction and Classifiers](#project-2-ensemble-classifier-with-boosting-for-drug-activity-prediction)
- [Predictive Modeling for Drug Activity Using Ensemble Models and Boosting ](#project-3-k-means-clustering-on-iris-dataset)
- [K-Means Clustering on High-Dimensional Data](#project-4-k-means-clustering-on-handwritten-digits)

##  Predictive Modeling for Drug Activity

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




## Predictive Modeling for Drug Activity Using Dimensionality Reduction and Classifiers

**Objective:**  
Evaluate the activity of a given drug compound efficiently, given a highly dimensional and imbalanced dataset, to facilitate the drug discovery process.

**Description:**  
This project focuses on predicting the activity of drug compounds using advanced dimensionality reduction and classification techniques. Given the highly sparse dataset with numerous features, the project emphasizes reducing dimensionality for better performance. Various classifiers were implemented to determine the most effective method for this prediction task.

**Key Techniques:**
- Feature selection and dimensionality reduction using PCA
- Handling imbalanced data
- Binary classification using Naive Bayes and Neural Network algorithms
- Performance evaluation using F1-score

**Technologies:**  
Python, scikit-learn, pandas, NumPy

**Challenges and Techniques:**
- **Handling Imbalanced Data:**  
  Managing an imbalanced dataset with a large number of inactive cases was a significant challenge. Specialized techniques were employed to address this issue.
  
- **Dimensionality Reduction:**  
  The dataset's high sparsity and large number of features necessitated immediate dimensionality reduction. PCA (Principal Component Analysis) was used over truncated SVD for its superior results and faster processing time. The dimensions of the testing and training data matrices were reduced from 100,000 to 100 using PCA, capturing 70% of the variance with 500 principal components.

- **Model Selection:**  
  For evaluating the activity of a given drug, both Neural Network and Naive Bayes classifiers were considered. Naive Bayes was preferred due to its scalability, optimal results, speed, insensitivity to insignificant details, and resistance to overfitting. The dataset's binary format also made it particularly suitable for Naive Bayes.

**Approach:**
- A sparse matrix with 0 and 1 values was created to determine the activity of the compounds as soon as the data was loaded.
- The data was initially projected with lesser dimensionality using PCA.
- The dimensionality of the dataset was reduced from 100,000 to 100 using PCA, as truncated SVD did not yield satisfactory results.
- A classifier algorithm was then used to predict the activity of the compound.

**Results:**
- **F1 Scores:**
  - Naive Bayes: 0.716
  - Neural Network: 0.53


## Predictive Modeling for Drug Activity Using Ensemble Models and Boosting

**Objective:**  
Enhance the accuracy of predicting the activity of drug compounds using ensemble models and boosting techniques, thereby improving the identification of potential drug candidates.

**Description:**  
This project focuses on employing ensemble models and boosting techniques to predict the activity of drug compounds. Given the challenges of handling highly dimensional and imbalanced datasets, the project utilizes various methodologies to optimize prediction accuracy.

**Key Techniques:**
- Dimensionality reduction using TruncatedSVD
- Handling imbalanced data using SMOTE oversampling
- Implementing ensemble models with different classifiers
- Boosting with AdaBoost to improve model accuracy
- Performance evaluation using precision and F1-score

**Technologies:**  
Python, scikit-learn, pandas, NumPy, AdaBoost

**Methodology:**
- A sparse matrix with 0 and 1 values was created to determine the activity of the compounds as soon as the data was loaded.
- The dimensionality of the dataset was reduced using TruncatedSVD.
- An ensemble model with three different classifiers (Decision Tree, SVM, KNN, Logistic Regression) was deployed to predict the activity of the compounds.
- SMOTE oversampling was used to balance the data by creating duplicates in the lesser class.
- The Decision Tree classifier provided the highest accuracy among the classifiers.
- Cross-validation was performed to evaluate metrics and determine the F1-score.

**Boosting:**
- AdaBoost was employed to increase the accuracy of the Decision Tree classifier.
- The weight was initially set to weight(xi) = 1/n.
- The class label for the weaker class was predicted and weighted by alpha z.
- The model was then fit, and the data was split for prediction.

**Results:**
- **Metrics Derived Using Various Classifiers:**
  - **KNN:** Precision - 0.67
  - **Decision Tree:** Precision - 0.83
  - **Logistic Regression:** Precision - 0.76
  - Initially, Naive Bayes was used to determine the activity of the compounds and provided decent results. However, using the Decision Tree with boosting resulted in a significant improvement in accuracy.


##  K-Means Clustering on High-Dimensional Data

**Objective:**  
Utilize dimensionality reduction and K-Means clustering to efficiently group high-dimensional data, demonstrating the effectiveness of clustering techniques in handling complex datasets.

**Description:**  
This project focuses on applying dimensionality reduction techniques followed by the K-Means clustering algorithm to group high-dimensional data. The goal is to identify optimal clusters in the dataset, which can reveal underlying patterns and insights.

**Key Techniques:**
- Pre-processing and normalization of datasets
- Dimensionality reduction using PCA, TSNE, and Truncated SVD
- Clustering using K-Means algorithm
- Performance evaluation using the Elbow method

**Technologies:**  
Python, scikit-learn, pandas, NumPy, scipy

**Methodology:**
- **Pre-processing and Normalization:**  
  The datasets were pre-processed and normalized to ensure consistency and improve the performance of clustering algorithms.
  
- **Dimensionality Reduction:**  
  Principal Component Analysis (PCA), TSNE, and Truncated SVD were used to reduce the dimensionality of the dataset. TSNE was found to be the most effective due to its efficiency in handling larger dimensionality, yielding the best results.

- **K-Means Clustering:**
  - **Step 1:** Initialize K clusters.
  - **Step 2:** Assign random centroids.
  - **Step 3:** Assign each point to the closest centroid.
  - **Step 4:** Compute the average of centroids.
  - **Step 5:** Plot graphs to depict optimal solutions.

- For K-Means implementation, random points were taken as the initial centroid points. Distance measures using the scipy library were computed, and centroids were updated with each iteration until the highest accuracy was reached.
- The output clusters from K-Means were stored as a .txt file.
- The Elbow method was employed to identify the optimal number of clusters.

**Results:**
- **Clustering Performance:**  
  K-Means was computed after reducing the dimensions of the dataset. The clustered data points revealed meaningful groups, and the Elbow method helped determine the optimal number of clusters.

**Conclusion:**
- **Metrics :**
- Accuracy for Iris data: 92%
- Accuracy for Image Clustering 85.4%
  
The project demonstrated the effectiveness of combining dimensionality reduction with K-Means clustering. The reduced dimensions facilitated more efficient clustering, and the use of TSNE provided superior results compared to other methods. Important graphs and visualizations illustrating the clustering results and optimal solutions are included below.

**Graphs and Visualizations:**

![Elbow Method](https://github.com/Lemonnycodes/Advanced-Data-Mining-and-Predictive-Modeling/blob/main/elbow_method_graph.png)
![TSNE Clustering](https://github.com/Lemonnycodes/Advanced-Data-Mining-and-Predictive-Modeling/blob/main/4kmeans.png)
