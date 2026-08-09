# Customer Segmentation using Unsupervised Learning

## Project Overview

This project focuses on customer segmentation using unsupervised machine learning techniques. The goal is to identify meaningful groups of customers based on their demographic information, purchasing behavior, campaign responses, and other customer attributes.

The project uses the `marketing_campaign.csv` dataset and applies data preprocessing, feature scaling, Principal Component Analysis (PCA), and K-Means clustering to identify distinct customer segments.

## Dataset

The dataset contains 2,240 customer records with 29 columns before preprocessing.

Key features include:

* Customer income
* Year of birth
* Education
* Marital status
* Number of children and teenagers
* Recency
* Product spending across different categories
* Web, catalog, and store purchases
* Website visits
* Marketing campaign responses
* Customer complaints

The dataset is loaded from:

```text
marketing_campaign.csv
```

## Project Workflow

### 1. Data Preparation and Scaling

The initial dataset was inspected to understand its structure, data types, and missing values.

The following preprocessing steps were performed:

* Loaded the dataset using Pandas.
* Identified 24 missing values in the `Income` column.
* Filled missing income values using the median.
* Detected income outliers using the IQR method.
* Removed income-based outliers.
* Dropped irrelevant columns:

  * `ID`
  * `Z_CostContact`
  * `Z_Revenue`
* Selected 23 numerical features.
* Standardized the numerical features using `StandardScaler`.

After preprocessing and outlier removal, the dataset contained 2,232 records and 23 numerical features.

## 2. Dimensionality Reduction with PCA

Principal Component Analysis (PCA) was applied to reduce the dimensionality of the dataset while preserving important information.

A full PCA analysis was first performed to examine the variance distribution.

The first three principal components explained:

* PC1: 29.76%
* PC2: 8.89%
* PC3: 8.24%

Together, the three components captured approximately **46.89% of the total variance**.

The final PCA representation contained:

```text
2232 samples × 3 principal components
```

These three components were then used as the input for clustering.

## 3. K-Means Clustering

K-Means clustering was used to identify groups of customers with similar characteristics.

### Elbow Method

The Within-Cluster Sum of Squares (WCSS) was calculated for values of K from 1 to 10.

The Elbow Method was used to analyze how clustering performance changed as the number of clusters increased.

### Silhouette Score

Silhouette scores were also calculated for K values from 2 to 10 to evaluate cluster separation and cohesion.

The scores obtained were:

```text
K=2: 0.4732
K=3: 0.4280
K=4: 0.4428
K=5: 0.3521
K=6: 0.3337
K=7: 0.3335
K=8: 0.3283
K=9: 0.3344
K=10: 0.3229
```

Based on the clustering analysis, the final K-Means model was implemented with **4 clusters**.

```python
KMeans(n_clusters=4, random_state=42)
```

The resulting clusters were visualized using a 3D plot based on the three PCA components.

## 4. Customer Personas

After clustering, the characteristics of each cluster were analyzed using group-level statistics.

### Cluster 3: The Premium Loyalists

Key characteristics:

* Highest average income, approximately 81k
* Highest overall spending
* Strong campaign response
* Very low number of children and teenagers

Business actions:

* Exclusive VIP treatment
* Early access to products
* Loyalty programs

### Cluster 2: The Affluent Explorers

Key characteristics:

* High average income, approximately 73k
* High spending across product categories
* Few or no children
* Strong purchasing behavior

Business actions:

* Cross-selling premium products
* Upselling through catalog and store channels

### Cluster 0: The Steady Moderates

Key characteristics:

* Moderate income, approximately 57k
* Balanced spending across product categories
* Moderate household characteristics
* Consistent purchasing behavior

Business actions:

* Value-based offers
* Bundle deals
* Strategies to encourage movement toward higher-value segments

### Cluster 1: The Budget-Conscious Families

Key characteristics:

* Lowest average income, approximately 34k
* Highest number of children
* Lowest overall spending
* Low campaign response

Business actions:

* Discount-driven marketing
* Family-value bundles
* Budget-focused offers

## Technologies and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Machine Learning Techniques

* Data Cleaning
* Missing Value Imputation
* Outlier Detection using IQR
* Feature Scaling
* Principal Component Analysis (PCA)
* K-Means Clustering
* Elbow Method
* Silhouette Score
* Customer Persona Analysis

## Project Structure

```text
Task_3/
│
├── Task_3.ipynb
├── marketing_campaign.csv
└── README.md
```

## Conclusion

This project demonstrates how unsupervised learning can be used to transform raw customer data into meaningful customer segments.

By combining data preprocessing, feature scaling, PCA, and K-Means clustering, four distinct customer personas were identified. These segments provide a foundation for targeted marketing strategies, personalized offers, customer retention initiatives, and improved business decision-making.
