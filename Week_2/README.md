Data Science Project 2: Fraud Detection Pipeline
DecodeLabs Internship — Week 2 (Batch 2026)

Note: ## Dataset

The dataset used for this task exceeds GitHub's file size limit, so it is not included in this repository.

The analysis and complete implementation are available in Task_2.ipynb.



Overview

This project focuses on building a supervised learning pipeline to detect fraudulent credit card transactions in a highly imbalanced dataset. The goal was not just to train a classification model, but to build one that is safe from data leakage, evaluated with the right metrics, and tuned for a real-world business trade-off between catching fraud and protecting genuine customers from false alarms.

Dataset
Source: Credit Card Fraud Detection Dataset (Kaggle, mlg-ulb)
Size: 284,807 transactions
Features: Time, V1–V28 (PCA-transformed features), Amount
Target: Class (0 = legitimate, 1 = fraud)
Class distribution: 284,315 legitimate (99.83%) vs 492 fraud (0.17%)
Problem Statement

Build and tune a classification model to identify fraudulent transactions in a dataset where the minority class (fraud) makes up only 0.17% of all records. A model that predicts "legitimate" for every transaction would achieve 99.83% accuracy while catching zero fraud, which makes accuracy a misleading metric for this task.

Key Objectives
Handle extreme class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
Prevent data leakage by applying SMOTE and scaling only within the training folds
Train and compare Logistic Regression and Random Forest classifiers
Evaluate models using Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC instead of accuracy
Tune hyperparameters using GridSearchCV without leaking data during cross-validation
Workflow
Exploratory Data Analysis Checked dataset structure, confirmed no missing values, and visualized the extreme class imbalance.
Train/Test Split Performed a stratified 80/20 split before any resampling, to preserve the original class ratio in both sets and avoid data leakage.
Pipeline Construction Built two pipelines using imblearn.pipeline.Pipeline:
Logistic Regression: StandardScaler followed by SMOTE followed by the classifier
Random Forest: SMOTE followed by the classifier (no scaling needed, since tree-based models are scale-invariant)
Model Training and Evaluation Trained both pipelines on the training set and evaluated on the untouched test set using Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.
Hyperparameter Tuning Used GridSearchCV with the imblearn pipeline to tune SMOTE's k_neighbors alongside Random Forest's n_estimators and max_depth, ensuring SMOTE was applied fresh inside every cross-validation fold.
Final Evaluation Evaluated the best tuned model on the test set and compared results against the untuned baseline.
Results
Metric	Logistic Regression	Random Forest (baseline)	Random Forest (tuned)
Precision (fraud)	0.06	0.84	0.82
Recall (fraud)	0.92	0.83	0.85
F1-score (fraud)	0.11	0.83	0.83
False Positives	1,467	16	18
False Negatives	8	17	15
ROC-AUC	0.971	0.964	0.961
Key Findings

Logistic Regression achieved high recall but extremely poor precision, flagging 1,467 legitimate transactions as fraud. In a real banking system, this would mean a large number of genuine customers facing blocked transactions or false fraud alerts.

Random Forest achieved a much more balanced result, with far fewer false positives while still catching the majority of fraud cases. Hyperparameter tuning shifted the balance slightly toward higher recall at the cost of a small increase in false positives, but overall performance remained comparable to the untuned baseline.

Model Selection and Business Reasoning

Random Forest was selected as the final model. While it misses slightly more fraud cases than Logistic Regression, it produces dramatically fewer false alarms. In a production fraud detection system, repeatedly flagging genuine customers erodes trust and increases support costs, which outweighs the benefit of catching a few additional fraud cases. This reflects a deliberate choice to prioritize customer experience over maximizing fraud capture, a trade-off that depends on business context rather than a purely technical decision.

Technical Concepts Applied
Handling imbalanced datasets with SMOTE
Preventing data leakage by resampling after the train/test split and only within cross-validation folds
Using imblearn.pipeline.Pipeline instead of sklearn.pipeline.Pipeline to support resampling steps
Feature scaling considerations for linear versus tree-based models
Evaluation metric selection for imbalanced classification problems
Hyperparameter tuning with GridSearchCV across preprocessing and model parameters simultaneously
Tools and Libraries

Python, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn, Jupyter Notebook
