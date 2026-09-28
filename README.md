Bank Loan Defaulter Prediction

A machine learning project that predicts whether a bank loan applicant is likely to become a defaulter based on financial, credit, employment, and loan-related attributes.

Note: The dataset used in this project is synthetic and intended for academic/demo purposes. It does not contain real bank customer information.

Project Overview

Loan default prediction is a binary classification problem where the model learns patterns associated with repayment risk.

The project includes:

Data preprocessing

Feature-based risk prediction

Multiple machine learning models

5-fold stratified cross-validation

Model comparison

Accuracy, Precision, Sensitivity/Recall, F1 and ROC-AUC

Confusion matrix

ROC curve

Predictions for the complete 1,000-row dataset

Dataset

The dataset contains 1,000 synthetic loan records.

Main Features

Feature

Description

Customer_ID

Unique customer identifier

Name

Customer name

Age

Customer age

Gender

Gender category

Employment_Type

Employment category

Monthly_Salary_INR

Monthly income

Loan_Amount_INR

Requested loan amount

Loan_Term_Months

Loan duration

Credit_Score

Credit score

Debt_to_Income_Percent

Debt-to-income ratio

Existing_EMI_INR

Existing monthly EMI

Missed_Payments_12M

Missed payments during previous 12 months

Previous_Default

Previous default indicator

Credit_Utilization_Percent

Credit utilization

Employment_Years

Years of employment

Savings_INR

Approximate savings

Collateral_Value_INR

Collateral value

Loan_Purpose

Purpose of the loan

Default

Target variable: 0 = Non-Defaulter, 1 = Defaulter

Machine Learning Models

The project compares three classification algorithms:

Extra Trees Classifier

Random Forest Classifier

Gradient Boosting Classifier

The final model is selected using 5-fold stratified cross-validation, with ROC-AUC used as the primary comparison metric.

Evaluation Metrics

The model reports:

Accuracy – percentage of correctly classified customers

Precision – proportion of predicted defaulters that are actually defaulters

Sensitivity / Recall – proportion of actual defaulters correctly detected

F1 Score – balance between precision and recall

ROC-AUC – ability of the model to distinguish defaulters from non-defaulters

Project Workflow

Dataset
   ↓
Data Loading
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Categorical Encoding
   ↓
5-Fold Cross Validation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Test Prediction
   ↓
Performance Evaluation
   ↓
ROC Curve + Confusion Matrix
   ↓
1,000 Customer Predictions

Project Structure

loan-defaulter-prediction/
│
├── prediction.py
├── strong_loan_defaulter_dataset_1000.csv
├── strong_loan_defaulter_dataset_1000.txt
├── README.md
│
├── confusion_matrix.png
├── roc_curve.png
├── model_comparison.png
│
├── test_predictions.csv
└── all_1000_predictions.csv

Installation

Install the required Python libraries:

pip install pandas numpy scikit-learn matplotlib seaborn

How to Run

Clone the repository:

git clone https://github.com/YOUR_USERNAME/loan-defaulter-prediction.git

Open the project folder:

cd loan-defaulter-prediction

Run the model:

python prediction.py

Output

After execution, the program generates:

test_predictions.csv

Predictions for the unseen test dataset, including:

Customer ID

Customer name

Financial information

Actual status

Predicted status

Default probability

all_1000_predictions.csv

Predictions and default probabilities for all 1,000 customers.

Visualization Files

confusion_matrix.png
roc_curve.png
model_comparison.png

Example Output

=================================================
FINAL UNSEEN TEST PERFORMANCE
=================================================

Model               : Extra Trees
Accuracy            : XX.XX%
Precision           : XX.XX%
Sensitivity/Recall  : XX.XX%
F1 Score            : XX.XX%
ROC-AUC             : X.XXXX

The exact values can vary depending on the model configuration and dataset version.

Why ROC-AUC?

Accuracy alone can be misleading in credit-risk classification, especially when the number of defaulters and non-defaulters is unbalanced.

ROC-AUC evaluates how well the model separates the two classes across different classification thresholds.

Limitations

The dataset is synthetic and should not be used for real-world lending decisions.

Real banking systems require much larger and independently validated datasets.

Actual credit-risk systems require regulatory, fairness, privacy, security, and explainability considerations.

Model performance on this synthetic dataset does not represent performance on real bank customers.

Future Improvements

Hyperparameter optimization using GridSearchCV or RandomizedSearchCV

XGBoost / LightGBM comparison

SHAP-based model explainability

Probability calibration

Threshold optimization based on business cost

Handling class imbalance with appropriate sampling methods

Deployment using Flask or FastAPI

Web-based prediction dashboard

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Machine Learning

Binary Classification

Ensemble Learning

Author

Varnika Singh

B.Tech CSE

This project was developed as an academic machine learning project for demonstrating loan-default prediction and model evaluation.