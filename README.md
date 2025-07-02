# Loan Approval Prediction Project

This project aims to predict loan approval decisions based on applicant information using a Logistic Regression classification model. The project utilizes F1-Score as the primary evaluation metric to ensure balanced consideration of both precision and recall, which is particularly important for handling class imbalance commonly found in loan approval datasets.

# Dataset

The dataset for this project was obtained from Kaggle, containing various features related to loan applicants, such as:

- Number of Dependents
- Education Status
- Employment Status
- Loan Amount
- Loan Term
- CIBIL Score (Credit Score)
- Residential and Commercial Asset Values
- Loan Approval Status (Target)

# Tools & Libraries

The project is built using:

- Python
- Streamlit (for interactive web app)
- Pandas, NumPy (data handling)
- Scikit-learn (modeling & evaluation)
- Imbalanced-learn (handling class imbalance)
- Matplotlib, Seaborn (visualizations)
- Joblib (model serialization)

# Project Highlights
✅ Logistic Regression model for binary classification

✅ F1-Score as the primary evaluation 

✅ Streamlit-based interactive web application for inference

✅ Exploratory Data Analysis (EDA) with visualizations

✅ Class imbalance handling using resampling techniques

# Evaluation Metric: F1-Score
F1-Score is chosen as the main metric due to its ability to balance:

- Precision: How many predicted approvals/rejections are correct
- Recall: How many actual approvals/rejections are correctly identified

This is crucial for real-world loan approval tasks where misclassification can have financial consequences.
