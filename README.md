AI-ML-breast-cancer-classification-Task-4
Predict breast cancer diagnosis (Malignant or Benign) using logistic regression in Python. Includes data cleaning, feature scaling, model training, evaluation with confusion matrix and ROC curve.


Breast Cancer Classification using Logistic Regression

This repository contains a complete machine learning project in Python to classify breast cancer tumors as Malignant (Cancerous) or Benign (Non-cancerous) using the Breast Cancer Wisconsin Diagnostic dataset.

The project demonstrates an end-to-end ML workflow, including:
- Data cleaning
- Exploratory Data Analysis (EDA)
- Preprocessing
- Feature scaling
- Model training
- Evaluation with Confusion Matrix, ROC curve, and AUC
- Threshold tuning

---

📌 Dataset

The dataset used is the Breast Cancer Wisconsin (Diagnostic) Data Set, typically available via UCI ML Repository or Kaggle.

- Each row represents measurements from a biopsy.
- Features are real-valued (e.g., radius, texture, perimeter, area, etc.).
- Target variable:  
  - M (Malignant) → 1
  - B (Benign) → 0

---

📌 Project Steps

1️⃣ Data Loading
- Loaded CSV data using pandas.
- Inspected structure with `.head()`, `.info()`, and `.describe()`.

2️⃣ Cleaning
- Removed unnecessary columns:
  - `id` (identifier)
  - `Unnamed: 32` (empty/garbage)
- Stripped spaces from column names.
- Mapped `diagnosis` column to numeric `target`.

3️⃣ Exploratory Data Analysis
- Checked class balance using `value_counts`.
- Verified feature types and cleaned column list.

4️⃣ Train-Test Split
- Split data into 80% training and 20% testing.
- Used stratification to maintain class balance.

5️⃣ Feature Scaling
- Applied `StandardScaler` to normalize features.
- Ensured mean=0, std=1 for better model convergence.

6️⃣ Model Training
- Trained a Logistic Regression model using scikit-learn.
- Chose Logistic Regression for its interpretability and baseline performance.

7️⃣ Predictions
- Generated class predictions and probability scores.

8️⃣ Evaluation
- Created Confusion Matrix to assess TP, TN, FP, FN.
- Printed Classification Report with precision, recall, F1-score, accuracy.
- Plotted ROC Curve and calculated AUC score.

9️⃣ Threshold Tuning
- Experimented with changing the classification threshold from 0.5 to 0.4.
- Observed the effect on confusion matrix and metrics.

---

📌 Technologies Used

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---


📈 Classification Report Explanation

              precision    recall  f1-score   support

           0       0.96      0.99      0.97        72
           1       0.97      0.93      0.95        42

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114


- Class 0 (Benign):
  - Precision: 0.96
    - 96% of predicted benign tumors were actually benign.
  - Recall: 0.99
    - 99% of actual benign tumors were correctly detected.
  - F1-score: 0.97
    - High balance of precision and recall.


- Class 1 (Malignant):
  - Precision: 0.97
    - 97% of predicted malignant tumors were actually malignant.
  - Recall: 0.93
    - 93% of actual malignant tumors were correctly detected.
  - F1-score: 0.95
    - Slightly lower recall than precision, but still strong.


- Accuracy: 0.96
  - Overall, 96% of test samples were classified correctly.


- Macro average:
  - Simple average across classes (treats classes equally).
  - Precision: 0.97, Recall: 0.96, F1-score: 0.96.


- Weighted average:
  - Accounts for class imbalance.
  - Precision: 0.97, Recall: 0.96, F1-score: 0.96.


This project builds a logistic regression model to classify breast tumors as benign or malignant with ~96% accuracy. Careful data cleaning, scaling, and evaluation (including ROC and confusion matrices) show the model’s strong ability to support early cancer diagnosis, highlighting the potential of machine learning in medical decision-making.
