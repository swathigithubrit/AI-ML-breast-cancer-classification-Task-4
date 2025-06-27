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


📋 Dataset Features and Their Role
Below is a description of each column in the dataset, and why it is useful for predicting whether a breast tumor is benign or malignant:

This dataset contains features computed from digitized images of breast mass fine-needle aspirate (FNA) biopsies. Each feature describes characteristics of the cell nuclei present in the image.


|   Column                  | Description                                              | Use in Prediction                                             |
| ------------------------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| `id`                      | Unique identifier for each patient/sample                | Not predictive. Used only for reference; usually dropped.     |
| `diagnosis`               | Diagnosis label: **M** = malignant, **B** = benign       | **Target variable** (converted to 1 for M, 0 for B).          |
| `radius_mean`             | Mean of distances from center to points on the perimeter | Larger radii often indicate malignancy.                       |
| `texture_mean`            | Variation in gray-scale values                           | Captures cell uniformity; high variation may indicate cancer. |
| `perimeter_mean`          | Mean size of the perimeter of the mass                   | Tumors with irregular shapes often have higher perimeters.    |
| `area_mean`               | Mean area of the mass                                    | Larger areas often correlate with malignancy.                 |
| `smoothness_mean`         | Variation in radius lengths                              | Less smooth (more irregular) can suggest malignancy.          |
| `compactness_mean`        | Perimeter² / area - 1.0                                  | Higher values may indicate irregular, complex shapes.         |
| `concavity_mean`          | Severity of concave portions of the contour              | Malignant tumors often have more pronounced concavities.      |
| `concave points_mean`     | Number of concave portions of the contour                | More concave points typically signal malignancy.              |
| `symmetry_mean`           | Symmetry of the cell shape                               | Asymmetry is a common indicator of malignancy.                |
| `fractal_dimension_mean`  | Complexity of the contour’s shape                        | Higher values indicate more complex, irregular borders.       |
| `radius_se`               | Standard error of radius measurements                    | Variability in radius can reveal heterogeneity.               |
| `texture_se`              | Standard error of texture measurements                   | Variability hints at inconsistent cell structure.             |
| `perimeter_se`            | Standard error of perimeter measurements                 | Captures shape irregularities.                                |
| `area_se`                 | Standard error of area measurements                      | Reflects variability in size.                                 |
| `smoothness_se`           | Standard error of smoothness measurements                | Indicates local irregularities.                               |
| `compactness_se`          | Standard error of compactness measurements               | Variability in compactness hints at heterogeneity.            |
| `concavity_se`            | Standard error of concavity measurements                 | Indicates shape variation.                                    |
| `concave points_se`       | Standard error of concave points measurements            | Variability of concave features is informative.               |
| `symmetry_se`             | Standard error of symmetry measurements                  | Indicates inconsistency in shape.                             |
| `fractal_dimension_se`    | Standard error of fractal dimension measurements         | Captures local border irregularity.                           |
| `radius_worst`            | Largest (worst) radius measurement across all images     | Helps identify extreme abnormal growth.                       |
| `texture_worst`           | Worst (largest) texture measurement                      | Captures maximum heterogeneity.                               |
| `perimeter_worst`         | Worst perimeter measurement                              | Highlights extreme shape irregularity.                        |
| `area_worst`              | Largest area observed                                    | Points to potentially large, aggressive tumors.               |
| `smoothness_worst`        | Worst smoothness measurement                             | Emphasizes most irregular contours.                           |
| `compactness_worst`       | Worst compactness measurement                            | Captures highest complexity in shape.                         |
| `concavity_worst`         | Worst concavity measurement                              | Indicates most severe contour indentations.                   |
| `concave points_worst`    | Worst count of concave points                            | Detects highest number of suspicious regions.                 |
| `symmetry_worst`          | Worst symmetry measurement                               | Asymmetry at its most severe is a malignancy indicator.       |
| `fractal_dimension_worst` | Worst fractal dimension measurement                      | Captures extreme border complexity.                           |


This project builds a logistic regression model to classify breast tumors as benign or malignant with ~96% accuracy. Careful data cleaning, scaling, and evaluation (including ROC and confusion matrices) show the model’s strong ability to support early cancer diagnosis, highlighting the potential of machine learning in medical decision-making.
