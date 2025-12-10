# ML-Learning-Journey
Daily hands-on notebooks exploring data preprocessing, EDA, categorical encoding, model building, and performance analysis.

---

<!-- =============================================== -->
<!--                ABOUT THE REPOSITORY             -->
<!-- =============================================== -->

## About This Repository
This repository documents my daily learning journey in Machine Learning.  
Each notebook covers a specific concept or dataset, combining both theory and practical implementation using:

- pandas  
- numpy  
- matplotlib  
- scikit-learn  

---

<!-- =============================================== -->
<!--                 CORE SKILLS COVERED             -->
<!-- =============================================== -->

## Core Skills Covered
- Exploratory Data Analysis (EDA)  
- Data Cleaning and Preprocessing  
- Feature Engineering  
- Handling Categorical and Numerical Data  
- Train–Test Split and Standardization  
- Supervised Learning Models  
- Model Evaluation and Hyperparameter Tuning  
- Building end-to-end ML pipelines  

---

<!-- =============================================== -->
<!--                     DAILY NOTES                 -->
<!-- =============================================== -->

# Daily Notes

---

<!-- =============================================== -->
<!--                      DAY 1                      -->
<!-- =============================================== -->

# Day 1 — Data Cleaning and Exploratory Data Analysis (EDA)

### Topics Covered
- Introduction to dataset  
- Handling missing values  
- Basic EDA  
- Understanding dataset structure using:  
  - `.shape`  
  - `.info()`  
  - `.describe()`  
- Univariate and bivariate analysis  
- Cleaning inconsistent entries  
- Visualizations (histograms, countplots, scatterplots)

Example:
```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.shape)
print(df.info())
print(df.describe())
```
---

<!-- =============================================== -->
<!--                      DAY 2                      -->
<!-- =============================================== -->

# Day 2 — Extended EDA and Preprocessing

**Topics Covered**
- Outlier detection using IQR
- Boxplot analysis
- Correlation analysis
- Heatmap visualization
- Feature understanding and preparation

---

<!-- =============================================== -->
<!--                      DAY 3                      -->
<!-- =============================================== -->

# Day 3 — Train–Test Split, Feature Scaling and Model Evaluation

**Concepts Applied**
- IRIS dataset
- Train–test split
- Feature scaling using StandardScaler

**Models**:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

**Evaluation Criteria**

| Train Accuracy – Test Accuracy | ≤ 5% |

---

<!-- =============================================== -->
<!--                      DAY 4                      -->
<!-- =============================================== -->

# Day 4 — Categorical Encoding and One-Hot Encoding
Concepts Covered
- Label Encoding
- One-Hot Encoding
- Handling encoded features
- Updated ML pipelines

---

<!-- =============================================== -->
<!--                      DAY 5                      -->
<!-- =============================================== -->

# Day 5 — Diamond Price Prediction (Regression)
Workflow
- Data cleaning and EDA
- Feature encoding
- Feature scaling
- Model training and evaluation
- Models Used
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

---

<!-- =============================================== -->
<!--                      DAY 6                      -->
<!-- =============================================== -->

# Day 6 — Spam vs Ham Classification (NLP Case Study)
Workflow
- Text preprocessing and cleaning
- TF-IDF vectorization
- Model training and evaluation
- Models Used
- Naive Bayes
- Logistic Regression
- Support Vector Machine (Best Model)

---

<!-- =============================================== -->
<!--                      DAY 7                      -->
<!-- =============================================== -->

# Day 7 — K-Nearest Neighbors (KNN) – Supervised Learning
Overview
- Focused on understanding the internal working of the K-Nearest Neighbors (KNN) algorithm using Euclidean distance and manual neighbor selection.
-**Topics Covered**
- Distance-based learning concepts
- Euclidean distance calculation
- Nearest neighbor selection
- Majority voting for classification
- Concept-first implementation of KNN

---

<!-- =============================================== -->
<!--                      DAY 8                     -->
<!-- =============================================== -->

## 🌳 Day 8 – Decision Tree (Supervised Learning)

In this stage of my machine learning learning journey, I explored the **Decision Tree algorithm**, a rule-based supervised learning model used for classification and regression. This notebook follows **KNN (Day 7)** and demonstrates the transition from a **distance-based approach** to a **decision-rule based and interpretable model**.

### Topics Covered
- Decision Tree intuition and working
- Classification trees
- Common split criteria (Gini, Entropy, Information Gain)
- Why feature scaling is not required for Decision Trees
- Handling categorical features using encoding
- Overfitting in Decision Trees and depth control
- Tree visualization and rule interpretation

### Implementation Highlights
- Built a **Decision Tree Classifier** using `scikit-learn`
- Trained on a small custom dataset
- Visualized the tree structure
- Extracted **human-readable if–else decision rules**
- Evaluated model performance using accuracy

### Key Learnings
- Decision Trees learn simple rules from data to make predictions
- No feature scaling is required for tree-based models
- Controlling tree depth is essential to prevent overfitting
- Decision Trees are interpretable but sensitive to data changes

---

<!-- =============================================== -->
<!--         REPOSITORY STRUCTURE                    -->
<!-- =============================================== -->

# Repository Structure
```text
ML-Learning-Journey/
│
├── Day1.ipynb
├── Day2.ipynb
├── Day3.ipynb
├── Day4.ipynb
├── Day5_DiamondPricePrediction.ipynb
├── Spam_Ham_Casestudy_day6.ipynb
├── Day7_KNN.ipynb
├── ML_Supervised_learning_DT.ipynb   # Day 8 – Decision Tree
└── README.md

