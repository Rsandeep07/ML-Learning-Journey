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

##  Core Skills & Techniques Covered

### Core Machine Learning Skills
- Data Cleaning and Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Handling Categorical and Numerical Data  
- Train–Test Split and Standardization  
- Supervised Learning Models  
- Model Evaluation and Hyperparameter Tuning  
- Building end-to-end ML pipelines  

### Hands-on Techniques Practiced
- Data loading and validation  
- Feature–target segregation  
- Train–test split  
- Baseline model building  
- Decision Tree modeling  
- Model complexity control  
- Feature importance analysis  
- Outlier handling techniques  

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
<!--                      DAY 8                      -->
<!-- =============================================== -->

#  Day 8 – Decision Tree (Supervised Learning)

- This notebook explores Decision Trees, transitioning from distance-based models (KNN) to rule-based and interpretable models.

### Topics Covered
- Decision Tree intuition and working
- Classification trees
- Common split criteria (Gini, Entropy, Information Gain)
- Why feature scaling is not required for Decision Trees
- Handling categorical features using encoding
- Overfitting in Decision Trees and depth control
- Tree visualization and rule interpretation

### Key Learnings
- Decision Trees learn simple rules from data to make predictions
- No feature scaling is required for tree-based models
- Controlling tree depth is essential to prevent overfitting
- Decision Trees are interpretable but sensitive to data changes

---

<!-- =============================================== -->
<!--                      DAY 9                      -->
<!-- =============================================== -->

# Day 9 - Wine Classification using KNN & Decision Tree

This notebook applies supervised learning models on the Wine dataset, focusing on model comparison and interpretability.

- **`Topics Covered`**
  - Dataset loading and validation
  - Minimal exploratory data analysis
  - Feature–target segregation
  - Baseline model using KNN
  - Accuracy-based evaluation
  - Decision Tree training and depth analysis
  - Tree visualization
  - Truncation for basic outlier handling
  - Feature importance extraction and visualization

### Key Focus
- Understanding model behavior, feature influence, and decision logic rather than optimization.




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
├── DecisionTrees_Classification_Regression.ipynb
├── Wine_Classification_KNN_DecisionTree.ipynb  # Day 9
└── README.md

