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
<!--                      DAY 10                     -->
<!-- =============================================== -->

# Day 10 — Missing Values, Feature Engineering & Model Evaluation

This notebook focuses on understanding **missing data mechanisms**, applying appropriate **imputation strategies**, and analyzing their impact on **model performance**.

### Topics Covered
- Importance of Feature Engineering in Machine Learning
- Role of domain knowledge in feature engineering
- Missing values identification and analysis
- Understanding why missing values occur in real-world data
- Missing value heuristics and decision rules

### Missing Value Mechanisms
- **MCAR (Missing Completely At Random)**
  - Missingness is purely random
  - No dependency on any feature or value
- **MAR (Missing At Random)**
  - Missingness depends on other observed features
- **MNAR (Missing Not At Random)**
  - Missingness depends on the value of the feature itself

### Imputation Strategies Applied
- **Simple Imputer**
  - Mean / Median / Mode
  - Constant value imputation (for learning purposes)
- **KNN Imputer**
  - Similarity-based imputation using nearest neighbors
  - Suitable for MAR-type missingness

### Key Heuristic Used
- If a feature has **≥ 40% missing values**:
  - Drop the feature if it is **fiscal/financial**
  - Otherwise, attempt **data collection or enrichment**

### Data Science Workflow Followed
- Dataset loading and understanding
- Missing value detection and percentage analysis
- Identification of missingness type (MCAR / MAR / MNAR)
- Appropriate imputation strategy selection
- Train–test split
- Model building with preprocessing
- Model performance evaluation
- Comparison of performance before and after imputation

### Best Practices Highlighted
- Imputation is performed **only on training data**
- Test data is transformed using the fitted imputer
- Prevention of **data leakage** using proper preprocessing flow
- Usage of **ColumnTransformer** for consistent transformations

### Key Takeaway
Feature engineering—especially handling missing values correctly—plays a crucial role in determining the success of a Machine Learning model. Model performance should always be evaluated before and after preprocessing decisions.


---

<!-- =============================================== -->
<!--                      DAY 11                     -->
<!-- =============================================== -->

# Day 11 — Advanced Missing Value Imputation & Preprocessing Pipelines

This notebook focuses on **advanced missing value handling techniques** and building
**production-style preprocessing pipelines** using Scikit-learn.

The goal is to move beyond basic imputation and understand how to handle
**mixed numerical and categorical data** in a clean, scalable way.

### Topics Covered
- Creating synthetic datasets to demonstrate imputation behavior
- Manual introduction of missing values
- KNN Imputation for numerical features
- Encoding categorical variables for imputation
- Iterative Imputer for multivariate missing value estimation
- Handling mixed data types (numerical + categorical)
- Random Forest–based imputation strategies
- ColumnTransformer for column-wise preprocessing
- Pipeline construction for end-to-end preprocessing
- Preventing data leakage using proper transformation flow

### Imputation Techniques Applied
- **KNN Imputer**
  - Distance-based imputation
  - Suitable for numerical features with similarity patterns
- **Iterative Imputer**
  - Predicts missing values using other features
  - Uses:
    - Decision Tree estimator
    - Random Forest Classifier (categorical features)
    - Random Forest Regressor (numerical features)

### Key Learnings
- Simple imputation is not always sufficient
- Feature relationships can significantly improve imputation quality
- Categorical and numerical features must be handled differently
- Pipelines ensure reproducibility and cleaner ML workflows
- Proper preprocessing is essential before model training

### Notebook
- `missing_value_imputation_pipeline.ipynb`

---

<!-- =============================================== -->
<!--                    DAY 12                       -->
<!-- =============================================== -->

## 📅 Day 12: Handling Imbalanced Data & Oversampling Techniques

On Day 12 of my Machine Learning learning journey, I focused on understanding the problem of **imbalanced datasets**, why they affect model performance, and explored different strategies to handle class imbalance using data-level, algorithm-level, and ensemble-based approaches.

---

### 📌 Topics Covered

#### 🔹 What is Imbalanced Data?
- When target classes are not represented equally in a dataset  
- Majority vs Minority class concept  
- Real-world examples (fraud detection, disease prediction, churn, etc.)

#### 🔹 Why Imbalanced Data is a Problem?
- Most ML algorithms assume balanced class distribution  
- Leads to biased models favoring the majority class  
- High accuracy but poor minority class performance  

#### 🔹 Detecting Imbalance
- Using class distribution:
```python
y.value_counts()
```

---

<!-- =============================================== -->
<!--                      DAY 13                     -->
<!-- =============================================== -->

# Day 13 — Outlier Detection | Credit Card Fraud Detection

This notebook focuses on understanding **outliers and anomalies** in data and applying
**unsupervised outlier detection algorithms** to a real-world **Credit Card Fraud Detection** dataset.

### Topics Covered
- What is an outlier and why it matters in Machine Learning  
- Types of outliers:
  - Errors and noise  
  - Rare events (fraud/anomalies)  
  - Natural extreme values  
- Impact of outliers on:
  - Statistical measures  
  - Model performance and generalization  
- Difference between:
  - **Univariate outlier detection**  
  - **Multivariate outlier detection**  

### Outlier Detection Approaches
- Statistical intuition for univariate methods  
- Distance & density-based methods for multivariate data  
- Overview of popular libraries:
  - **PyOD** ecosystem  
  - Scikit-learn implementations  

### Algorithms Explored
- **Isolation Forest**
  - Tree-based anomaly detection  
  - Isolates rare points using random splits  
  - Uses anomaly score and path length intuition  
- **Local Outlier Factor (LOF)**
  - Density-based approach  
  - Compares local density of a point with its neighbors  
  - Detects samples that lie in sparse regions  

### Key Concepts
- **Contamination parameter**
  - Represents expected proportion of anomalies  
  - Used to set threshold for labeling fraud cases  
- Mapping anomaly predictions to binary fraud labels  
- Importance of handling highly imbalanced anomaly datasets  

### Case Study: Credit Card Fraud Detection
- Dataset loading and inspection  
- Severe class imbalance analysis  
- Duplicate record removal  
- Feature–target segregation  
- Train–test split with stratification  
- Baseline dummy prediction to show imbalance bias  
- Applying:
  - Isolation Forest on training data  
  - LOF with contamination tuned to fraud ratio  
- Evaluation using accuracy to highlight imbalance effects  

### Key Learnings
- Outlier detection is crucial for rare-event problems like fraud  
- Unsupervised models can detect anomalies without labels  
- Isolation Forest is efficient for high-dimensional data  
- LOF is powerful for local density-based anomalies  
- Accuracy alone is misleading for imbalanced anomaly detection  
- Proper contamination setting strongly influences results  

### Focus
To understand anomaly detection techniques and apply them to a real-world fraud dataset, emphasizing intuition, workflow, and challenges of imbalanced data.

---

<!-- =============================================== -->
<!--                      DAY 15                     -->
<!-- =============================================== -->

# Day 15 — Handling Imbalanced Data: Oversampling & SMOTE

This notebook focuses on **data-level techniques** to address class imbalance and improve model learning for minority classes.

### Topics Covered
- Understanding class imbalance and its impact on ML models  
- Data-level approaches: Over-sampling vs Under-sampling  
- Random Over-sampling (ROS) and its limitations  
- Over-sampling with noise and shrinkage factor concept  
- SMOTE: Synthetic Minority Over-sampling Technique  
- How synthetic samples are generated using interpolation  
- When to use:
  - SMOTE (numerical features)  
  - SMOTENC (mixed features)  
  - SMOTEN (categorical features)  
- Class distribution comparison before and after resampling  

### Key Learnings
- Imbalanced data can bias models toward the majority class  
- Random oversampling may lead to overfitting due to duplication  
- Adding noise introduces variation but must be controlled  
- SMOTE generates meaningful synthetic samples instead of copies  
- Choosing the right resampling strategy depends on data type  

### Focus
To understand how resampling techniques help models learn better decision boundaries and improve minority class performance.

---

<!-- =============================================== -->
<!--                      DAY 16                     -->
<!-- =============================================== -->  

# Day 16 — Handling Imbalanced Data: ADASYN, Under-Sampling & Evaluation Metrics

This notebook extends the imbalanced learning concepts from Day 15 by exploring
**adaptive over-sampling** and **cleaning-based under-sampling techniques**, applied on the
**Breast Cancer dataset**, along with a revision of key evaluation metrics.

### Topics Covered
- Limitations of basic over-sampling and SMOTE  
- ADASYN: Adaptive Synthetic Sampling technique  
- KNN-based difficulty score for minority samples  
- How ADASYN focuses on hard-to-learn boundary points  
- Under-sampling strategies:
  - Random Under-Sampling (RUS)  
  - Tomek Links for boundary cleaning  
  - Condensed Nearest Neighbors (CNN)  
- Applying resampling **only on training data**  
- Feature scaling before distance-based resampling  
- Class distribution comparison after each method  

### Dataset Used
- **Breast Cancer Wisconsin Dataset** (`sklearn.datasets`)
- Binary classification problem:
  - Malignant vs Benign

### Key Learnings
- Not all minority samples are equally important — ADASYN adapts to data difficulty  
- ADASYN generates more samples near decision boundaries  
- Random under-sampling is fast but risks information loss  
- Tomek Links help remove noisy and overlapping majority samples  
- CNN retains only critical boundary samples for compact datasets  
- Resampling strategies significantly affect class distributions and model learning  
- For imbalanced data, accuracy alone is misleading — better to use:
  - Precision, Recall, F1-score, ROC-AUC  

### Focus
To understand **adaptive over-sampling**, **cleaning-based under-sampling**, and how
different imbalance handling techniques reshape the dataset and influence learning
before model training.


---

<!-- =============================================== -->
<!--                      DAY 17                     -->
<!-- =============================================== -->

# Day 17 — Evaluation Metrics, Baseline Models & Tomek Links

This notebook focuses on understanding how to **evaluate machine learning models properly** and how to use
**simple baseline models** as benchmarks, along with revisiting **Tomek Links** as a boundary cleaning technique
for imbalanced datasets.

The goal is to strengthen intuition around **model usefulness** rather than just building models.

### Topics Covered
- Classification vs Regression problems  
- Role of evaluation in Machine Learning  
- Baseline models:
  - **Mean model** for regression  
  - **Mode model** for classification  
- Why baseline models are important as benchmarks  
- Revisiting imbalanced data challenges  
- **Tomek Links**:
  - Nearest neighbor pairs from opposite classes  
  - Removing majority class samples near decision boundary  
- Understanding **True / False** and **Positive / Negative** cases  
- Confusion Matrix:
  - TP, TN, FP, FN  

### Evaluation Metrics — Classification
- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **F1 Score**
- **ROC–AUC Curve**
- **Precision–Recall Curve**
- Why accuracy alone is misleading for imbalanced datasets  

### Regression Evaluation
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  
- Comparing ML models against the **Mean baseline model**  

### Real-World Intuition
- COVID-19 example:
  - Positive → infected  
  - Negative → not infected  
- Spam vs Ham email classification  
- Why we mainly focus on the **positive class** in evaluation  

### Key Learnings
- Always build a simple baseline before training ML models  
- ML models are useful only if they outperform baselines  
- Precision and Recall are more meaningful than Accuracy for imbalanced data  
- Confusion matrix forms the base for all classification metrics  
- Tomek Links help clean noisy majority samples near class boundaries  
- Proper evaluation is critical for reliable ML systems  

### Focus
To understand **how to judge model performance**, use **baseline benchmarks**, and apply
evaluation metrics correctly, especially in the presence of **imbalanced datasets**.

### Notebook
- `day_17_evaluation_metrics_and_baselines.ipynb`

---

<!-- =============================================== -->
<!--                      DAY 18                     -->
<!-- =============================================== -->

#  Day 18 – Classification Metrics (Part 2)

This notebook is part of my **ML Learning Journey**, focusing on a deep understanding of **classification evaluation metrics** and how to choose the **right metric based on real-world problem context**.

---

##  Objective
To understand how classification models are evaluated using confusion-matrix–based metrics and to learn **when and why** each metric should be used.

---

##  Topics Covered

###  Confusion Matrix
- True Positive (TP)  
- True Negative (TN)  
- False Positive (FP)  
- False Negative (FN)  

###  Classification Metrics
- Accuracy  
- Precision (Positive Predictive Value)  
- Recall (Sensitivity / True Positive Rate)  
- Specificity (True Negative Rate)  
- False Positive Rate (FPR)  
- False Negative Rate (FNR)  
- F1 Score (Harmonic Mean of Precision & Recall)

---

##  Metric Formulae

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)  
- **Precision** = TP / (TP + FP)  
- **Recall** = TP / (TP + FN)  
- **Specificity** = TN / (TN + FP)  
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)

---

##  Key Learning: Choosing the Right Metric

Metric selection depends on the **cost of misclassification**:

| Scenario | Costly Error | Preferred Metric |
|--------|-------------|------------------|
| Spam Detection | False Positive | Precision |
| Disease Screening | False Negative | Recall |
| Fraud Detection | False Negative | Recall |
| Balanced Importance | FP & FN | F1 Score |

---

##  Real-World Case Studies
- **Spam vs Ham classification**
- **Healthcare / Covid testing**
- Understanding why accuracy alone is often misleading

---

##  Key Takeaways
- Confusion matrix is the foundation of all classification metrics  
- Accuracy is not reliable for imbalanced datasets  
- Precision, Recall, and F1 Score provide better insights  
- Metric choice must align with **business and real-world impact**

---

##  Notebook Included
- `Day_18_Classification_Metrics_Part_2.ipynb`

---

##  Part of ML Learning Journey
This notebook continues my structured, day-wise exploration of Machine Learning concepts with both **theory and intuition**.







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
├── Day10_Missing_Values_Feature_Engineering.ipynb
├── Day11_missing_value_imputation.ipynb
├── Day12_Handling_Imbalanced_Data_and_Oversampling.ipynb
├── Day15_Imbalanced_Data_SMOTE.ipynb        # Handling Imbalanced Data
├── Day16_Imbalanced_Data_ADASYN_Breast_Cancer.ipynb
├── day_17_evaluation_metrics_and_baselines.ipynb
├── Day_18_Classification_Metrics_Part_2.ipynb
└── README.md

