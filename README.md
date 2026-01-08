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
<!--                      DAY 14                     -->
<!-- =============================================== -->

# Day 14 — Handling Imbalanced Data: Oversampling & SMOTE

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
<!--                      DAY 15                     -->
<!-- =============================================== -->  

# Day 15 — Handling Imbalanced Data: ADASYN, Under-Sampling & Evaluation Metrics

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
<!--                      DAY 16                     -->
<!-- =============================================== -->

# Day 16 — Evaluation Metrics, Baseline Models & Tomek Links

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
- `day_16_evaluation_metrics_and_baselines.ipynb`

---

<!-- =============================================== -->
<!--                      DAY 17                     -->
<!-- =============================================== -->

#  Day 17 – Classification Metrics (Part 2)

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

---



<!-- =============================================== -->
<!--                       DAY 18                    -->
<!-- =============================================== -->

## Day 18 – Decision Tree Classification

### Overview
This day focuses on understanding **Decision Tree algorithms**, combining **theoretical concepts** with **hands-on implementation** using a classification dataset.

The objective is to learn how Decision Trees make decisions, how splits are formed, and how model complexity affects performance.

---

<!-- =============================================== -->
<!--               CONCEPTS COVERED                 -->
<!-- =============================================== -->

### Concepts Covered
- What is a Decision Tree and how it works
- Split criteria in Decision Trees
- Gini Impurity vs Entropy
- Overfitting in Decision Trees
- Role of `max_depth` in controlling model complexity

---

<!-- =============================================== -->
<!--               HANDS-ON PRACTICE                -->
<!-- =============================================== -->

### Hands-on Implementation
- Loaded and explored the dataset
- Identified target and feature columns
- Checked missing values and data types
- Performed train–test split (80–20)
- Built a Decision Tree Classifier using Gini impurity
- Trained the model and made predictions on test data

---

<!-- =============================================== -->
<!--               MODEL EVALUATION                 -->
<!-- =============================================== -->

### Model Evaluation
- Calculated accuracy score
- Analyzed predictions
- Observed model behavior related to overfitting and underfitting

---

<!-- =============================================== -->
<!--               KEY LEARNINGS                    -->
<!-- =============================================== -->

### Key Learnings
- Decision Trees are intuitive but prone to overfitting
- Controlling tree depth is crucial for generalization
- Choice of split criterion impacts decision boundaries
- Simple models can sometimes outperform complex ones

---

<!-- =============================================== -->
<!--               TOOLS USED                       -->
<!-- =============================================== -->

### Tools & Libraries Used
- Python
- pandas
- numpy
- scikit-learn

---

<!-- =============================================== -->
<!--               NOTE                             -->
<!-- =============================================== -->

### Note
This day emphasizes **core understanding** of Decision Trees.  
Further practice and reinforcement through assignments are continued in the following days.

---



<!-- =============================================== -->
<!--                      DAY 19                     -->
<!-- =============================================== -->

## Day 19 – Model Evaluation Metrics (Classification & Regression)

### Overview
This day focuses on a **deep dive into model evaluation metrics**, covering both **classification and regression problems**.

The emphasis is on understanding **how to choose the right metric based on problem context**, data distribution, and cost of misclassification rather than relying only on accuracy.

---

<!-- =============================================== -->
<!--               CONCEPTS COVERED                 -->
<!-- =============================================== -->

### Concepts Covered

#### Classification Metrics
- Precision
- Recall
- F1 Score
- Fβ Score
- Accuracy vs Balanced Accuracy
- Cost of Misclassification (FP vs FN)

#### Metric Selection
- When to prioritize Precision
- When to prioritize Recall
- Choosing metrics based on business objectives
- Baseline (mode-based) model concept

#### Regression Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² (R-Squared)
- Explained vs Unexplained variance (TSS, ESS, RSS)

---

<!-- =============================================== -->
<!--               HANDS-ON FOCUS                   -->
<!-- =============================================== -->

### Hands-on Focus
- Interpreted classification metrics using confusion matrix
- Analyzed metric behavior for balanced vs imbalanced datasets
- Understood regression performance using error-based metrics
- Compared trained models against baseline models

---

<!-- =============================================== -->
<!--               KEY LEARNINGS                    -->
<!-- =============================================== -->

### Key Learnings
- Accuracy alone is not reliable for imbalanced datasets
- Metric choice depends on the **cost of wrong predictions**
- Fβ score helps prioritize Precision or Recall as needed
- A useful model must always outperform a baseline
- Regression metrics quantify prediction errors and explained variance

---

<!-- =============================================== -->
<!--               TOOLS USED                       -->
<!-- =============================================== -->

### Tools & Libraries Used
- Python
- pandas
- numpy
- scikit-learn

---

<!-- =============================================== -->
<!--               NOTE                             -->
<!-- =============================================== -->

### Note
This day builds a **strong evaluation mindset**, which is essential for assessing and improving Machine Learning models in real-world scenarios.




---

<!-- =============================================== -->
<!--                      DAY 20                     -->
<!-- =============================================== -->

# Day 20— R², Adjusted R² & Model Selection

This notebook extends my Machine Learning learning journey by focusing on **regression model evaluation**, **feature impact**, and **generalization concepts**.

The session bridges the gap between **error-based metrics** and **variance-based metrics**, helping understand **why model selection matters beyond accuracy**.

---

##  Topics Covered

### Regression Evaluation Metrics
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

### Variance-Based Metrics
- R² (Coefficient of Determination)
- Limitations of R²
- Adjusted R² and feature penalization

### Model Selection Concepts
- Mean model as baseline
- Error decomposition (TSS, RSS)
- Overfitting due to unnecessary features
- Why Adjusted R² is preferred for comparison

### Generalization & Learning
- Memorization vs Generalization
- Hyperparameter tuning intuition
- Cross-validation overview
- Learning curves (high-level understanding)

---

##  Key Learnings

- R² explains **how much variance** a model captures
- R² alone is **not reliable** for feature selection
- Adjusted R² penalizes irrelevant features
- More features do not guarantee a better model
- Generalization is the ultimate goal of Machine Learning

---

##  Tools & Libraries

- Python
- NumPy
- pandas
- scikit-learn
- matplotlib

---

 *This notebook is part of my continuous Machine Learning learning journey, focusing on building strong conceptual foundations alongside practical understanding.*

 ---

 # ML-Learning-Journey
Daily hands-on notebooks exploring data preprocessing, EDA, categorical encoding, model building, and performance analysis.

---

<!-- =============================================== -->
<!--                      DAY 21                     -->
<!-- =============================================== -->

## Day 21 – Bias–Variance Tradeoff & Generalization

This notebook focuses on understanding one of the most fundamental concepts in Machine Learning —  
**Bias–Variance Tradeoff and model generalization**.

The session is theory-focused and explains *why models fail or succeed on unseen data*, going beyond accuracy and training performance.

---

<!-- =============================================== -->
<!--             CONCEPTS COVERED (DAY 21)           -->
<!-- =============================================== -->

## Concepts Covered

### Model Evaluation in Regression
- Loss functions and evaluation metrics
- MAE, MSE, RMSE, MAPE
- Interpretation of prediction error
- Comparing predicted values (ŷ) with actual values (y)

### R² and Adjusted R²
- Meaning of explained vs unexplained variance
- R² as a goodness-of-fit measure
- Limitations of R²
- Why R² always increases with more features
- Adjusted R² and feature penalization
- When to prefer Adjusted R² over R²

### Model Building Pipeline
- Data → ML Algorithm → Model → Predictions
- Training performance vs testing performance
- Why high training accuracy can be misleading

---

<!-- =============================================== -->
<!--           GENERALIZATION & OVERFITTING          -->
<!-- =============================================== -->

## Generalization Concepts
- What is model generalization
- Difference between memorization and learning
- Population vs sample data
- Population model vs sample model
- Importance of representative samples
- Role of randomness and independence in sampling

### Overfitting and Underfitting
- Underfitting as a result of high bias
- Overfitting as a result of high variance
- Train vs test error patterns
- Identifying overfitted and underfitted models

---

<!-- =============================================== -->
<!--            BIAS–VARIANCE TRADEOFF               -->
<!-- =============================================== -->

## Bias–Variance Tradeoff
- Bias:
  - Error due to incorrect assumptions
  - Simple models and underfitting
- Variance:
  - Sensitivity of model to data changes
  - Complex models and overfitting
- Tradeoff between bias and variance
- Why both cannot be minimized simultaneously

### Role of Model Complexity
- Effect of hyperparameters on bias and variance
- k in k-Nearest Neighbors
- max_depth in Decision Trees
- Simple vs complex model behavior

---

<!-- =============================================== -->
<!--          CROSS-VALIDATION & MODEL SELECTION     -->
<!-- =============================================== -->

## Cross-Validation & Model Selection
- Need for cross-validation
- Evaluating generalization performance
- Selecting models closer to population behavior
- Avoiding memorization during model selection

---

<!-- =============================================== -->
<!--                 KEY TAKEAWAYS                  -->
<!-- =============================================== -->

## Key Takeaways
- High accuracy alone does not guarantee a good model
- Generalization is the primary goal of Machine Learning
- Bias–Variance Tradeoff is central to model selection
- Model complexity must be carefully controlled
- Cross-validation helps estimate real-world performance

---

<!-- =============================================== -->
<!--               DAY 21 SUMMARY                   -->
<!-- =============================================== -->

## Day 21 Summary
Day 21 builds strong theoretical foundations required for:
- Hyperparameter tuning
- Model evaluation
- Understanding overfitting and underfitting
- Designing robust and generalizable ML models

This day connects theory directly to practical modeling decisions used in real-world Machine Learning workflows.

---


# ML-Learning-Journey
Daily hands-on notebooks exploring data preprocessing, EDA, categorical encoding, model building, and performance analysis.

---

<!-- =============================================== -->
<!--                     DAY 22                      -->
<!-- =============================================== -->

## Day 22 – Cross Validation & Hyperparameter Tuning

This notebook documents **Day 22** of my Machine Learning learning journey.  
The focus of this day is on understanding how to build **generalizable models** by addressing **overfitting**, **underfitting**, and **data leakage** using proper validation strategies.

The notebook combines **theoretical intuition** with **hands-on implementation** to understand why cross validation is a critical step in real-world ML workflows.

---

<!-- =============================================== -->
<!--                 CONCEPTS COVERED                -->
<!-- =============================================== -->

## Concepts Covered

### Generalization in Machine Learning
- Meaning of generalization
- Difference between training performance and real-world performance
- Why generalization is the true goal of ML models

### Bias–Variance Tradeoff
- Bias and variance explained intuitively
- Relationship between model complexity and error
- Underfitting vs overfitting scenarios
- Generalization error = Bias² + Variance + Noise

### Model Evaluation Challenges
- Limitations of train–test split
- Why repeated test evaluation is dangerous
- Understanding data leakage and its consequences

---

<!-- =============================================== -->
<!--            VALIDATION & CROSS VALIDATION        -->
<!-- =============================================== -->

## Validation & Cross Validation Techniques

### Validation Set (Hold-Out Method)
- Train–Validation–Test split
- Role of validation data in hyperparameter tuning
- Keeping test data untouched for final evaluation

### Cross Validation
- Motivation behind cross validation
- K-Fold Cross Validation workflow
- Why cross validation gives more reliable performance estimates

### Types of Cross Validation
- K-Fold Cross Validation
- Repeated K-Fold Cross Validation
- Stratified K-Fold Cross Validation (for classification)
- Leave-One-Out Cross Validation (LOOCV)
- Leave-P-Out Cross Validation (LPOCV)

---

<!-- =============================================== -->
<!--           HYPERPARAMETER TUNING                 -->
<!-- =============================================== -->

## Hyperparameter Tuning

- Role of hyperparameters in controlling model complexity
- Preventing overfitting during model selection
- Using cross validation for hyperparameter optimization
- Automated tuning using GridSearchCV

Models explored include:
- Decision Tree Classifier
- General cross-validation-based model selection workflow

---

<!-- =============================================== -->
<!--                 KEY LEARNINGS                   -->
<!-- =============================================== -->

## Key Learnings

- High training accuracy does not guarantee good generalization
- Bias–Variance tradeoff governs model behavior
- Test data should never be used during model tuning
- Cross validation provides stable and reliable evaluation
- Proper hyperparameter tuning improves real-world performance

---

<!-- =============================================== -->
<!--                 NEXT STEPS                      -->
<!-- =============================================== -->

## Next Steps

- Apply cross validation to additional models (KNN, Random Forest)
- Explore RandomizedSearchCV for efficient tuning
- Study nested cross validation for advanced workflows

---

 **Notebook Name:**  
`Day22_Cross_Validation_and_Hyperparameter_Tuning.ipynb`

---

<!-- =============================================== -->
<!--                     DAY 23                      -->
<!-- =============================================== -->


# Day 23 – Cross Validation & Hyperparameter Tuning

This notebook is part of my **Machine Learning Learning Journey**, focusing on understanding
model generalization, bias–variance tradeoff, and systematic techniques to evaluate and tune
machine learning models.

---

<!-- =============================================== -->
<!--                TOPICS COVERED                   -->
<!-- =============================================== -->

##  Topics Covered

- Prediction error in Machine Learning  
- Bias and its impact on underfitting  
- Variance and its role in overfitting  
- Bias–Variance tradeoff  
- Detecting underfitting and overfitting  
- Problems with using test data for tuning  
- Training, validation, and test data split  
- Cross Validation concept and workflow  
- Types of Cross Validation:
  - K-Fold
  - Repeated K-Fold
  - Stratified K-Fold
  - Leave-One-Out (LOOCV)
  - Leave-P-Out (LPOCV)
- Model generalization detection  
- Regularization techniques (L1, L2, Elastic Net)  
- Ensemble methods (Bagging & Boosting)  
- Within-subject vs Between-subject data design  

---

<!-- =============================================== -->
<!--               LEARNING OUTCOMES                 -->
<!-- =============================================== -->

##  Learning Outcomes

After completing this notebook, I can:

- Explain bias and variance intuitively  
- Identify underfitting and overfitting scenarios  
- Choose appropriate validation strategies  
- Apply cross validation for model evaluation  
- Avoid data leakage while tuning models  
- Improve model generalization using regularization and ensembles  

---

<!-- =============================================== -->
<!--              TOOLS & LIBRARIES                  -->
<!-- =============================================== -->

##  Tools & Libraries

- Python  
- NumPy  
- Pandas  
- scikit-learn  

---

<!-- =============================================== -->
<!--              REPO CONTEXT                       -->
<!-- =============================================== -->

##  Repository Context

This notebook is part of my **ML Learning Journey Repository**, where I document
daily hands-on practice covering:

- Data preprocessing & EDA  
- Feature engineering  
- Supervised learning algorithms  
- Model evaluation techniques  
- Performance optimization  
- End-to-end ML workflows  

---

 *Day 23 corresponds to the sequence in my personal ML learning roadmap.*

---

<!-- =============================================== -->
<!--                     DAY 25                      -->
<!-- =============================================== -->

# Day 25 – Hyperparameter Tuning & Linear Regression (Foundations)

This notebook is part of my **Machine Learning Learning Journey**, focusing on
**model selection**, **hyperparameter tuning strategies**, and building a strong
**conceptual foundation for Linear and Logistic Regression**.

The emphasis of this day is on understanding **why tuning is required**, how
different tuning strategies behave computationally, and **why simple linear
models are still critical in real-world ML workflows**.

---

<!-- =============================================== -->
<!--                TOPICS COVERED                   -->
<!-- =============================================== -->

## Topics Covered

### Hyperparameter Tuning (HPT)
- Why hyperparameter tuning is required
- Model generalization vs memorization
- Effect of poor hyperparameter configuration
- Hyperparameter tuning strategies:
  - Manual Search
  - Grid Search
  - Random Search
  - Sequential Search (Bayesian Optimization)
- Comparison of Grid vs Random Search
- Parallelization and computational cost
- When Grid / Random tuning becomes inefficient
- Introduction to Bayesian Optimization
- Surrogate models and Expected Improvement (EI)
- Optuna overview:
  - Study
  - Objective function
  - Trials
  - Best parameters and best score
- Tree-structured Parzen Estimator (TPE)

---

### Bias–Variance & Model Selection Context
- Bias–Variance tradeoff revisited
- Role of hyperparameters in controlling model complexity
- Overfitting vs underfitting from a tuning perspective
- Why tuning must be done using validation / cross-validation data
- Avoiding test data leakage during model selection

---

### Linear Regression – Conceptual Foundations
- Why Linear Regression is still important
- Linear Regression as a strong baseline model
- Interpretability vs complex models
- Feature importance and coefficient interpretation
- Comparison with Decision Trees and KNN
- Assumptions of Linear Regression:
  - Linearity
  - Independence
  - Homoscedasticity
  - No multicollinearity
  - Normality of errors

---

### Geometric Interpretation
- One feature → straight line
- Two features → plane
- Multiple features → hyperplane
- Linear Regression as best-fit hyperplane

---

### Learning the Parameters
- Ordinary Least Squares (OLS)
- Gradient Descent (iterative optimization)
- Objective: minimizing sum of squared errors
- Analytical vs iterative solutions

---

### Logistic Regression (High-Level Introduction)
- Logistic Regression as a GLM
- Difference between Linear & Logistic Regression
- Continuous vs probabilistic output
- Regression vs classification use cases

---

<!-- =============================================== -->
<!--               KEY LEARNINGS                     -->
<!-- =============================================== -->

## Key Learnings

- Hyperparameter tuning is essential for model generalization
- Grid Search is exhaustive but computationally expensive
- Random Search is often more efficient in large search spaces
- Bayesian Optimization reduces the number of required trials
- Linear Regression remains highly valuable due to interpretability
- Simple models should always be tried before complex ones
- Model selection is about **generalization**, not training accuracy

---

<!-- =============================================== -->
<!--               NOTEBOOK                          -->
<!-- =============================================== -->

## Notebook
- `Day_25_Hyperparameter_Tuning_and_Linear_Regression.ipynb`

---

*Day 25 strengthens the foundation required for advanced model optimization,
regularization, and future deep learning workflows.*

---




<!-- =============================================== -->
<!--         REPOSITORY STRUCTURE                    -->
<!-- =============================================== -->

# Repository Structure
```text
ML-Learning-Journey/
│
├── Day_01_Iris_EDA.ipynb
├── Day_02_Iris_Metrics_Model_Evaluation.ipynb
├── Day_03_Iris_Advanced_Evaluation.ipynb
├── Day_04_Iris_Scaling_Preprocessing.ipynb
├── Day_05_Diamond_Price_Prediction.ipynb
├── Day_06_Spam_vs_Ham_Classification.ipynb
├── Day_07_Spam_Ham_NLP_Case_Study.ipynb
├── Day_08_KNN_Classification.ipynb
├── Day_09_Decision_Tree_Classification.ipynb
├── Day_10_Feature_Engineering_Missing_Values.ipynb
├── Day_11_Advanced_Imputation_Pipelines.ipynb
├── Day_12_Imbalanced_Data_Oversampling.ipynb
├── Day_13_Outlier_Detection_Credit_Fraud.ipynb
├── Day_14_SMOTE_and_Imbalanced_Data.ipynb
├── Day_15_ADASYN_Breast_Cancer.ipynb
├── Day_16_Evaluation_Metrics_Imbalanced.ipynb
├── Day_17_Classification_Metrics_Part_2.ipynb
├── Day_18_Decision_Tree_Classification.ipynb
├── Day19_Model_Evaluation_Metrics.ipynb
├── Day_20_R2_Adjusted_R2_Model_Selection.ipynb
├── Day_21_Bias_Variance_Tradeoff_and_Generalization.ipynb
├── Day_22_Cross_Validation_and_Hyperparameter_Tuning.ipynb
├── Day_23_Cross_Validation_and_Hyperparameter_Tuning.ipynb
├── Day_25_Hyperparameter_Tuning_and_Linear_Regression.ipynb
└── README.md


