<!-- ===================================================== -->
<!-- ML LEARNING JOURNEY - COMPLETE README (DAY 1 TO DAY 4) -->
<!-- ===================================================== -->

# ML-Learning-Journey  
Daily hands-on notebooks exploring **data preprocessing**, **EDA**, **categorical encoding**, **model building**, and **performance analysis**.

---

##  About This Repository  
This repository documents my **daily learning journey in Machine Learning**.  
Each notebook focuses on one concept or dataset, combining both **theory and practical implementation** using Python libraries such as `pandas`, `numpy`, `matplotlib`, and `scikit-learn`.

The goal is to build a strong foundation in:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Handling Categorical & Numerical Data
- Supervised and Unsupervised Learning
- Train–Test Splits, Scaling & Encoding
- Model Building, Evaluation, and Hyperparameter Tuning
- Real-world ML Project Workflows

---

#  Daily Notes  

---

<!-- ============================ -->
<!--        DAY 1 CONTENT         -->
<!-- ============================ -->

#  Day 1 — Data Cleaning & Exploratory Data Analysis (EDA)

###  Topics Covered:
- Introduction to the dataset  
- Handling missing values  
- Basic EDA  
- Understanding dataset structure:  
  - `.shape`  
  - `.info()`  
  - `.describe()`  
- Univariate & bivariate analysis  
- Cleaning inconsistent entries  
- Visualizations:  
  - histograms  
  - countplots  
  - scatterplots  

---

<!-- ============================ -->
<!--        DAY 2 CONTENT         -->
<!-- ============================ -->

#  Day 2 — Extended EDA & Preprocessing

### ✔ Topics Covered:
- Outlier detection (IQR, boxplots)  
- Treating or removing outliers  
- Correlation analysis  
- Heatmap visualization  
- Advanced feature understanding  
- Dataset preparation for ML models  

---

<!-- ============================ -->
<!--        DAY 3 CONTENT         -->
<!-- ============================ -->

#  Day 3 — Data Loading, Train–Test Split & Model Evaluation  
(Ends at: **“Differences between train and test scores (accuracy) <= 5”**)

###  Loading the Dataset  
- Load IRIS dataset using `load_iris()`  
- Extract features, target & names  
- Convert into DataFrame  
- Add target column  

###  Basic EDA  
- `.head()`  
- Unique target values  
- Feature distribution  

###  Train–Test Split  
- Split features & labels using `train_test_split`  
- Create train & test datasets  

###  Feature Scaling  
- Standardize data using `StandardScaler()`  

###  Model Training  
Models experimented with:
- Logistic Regression  
- KNN  
- Decision Tree / Random Forest  

###  Model Evaluation  
- Train accuracy  
- Test accuracy  

###  Final Day 3 Check:  
Ensure:

```text
| Train Accuracy – Test Accuracy |  <= 5%
- This helps detect overfitting or underfitting and ensures model stability.

<!-- ============================ -->
<!--        DAY 4 CONTENT         -->
<!-- ============================ -->

# Day 4 — Categorical Encoding & One-Hot Encoding
(Starts from: “Categorical Encoding and One Hot Encoding”)

### Why Encoding is Needed?
- Machine Learning models work only with numerical data → categorical columns must be encoded.

---

## 🔹 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['column'] = le.fit_transform(df['column'])


### Day 4 — Categorical Encoding & One-Hot Encoding:
(Starts from: “Categorical Encoding and One Hot Encoding”)

 Why Encoding is Needed?
- Machine Learning models work only with numerical data → categorical columns must be encoded.
- Label Encoding
    - Converts categories into integer labels.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['column'] = le.fit_transform(df['column'])

- One-Hot Encoding
Creates binary columns for each category.

```Using Pandas
pd.get_dummies(df, drop_first=True)

- Using Scikit-Learn:
```Python
from sklearn.preprocessing import OneHotEncoder

- Handling Encoded Data
  - Combine encoded columns
  - Drop original categorical columns
  - Final ML-ready DataFrame creation

## Updated ML Pipeline
- Encode categorical variables
- Train–test split
- Standardize numerical features
- Train ML model
- Evaluate model performance
- Compare with Day 3 metrics

ML-Learning-Journey/
│
├── Day1.ipynb
├── Day2.ipynb
├── Day3.ipynb
├── Day4.ipynb
└── README.md
