# ML-Learning-Journey  
Daily hands-on notebooks exploring **data preprocessing**, **EDA**, **categorical encoding**, **model building**, and **performance analysis**.

---

## 📌 About This Repository  
This repository documents my **daily learning journey in Machine Learning**.  
Each notebook focuses on one concept or dataset, combining both **theory and practical implementation** using Python libraries such as:

- pandas  
- numpy  
- matplotlib  
- scikit-learn  

---

## 🎯 Core Skills Covered
- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- Handling Categorical & Numerical Data
- Train–Test Split & Standardization
- Supervised Learning Models
- Model Evaluation & Hyperparameter Tuning
- Real-world ML Pipelines

---

# 📘 Daily Notes

---

# 🔹 Day 1 — Data Cleaning & Exploratory Data Analysis (EDA)

### ✔ Topics Covered
- Introduction to dataset  
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

# 🔹 Day 2 — Extended EDA & Preprocessing

### ✔ Topics Covered
- Outlier detection:
  - IQR  
  - Boxplots  
- Treating/removing outliers  
- Correlation analysis  
- Heatmap visualization  
- Advanced feature understanding  
- Dataset preparation for ML models  

---

# 🔹 Day 3 — Train–Test Split, Feature Scaling & Model Evaluation

### ✔ Steps Performed

### **1. Loading the Dataset**
- Loaded IRIS dataset using `load_iris()`  
- Converted features + target to DataFrame  

### **2. Basic EDA**
- `.head()`, unique targets, feature distributions  

### **3. Train–Test Split**
```python
from sklearn.model_selection import train_test_split

### **4. Feature Scaling**
from sklearn.preprocessing import StandardScaler


### **5. Models Used**
- Logistic Regression
- KNN
- Decision Tree / Random Forest

### **6. Model Evaluation**
- Train accuracy
- Test accuracy

### **Final Day 3 Requirement**
- | Train Accuracy – Test Accuracy | <= 5%

# 🔹 Day 4 — Categorical Encoding & One-Hot Encoding

(Starts from “Categorical Encoding and One Hot Encoding”)
- Why Encoding is Needed?
- ML models work only with numbers, not text
- Categorical features must be encoded

# Label Encoding
- Converts categories → numbers.
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['column'] = le.fit_transform(df['column'])

# One-Hot Encoding
pd.get_dummies(df, drop_first=True)



```Using Scikit-Learn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse=False)
encoded = ohe.fit_transform(df[['column']])


# Handling Encoded Data
- Convert encoded arrays to DataFrame
- Merge with original data
- Drop original categorical columns
- Final ML-ready DataFrame created

# pdated ML Pipeline After Day 4
  - Encode categorical variables
  - Train–test split
  - Standardize numerical features
  - Train ML model
  - Evaluate performance
  - Compare metrics with Day 3

# Repository Structure
ML-Learning-Journey/
│
├── Day1.ipynb
├── Day2.ipynb
├── Day3.ipynb
├── Day4.ipynb
└── README.md
