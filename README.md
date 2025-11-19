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

Example (inspect dataset):
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

### Topics Covered
- Outlier detection methods:  
  - IQR (Interquartile Range)  
  - Boxplots  
- Outlier treatment or removal  
- Correlation analysis  
- Heatmap visualization  
- Understanding feature relationships  
- Preparing dataset for ML models  

Example (IQR outlier detection):
```python
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df_filtered = df[(df['column'] >= Q1 - 1.5 * IQR) & (df['column'] <= Q3 + 1.5 * IQR)]
```

Example (correlation heatmap):
```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()
```

---

<!-- =============================================== -->
<!--                      DAY 3                      -->
<!-- =============================================== -->

# Day 3 — Train–Test Split, Feature Scaling and Model Evaluation

### Steps Performed

### 1. Loading the Dataset
- Used IRIS dataset via `load_iris()`  
- Converted into pandas DataFrame  

Example:
```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
```

### 2. Basic EDA
- `.head()`, `.unique()`, feature inspection  

### 3. Train–Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 4. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5. Models Used
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  

Example (train Logistic Regression):
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
```

### 6. Model Evaluation
- Training vs testing accuracy comparison  
- Checked for overfitting or underfitting  

### Day 3 Requirement
| Train Accuracy – Test Accuracy | ≤ 5%  

---

<!-- =============================================== -->
<!--                      DAY 4                      -->
<!-- =============================================== -->

# Day 4 — Categorical Encoding and One-Hot Encoding

### Why Encoding Is Needed
Machine Learning models cannot work with text data.  
Categorical columns must be converted to numeric form.

---

## Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['column'] = le.fit_transform(df['column'])
```

---

## One-Hot Encoding (Using Pandas)

```python
df_encoded = pd.get_dummies(df, drop_first=True)
```

---

## One-Hot Encoding (Using Scikit-Learn)

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse=False)
encoded = ohe.fit_transform(df[['column']])
```

---

## Handling Encoded Output
- Convert encoded output to a DataFrame  
- Merge encoded columns with original dataset  
- Drop original categorical columns  
- Final ML-ready DataFrame created  

Example (merge encoded output):
```python
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['column']))
df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
df = df.drop(columns=['column'])
```

---

## Updated ML Pipeline After Day 4
1. Encode categorical variables  
2. Perform train–test split  
3. Standardize numerical features  
4. Train ML model  
5. Evaluate the model  
6. Compare results with previous days  

---

<!-- =============================================== -->
<!--                      DAY 5                      -->
<!-- =============================================== -->

# Day 5 — Diamond Price Prediction (Regression Model)

### Overview
Day 5 focuses on building a complete regression model to predict diamond prices using the Diamond Price Dataset.

### Steps Completed

### 1. Data Loading & Cleaning
```python
import pandas as pd

df = pd.read_csv("diamonds.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
df = df.drop_duplicates()
df = df.dropna()
```

### 2. EDA Performed
- Distribution of `price`  
- Relationship between `price` vs `carat`, `depth`, `table`, `x`, `y`, `z`  
- Count analysis of categorical features (`cut`, `color`, `clarity`)

Example (distribution and scatter):
```python
import matplotlib.pyplot as plt

plt.hist(df['price'], bins=50)
plt.title("Price Distribution")
plt.show()

plt.scatter(df['carat'], df['price'])
plt.xlabel("Carat")
plt.ylabel("Price")
plt.show()
```

### 3. Feature Encoding

#### Label Encoding for ordinal categories
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['cut'] = le.fit_transform(df['cut'])
df['color'] = le.fit_transform(df['color'])
df['clarity'] = le.fit_transform(df['clarity'])
```

### 4. Feature Selection
```python
X = df.drop("price", axis=1)
y = df["price"]
```

### 5. Train–Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 6. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 7. Regression Models Used

#### Linear Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)
print("Linear Regression R2:", r2_score(y_test, pred_lr))
print("Linear Regression MAE:", mean_absolute_error(y_test, pred_lr))
```

#### Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)  # using unscaled features for tree-based models is fine
pred_rf = rf.predict(X_test)
print("Random Forest R2:", r2_score(y_test, pred_rf))
print("Random Forest MAE:", mean_absolute_error(y_test, pred_rf))
```

#### Gradient Boosting Regressor
```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
pred_gbr = gbr.predict(X_test)
print("GBR R2:", r2_score(y_test, pred_gbr))
print("GBR MAE:", mean_absolute_error(y_test, pred_gbr))
```

### 8. Model Evaluation Metrics
- R² Score  
- MAE (Mean Absolute Error)  
- MSE (Mean Squared Error)  
- RMSE (Root Mean Squared Error)

Example (compute RMSE):
```python
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, pred_gbr))
print("GBR RMSE:", rmse)
```

### 9. Final Conclusion
- Tree-based ensemble models (Random Forest, Gradient Boosting) usually outperform simple linear models on this dataset due to non-linear relationships and interactions.  
- Select your final model based on R², MAE, RMSE and cross-validation results.

---

<!-- =============================================== -->
<!--                 REPOSITORY STRUCTURE            -->
<!-- =============================================== -->

# Repository Structure

```
ML-Learning-Journey/
│
├── Day1.ipynb
├── Day2.ipynb
├── Day3.ipynb
├── Day4.ipynb
├── Day5_DiamondPricePrediction.ipynb
└── README.md
```

---
