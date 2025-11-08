# Multi-class-Classification-Logistic-Regression

The notebook is a complete **hands-on lab** for performing **multi-class classification** using a real-world dataset related to **obesity levels**. Here's a structured breakdown of its content:

---

### 🔍 **Main Purpose**
To demonstrate how to implement **multi-class classification strategies** in Python using scikit-learn on a labeled dataset about obesity.

---

### 📂 **Dataset Used**
- File: `https://www.kaggle.com/datasets/ezzaldeenesmail/obesitydataset-raw-and-data-sinthetic`
- Loaded with: `pandas.read_csv()`
- Target column: `NObeyesdad` (which represents obesity categories)

---

### 📌 **Notebook Structure**

#### 1. **Setup and Imports**
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
```
These are the main tools used for:
- Data processing
- Visualization
- Training and evaluating classification models

---

#### 2. **Data Loading & Exploration**
- Load the dataset into a DataFrame
- Display first few records (`data.head()`)
- Visualize target variable distribution:
```python
sns.countplot(y='NObeyesdad', data=data)
```

---

#### 3. **Preprocessing**
- Apply **One-Hot Encoding** for categorical variables
- Use **StandardScaler** to normalize numerical features
- Split data into **training** and **testing sets**

---

#### 4. **Modeling**
- Implements a **Logistic Regression** model using:
  - **One-vs-Rest (OvR)** strategy
  - **One-vs-One (OvO)** strategy

```python
# Example:
model = LogisticRegression()
ovo = OneVsOneClassifier(model)
ovo.fit(X_train, y_train)
```

---

#### 5. **Evaluation**
- Evaluate model performance using **accuracy_score**
- Compare results from OvO and OvR classifiers

---

#### 6. **Conclusion & Summary**
- Analyzes which strategy works better for the dataset
- Shows final accuracy and insights

---

Would you like a copy of the dataset used, a visual chart of the workflow, or a restructured version of the notebook for learning purposes?
