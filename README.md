# 🤖 ML-LAB — Machine Learning Laboratory

> A collection of hands-on Machine Learning notebooks covering core algorithms, EDA workflows, and deep learning, built as part of my AI/ML coursework and self-study.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-yellow?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-red?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---

## 📋 Table of Contents

- [About](#-about)
- [Notebooks](#-notebooks)
- [Tech Stack](#-tech-stack)
- [File Renaming Guide](#-file-renaming-guide)
- [Getting Started](#-getting-started)

---

## 📌 About

This repository contains Jupyter Notebooks I built while exploring Machine Learning concepts from scratch. Each notebook is a self-contained project covering a specific ML algorithm or data science workflow — from Python fundamentals and exploratory data analysis all the way through neural networks and support vector machines.

---

## 📓 Notebooks

### 1. `01_Python_NumPy_Pandas_Basics.ipynb`
> **Topic:** Python Fundamentals, NumPy & Pandas

This notebook serves as a foundation for data science work, covering Python control structures, functions, and list comprehensions. It demonstrates numerical operations using NumPy (mean, std, sort) and introduces data manipulation using Pandas DataFrames. A student score dataset and a sample email CSV file are used to practice filtering and descriptive statistics. Key results include computed means (5.5 and 18.6), identifying high-scoring students (score ≥ 70), and generating prime/Fibonacci sequences.

**Tools:** `Python`, `NumPy`, `Pandas`

---

### 2. `02_Decision_Tree_Housing_Price_Prediction.ipynb`
> **Topic:** Decision Tree Regression — Iowa Housing Dataset

This notebook tackles housing price prediction using Scikit-Learn's `DecisionTreeRegressor`, following a Kaggle-style tutorial structure. Features such as `LotArea`, `YearBuilt`, `1stFlrSF`, `2ndFlrSF`, `FullBath`, `BedroomAbvGr`, and `TotRmsAbvGrd` are selected for training. Pandas is used for dataset inspection and summary statistics. The fitted model successfully generates predictions matching benchmark values (e.g., $208,500 and $181,500).

**Dataset:** Iowa Housing Dataset — 1,460 rows  
**Tools:** `Pandas`, `scikit-learn (DecisionTreeRegressor)`

---

### 3. `03_Pulsar_Star_ANN_Classification.ipynb`
> **Topic:** Artificial Neural Network (ANN) — Pulsar Star Binary Classification

A deep learning binary classification notebook that identifies Pulsar Stars from radio telescope observations using the HTRU2 dataset (17,898 samples, 9 features). A Keras/TensorFlow ANN is trained and evaluated using 10-Fold Cross-Validation. Data is normalized with `StandardScaler` before training. The model achieves a **cross-validation accuracy of 97.70% ± 0.35%**, a **test accuracy of 97.99%**, and a **Macro F1-Score of 0.94**.

**Dataset:** HTRU2 Pulsar Stars — 17,898 rows  
**Tools:** `TensorFlow/Keras`, `scikit-learn (KFold, StandardScaler)`, `Pandas`, `Matplotlib`  
**Results:** ✅ Test Accuracy: **97.99%** | Macro F1: **0.94**

---

### 4. `04_California_Housing_EDA_Mid_Exam.ipynb`
> **Topic:** Exploratory Data Analysis (EDA) — California Housing Dataset

A mid-exam EDA notebook exploring the California Housing Dataset (20,640 rows, 10 features). Pandas is used to inspect data types, compute descriptive statistics, and identify missing values (207 missing entries in `total_bedrooms`). Seaborn and Matplotlib are used to visualize the distribution of house values and the geographical relationship between median income and median house value via scatter plots and histograms.

**Dataset:** California Housing — 20,640 rows  
**Tools:** `Pandas`, `Seaborn`, `Matplotlib`  
**Results:** 📊 Found 207 missing values; strong positive correlation between median income and house values

---

### 5. `05_Email_Spam_Classification_Naive_Bayes.ipynb`
> **Topic:** NLP Text Classification — Spam vs. Ham (Naive Bayes)

This notebook builds an email/SMS spam detection classifier using Natural Language Processing. Raw text messages from 5,572 samples are converted into feature vectors using `CountVectorizer` (bag-of-words), and a `MultinomialNB` model is trained for binary classification. Distribution plots visualize class imbalance and message length differences between spam and ham. The model is evaluated using accuracy, precision, recall, and a confusion matrix.

**Dataset:** Email/SMS Spam Dataset — 5,572 messages  
**Tools:** `scikit-learn (CountVectorizer, MultinomialNB)`, `Seaborn`, `Matplotlib`  
**Results:** ✅ Spam classifier evaluated with precision, recall & confusion matrix metrics

---

### 6. `06_Linear_Regression_USA_Housing.ipynb`
> **Topic:** Multiple Linear Regression — USA Housing Price Prediction

Implements a Multiple Linear Regression model using the USA Housing Dataset (5,000 rows) to predict residential property prices. The pipeline includes data loading, Seaborn-based null value heatmaps for data quality checks, and a train/test split before fitting Scikit-Learn's `LinearRegression`. Features used include average area income, house age, number of rooms, number of bedrooms, and area population.

**Dataset:** USA Housing — 5,000 rows  
**Tools:** `Pandas`, `Seaborn`, `scikit-learn (LinearRegression, train_test_split)`  
**Results:** 📈 Linear Regression model trained and evaluated on 5 continuous housing features

---

### 7. `07_Logistic_Regression_Titanic_Survival.ipynb`
> **Topic:** Binary Classification — Titanic Survival Prediction (Logistic Regression)

Predicts Titanic passenger survival using Logistic Regression on the classic Titanic dataset (891 rows). Data preprocessing includes imputing missing `Age` values based on passenger class (`Pclass`) using median statistics derived from Seaborn box plots. Categorical features are dropped and the final `LogisticRegression` model is trained on demographic, ticket, and class-based features to classify survival outcomes.

**Dataset:** Titanic Passenger Dataset — 891 rows  
**Tools:** `Pandas`, `Seaborn`, `scikit-learn (LogisticRegression)`  
**Results:** 🚢 Logistic Regression survival classifier trained with data cleaning & imputation pipeline

---

### 8. `08_Random_Forest_Tutorial.ipynb`
> **Topic:** Ensemble Learning — Decision Trees & Random Forest

A conceptual tutorial notebook demonstrating how Decision Trees work before scaling up to Random Forests. A synthetic 2D binary classification dataset is generated and used to train a `DecisionTreeClassifier` (Gini impurity, max depth 3), producing a 9-node tree with **100% training accuracy** on the sample. The tree is exported and visualized using Graphviz (`tree.png`). The notebook then demonstrates how an ensemble of trees forms a `RandomForestClassifier`.

**Dataset:** Synthetic 2D binary dataset  
**Tools:** `scikit-learn (DecisionTreeClassifier, RandomForestClassifier)`, `Graphviz`, `Matplotlib`  
**Results:** 🌲 Decision tree: 9 nodes, max depth 3, 100% training accuracy on synthetic data

---

### 9. `09_Support_Vector_Machine_Heart_Disease.ipynb`
> **Topic:** Binary Classification — Heart Disease Detection (SVM)

Applies a Support Vector Machine (SVM) with a linear kernel to classify patients as having heart disease or not using the Heart Disease dataset (303 rows, 14 features). Features `trestbps` (resting blood pressure) and `chol` (cholesterol) are used to create a 2D decision boundary visualization with highlighted support vectors using `np.meshgrid` and `plt.contourf`. Model performance is evaluated using a confusion matrix and classification report.

**Dataset:** Heart Disease Dataset — 303 rows, 14 features  
**Tools:** `scikit-learn (SVC, StandardScaler)`, `NumPy (meshgrid)`, `Matplotlib`  
**Results:** ❤️ Linear SVM decision boundary visualized with support vectors highlighted

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| `Python 3.8+` | Core programming language |
| `NumPy` | Numerical computation |
| `Pandas` | Data manipulation & EDA |
| `Matplotlib` | Data visualization |
| `Seaborn` | Statistical data visualization |
| `scikit-learn` | ML algorithms, preprocessing & evaluation |
| `TensorFlow / Keras` | Deep learning (ANN) |
| `Graphviz` | Decision tree visualization |

---

## 🗂 File Renaming Guide

> The notebooks have been renamed from their original lab submission names to descriptive, content-based names for better readability.

| # | Original Filename | New Descriptive Filename |
|---|---|---|
| 1 | `Affan_ML_Lab_1.ipynb` | `01_Python_NumPy_Pandas_Basics.ipynb` |
| 2 | `Affan_ML_Lab_2.ipynb` | `02_Decision_Tree_Housing_Price_Prediction.ipynb` |
| 3 | `Affan_Zulfiqar_B22F0144AI050_Lab_06.ipynb` | `03_Pulsar_Star_ANN_Classification.ipynb` |
| 4 | `Affan_Zulfiqar_B22F0144AI050_MID_EXAM.ipynb` | `04_California_Housing_EDA_Mid_Exam.ipynb` |
| 5 | `Affan_Zulfiqar_B22F0144AI050_ML_Lab_10 (1).ipynb` | `05_Email_Spam_Classification_Naive_Bayes.ipynb` |
| 6 | `Liner_Regression_ML_Lab_5.ipynb` | `06_Linear_Regression_USA_Housing.ipynb` |
| 7 | `Logistic_Regression_ml_Lab_6.ipynb` | `07_Logistic_Regression_Titanic_Survival.ipynb` |
| 8 | `Random_Forest_Tutorial.ipynb` | `08_Random_Forest_Tutorial.ipynb` |
| 9 | `Support_Vector_Machine.ipynb` | `09_Support_Vector_Machine_Heart_Disease.ipynb` |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/AffanZulfiqar/ML-LAB.git
cd ML-LAB

# Install required dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter graphviz

# Launch Jupyter Notebook
jupyter notebook
```

Open any `.ipynb` file in the browser to explore the notebook.

<p align="center">⭐ If you found this helpful, consider starring the repo!</p>
