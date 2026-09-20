# 🛡️ Fraud Anomaly Detection: Credit Card Transactions

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-FF6A00.svg)](https://scikit-learn.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-8A2BE2.svg)](https://git-lfs.github.com/)

## 📌 Project Overview
This project implements a comprehensive Data Mining pipeline to tackle the credit card fraud detection problem. In the context of cybersecurity, detecting fraud is fundamentally an **Anomaly Detection** task, requiring the system to identify sophisticated malicious behaviors hidden among hundreds of thousands of legitimate transactions.

The core challenge of this project is handling **Highly Imbalanced Data**, where the fraudulent class accounts for only **0.1727%** of the total transactions.

## 📊 Dataset
*   **Data Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
*   **Shape:** 284,807 rows x 31 columns.
*   **Class Distribution:** `0` (Valid Transactions - 99.8273%) | `1` (Fraudulent Transactions - 0.1727%).
*   **Storage:** To optimize bandwidth limits, the original dataset has been compressed into `creditcard.zip` and is managed directly on GitHub using **Git LFS (Large File Storage)**.

## 🛠️ Methodology & Technologies
The project is deployed through 4 in-depth analysis phases:

### 1. Exploratory Data Analysis (EDA) & Preprocessing
* Evaluate data integrity, handle missing values, and analyze the distribution of continuous variables.
* Standardize data using `StandardScaler` to ensure the reliability of distance- and variance-based algorithms.

### 2. Dimensionality Reduction & Visualization
* Apply **PCA** to project and evaluate the overall linear structure of the data space.
* Implement **t-SNE** (with an optimized sampling strategy) to explore non-linear relationships and preserve local neighborhood structures, effectively visualizing the boundaries between valid and fraudulent transactions in a 2D space.

### 3. Supervised Learning
* Deploy 3 representative algorithms: **Naive Bayes**, **AdaBoost**, and **Random Forest**.
* **Optimization Metric:** Utilize the **Macro F1-Score** instead of Accuracy to avoid the "accuracy paradox" on imbalanced data, forcing the model to accurately detect the minority class (fraud).
* Perform hyperparameter tuning using `GridSearchCV` and ensure objective evaluation via `Stratified 10-Fold Cross-Validation`.

### 4. Unsupervised Learning
* Approach the problem from a "blind threat hunting" perspective using **K-Means** and **DBSCAN**.
* Plot the *k-distance* graph to determine the optimal `eps` radius for the DBSCAN algorithm.

## 🏆 Key Results
The 10-Fold CV results demonstrate that the **Random Forest** model achieves outstanding performance in controlling both the false negative (missed frauds) and false positive rates:

| Model | Macro F1-Score | Weighted F1 | Fraud F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.8847** | **0.9992** | **0.7699** | 0.9992 |
| **AdaBoost** | 0.8531 | 0.9990 | 0.7067 | 0.9991 |
| **Naive Bayes** | 0.5510 | 0.9872 | 0.1133 | 0.9777 |

## 🚀 Installation & Usage

**Step 1: Clone the repository and pull LFS data**
```bash
git clone [https://github.com/hien-cybers/Fraud-Anomaly-Detection.git](https://github.com/hien-cybers/Fraud-Anomaly-Detection.git)
cd Fraud-Anomaly-Detection
git lfs pull
**Step 2: Set up the environment**
```bash
pip install -r requirements.txt
