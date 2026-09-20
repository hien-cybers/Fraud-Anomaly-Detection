# 🛡️ Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-FF6A00.svg)](https://scikit-learn.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-8A2BE2.svg)](https://git-lfs.github.com/)

## 📌 Project Overview

This project implements a comprehensive **Data Mining and Machine Learning pipeline** for detecting fraudulent credit card transactions.

Credit card fraud detection is a challenging problem because fraudulent transactions represent only a very small proportion of all transactions. Therefore, the dataset is **highly imbalanced**, making conventional accuracy-based evaluation insufficient.

The project combines **Exploratory Data Analysis (EDA), preprocessing, dimensionality reduction, supervised learning, and unsupervised learning** to analyze and detect fraudulent transactions.

## 📊 Dataset

- **Data Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Dataset Size:** 284,807 transactions × 31 columns
- **Features:** 30 numerical features + 1 target/class column
- **Class Distribution:**
  - `0` – Legitimate transactions: **99.8273%**
  - `1` – Fraudulent transactions: **0.1727%**
- **Storage:** The dataset is compressed into `creditcard.zip` and managed using **Git LFS (Large File Storage)**.

## 🗂️ Project Structure

```text
Fraud-Anomaly-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── creditcard.zip
├── requirements.txt
└── README.md
```

## 🛠️ Methodology

The project is divided into four main analysis phases.

### 1. 🔍 Exploratory Data Analysis & Preprocessing

- Inspect the dataset structure and data types.
- Check data integrity and missing values.
- Analyze the distribution of numerical features.
- Examine the class imbalance between legitimate and fraudulent transactions.
- Apply `StandardScaler` to standardize numerical features.
- Prepare the dataset for subsequent machine learning algorithms.

### 2. 📉 Dimensionality Reduction & Visualization

#### PCA – Principal Component Analysis

PCA is applied to reduce the dimensionality of the dataset and analyze its overall linear structure.

It helps:

- Reduce the number of dimensions.
- Preserve important variance in the data.
- Visualize the general structure of the dataset.

#### t-SNE – t-Distributed Stochastic Neighbor Embedding

t-SNE is used to explore non-linear relationships and local structures within the data.

An optimized sampling strategy is applied because the original dataset contains more than 280,000 transactions.

The resulting 2D visualization helps investigate whether legitimate and fraudulent transactions form distinguishable patterns.

### 3. 🤖 Supervised Learning

Three representative classification algorithms are implemented:

- **Naive Bayes**
- **AdaBoost**
- **Random Forest**

Because the dataset is highly imbalanced, **Macro F1-Score** is emphasized instead of relying solely on Accuracy.

Macro F1 calculates the F1-score independently for each class and then averages the results, giving the minority fraud class greater influence on the evaluation.

The project also applies:

- `GridSearchCV` for hyperparameter tuning.
- **Stratified 10-Fold Cross-Validation** to maintain class proportions across folds.
- Multiple evaluation metrics including Macro F1, Weighted F1, Fraud F1, and Accuracy.

## 🏆 Model Performance

The reported **Stratified 10-Fold Cross-Validation** results are:

| Model | Macro F1-Score | Weighted F1 | Fraud F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.8847** | **0.9992** | **0.7699** | **0.9992** |
| **AdaBoost** | **0.8531** | **0.9990** | **0.7067** | **0.9991** |
| **Naive Bayes** | **0.5510** | **0.9872** | **0.1133** | **0.9777** |

The results show that **Random Forest achieved the highest reported Macro F1-Score and Fraud F1-Score among the three supervised models**.

> **Note:** Accuracy and Weighted F1 are very high partly because legitimate transactions dominate the dataset. Therefore, Macro F1 and Fraud F1 are particularly useful for assessing fraud detection performance.

## 4. 🔎 Unsupervised Learning

The project also approaches fraud detection from an **unsupervised anomaly detection** perspective.

Two clustering algorithms are explored:

### K-Means

K-Means is used to group transactions into clusters based on their feature similarity.

This allows the project to investigate whether fraudulent transactions can be separated from legitimate transactions without using class labels during clustering.

### DBSCAN

DBSCAN is a density-based clustering algorithm that can identify dense regions and outliers.

The project uses a **k-distance graph** to help determine a suitable `eps` value for DBSCAN.

DBSCAN is particularly interesting for anomaly detection because points that do not belong to dense clusters may be treated as potential outliers.

## 🧠 Why Use Both Supervised and Unsupervised Learning?

Using both approaches provides two complementary perspectives:

| Approach | Main Idea | Purpose |
| :--- | :--- | :--- |
| **Supervised Learning** | Learn from labeled transactions | Directly classify legitimate and fraudulent transactions |
| **Unsupervised Learning** | Discover patterns without labels | Explore clusters and identify potential anomalies |

Supervised learning focuses on classification performance when labeled historical data is available, while unsupervised learning provides an additional perspective for discovering unusual transaction patterns.

## 🧰 Technologies Used

- **Python 3.x**
- **Jupyter Notebook**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Git**
- **Git LFS**

## 🚀 Installation & Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/hien-cybers/Fraud-Anomaly-Detection.git
cd Fraud-Anomaly-Detection
```

### Step 2: Pull the Dataset Using Git LFS

Make sure Git LFS is installed, then run:

```bash
git lfs install
git lfs pull
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Project

Open:

```text
Credit_Card_Fraud_Detection.ipynb
```

using **Jupyter Notebook**, **JupyterLab**, or **Visual Studio Code**.

The notebook reads the dataset from the downloaded `creditcard.zip` file.

## 👥 Development Team – Group 3

| Member | Responsibility |
| :--- | :--- |
| **Nguyen Duc Hien** | Project Management & Data Exploration / Preprocessing |
| **Hoang Thi Ngoc Tram** | Dimensionality Reduction: PCA / t-SNE & 2D Visualization |
| **Tran Truong Thinh** | Classification Pipeline & Hyperparameter Tuning |
| **Nguyen Luong Vinh Chi** | Unsupervised Learning: K-Means / DBSCAN |
| **Nguyen Anh Vu** | Stratified 10-Fold Cross-Validation Evaluation & Final Report |

**Institution:** University of Transport and Communications (UTH)

## 📌 Conclusion

This project demonstrates a complete data mining workflow for credit card fraud detection, from data exploration and preprocessing to dimensionality reduction, supervised classification, and unsupervised clustering.

The results highlight the importance of using appropriate evaluation metrics for highly imbalanced datasets. In particular, **Macro F1-Score and Fraud F1-Score** provide more informative insights into minority-class detection than Accuracy alone.

The combination of supervised and unsupervised techniques also provides a broader perspective on both known fraud patterns and potential anomalies in transaction data.

## 📄 License

This project is developed for **educational and academic purposes**.

The dataset is provided by the original Kaggle source and is subject to its respective terms of use.

## 🙏 Acknowledgements

- **Kaggle** – for providing the Credit Card Fraud Detection dataset.
- **Scikit-learn** – for the machine learning algorithms and evaluation tools.
- **Jupyter Project** – for the interactive notebook environment.
