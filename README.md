# Diabetes Prediction: Logistic Regression vs. LDA

## 📌 Project Overview
This project focuses on the early detection of diabetes using the **Pima Indians Diabetes Dataset
The primary objective is to implement and strictly compare two linear classification models to determine which offers superior diagnostic ability:
1.  **Logistic Regression** (Probabilistic Discriminative)
2.  **Linear Discriminant Analysis (LDA)** (Linear Generative)

This study aims to identify the model with the best balance of **Precision, Recall, and F1-Score** while interpreting the coefficients to understand significant medical predictors.

## 🎯 Objectives
* **Preprocessing:** Handle missing values (e.g., zero-value Glucose/BMI) using median imputation and apply **StandardScaler** to normalize feature distributions.
* **Modeling:** Train Logistic Regression and LDA models on the processed dataset.
* **Evaluation:** Compare performance using Accuracy, F1-Score, and **ROC Curve Analysis**
* **Interpretation:** Analyze model coefficients to identify key risk factors like Glucose and BMI

## 🛠️ Technologies Used
* **Python 3.x**
* **Scikit-Learn:** For model training, scaling, and evaluation metrics.
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Matplotlib & Seaborn:** For visualizing ROC curves and correlation heatmaps.

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/officialakash96/diabetes-Prediction-LDA.git
    cd diabetes-Prediction-LDA
    ```

2.  **Create and Activate Virtual Environment**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn python-dotenv
    ```

4.  **Environment Configuration (.env)**
    **Crucial Step:** This project uses a `.env` file to manage the dataset URL securely.
    Create a file named `.env` in the root directory and add the following line:
    ```env
    DATASET_URL=[https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv](https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv)
    ```

## 🚀 How to Run
Execute the main script to load data, train models, and generate the ROC plot:
```bash
python main.py


# ---Other information---:

# The script will output the Accuracy and F1-Scores for both models in the terminal.
# It will save the ROC Comparison Plot as roc_comparison.png in the project folder.

## 📊 Results Summary

# Logistic Regression: Demonstrated robust performance with high interpretability via coefficients.
# LDA: Provided a competitive linear baseline, confirming that the decision boundary between classes is largely linear.
# Key Findings: Feature scaling was critical for both models. Glucose and BMI were identified as the most significant predictors of diabetes risk.