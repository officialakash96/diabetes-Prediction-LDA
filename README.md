# Diabetes Prediction: Logistic Regression vs. LDA

## 📌 Project Overview
[cite_start]This project focuses on the early detection of diabetes using the **Pima Indians Diabetes Dataset**[cite: 15, 103]. The primary objective is to implement and strictly compare two linear classification models to determine which offers superior diagnostic ability:
1.  [cite_start]**Logistic Regression** (Probabilistic Discriminative) [cite: 101]
2.  [cite_start]**Linear Discriminant Analysis (LDA)** (Linear Generative) [cite: 102]

[cite_start]This study aims to identify the model with the best balance of **Precision, Recall, and F1-Score** while interpreting the coefficients to understand significant medical predictors[cite: 13, 112].

## 🎯 Objectives
* [cite_start]**Preprocessing:** Handle missing values (e.g., zero-value Glucose/BMI) using median imputation and apply **StandardScaler** to normalize feature distributions[cite: 35, 37].
* [cite_start]**Modeling:** Train Logistic Regression and LDA models on the processed dataset[cite: 40].
* [cite_start]**Evaluation:** Compare performance using Accuracy, F1-Score, and **ROC Curve Analysis**[cite: 46, 111].
* [cite_start]**Interpretation:** Analyze model coefficients to identify key risk factors like Glucose and BMI[cite: 66, 112].

## 🛠️ Technologies Used
* **Python 3.x**
* [cite_start]**Scikit-Learn:** For model training, scaling, and evaluation metrics[cite: 105].
* [cite_start]**Pandas & NumPy:** For data manipulation and numerical operations[cite: 105].
* [cite_start]**Matplotlib & Seaborn:** For visualizing ROC curves and correlation heatmaps[cite: 105].

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