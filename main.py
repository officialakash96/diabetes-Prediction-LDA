# Feature: Environment Setup and Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, f1_score

# Load environment variables
load_dotenv()

print("Libraries loaded and environment configured.")

# Feature: Data Loading Pipeline
def load_data():
    url = os.getenv("DATASET_URL")
    if not url:
        raise ValueError("DATASET_URL not found in .env file")
    
    print(f"Loading data from: {url}")
    df = pd.read_csv(url)
    return df

# Execute Load
df = load_data()
print(f"Data Loaded. Shape: {df.shape}")

# Feature: Preprocessing and Scaling
def preprocess_data(df):
    # Replace zeros with median
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, df[col].median())
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale (Critical for LDA/LogReg)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
print("Data preprocessed and scaled.")


# Feature: Model Training Implementation
def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'LDA': LinearDiscriminantAnalysis()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained.")
        
    return models

trained_models = train_models(X_train, y_train)

# Feature: Evaluation and ROC Plotting
def evaluate_models(models, X_test, y_test):
    results = {}
    plt.figure(figsize=(10, 6))
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: LogReg vs LDA')
    plt.legend()
    plt.savefig('roc_comparison.png')
    print("ROC Curve saved as 'roc_comparison.png'")

evaluate_models(trained_models, X_test, y_test)