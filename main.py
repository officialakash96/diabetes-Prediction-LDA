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