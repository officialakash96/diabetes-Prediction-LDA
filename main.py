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