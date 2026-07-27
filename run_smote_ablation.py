import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("Toddler Autism dataset July 2018.csv")
df.drop(columns=['Case_No'], errors='ignore', inplace=True)
df.columns = df.columns.str.strip()

df.drop_duplicates(inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

for col in df_encoded.columns:
    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
df_encoded.fillna(df_encoded.mean(), inplace=True)

# Drop target leakage columns (Qchat-10-Score)
leaky_cols = [c for c in df_encoded.columns if 'qchat' in c.lower() or 'score' in c.lower()]
drop_cols = list(set(['Class/ASD Traits'] + leaky_cols))

X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
y = df_encoded['Class/ASD Traits']

# Train-Test Split (fixed seed)
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Setup SMOTE vs Non-SMOTE datasets
scaler = StandardScaler()

# Pipeline A: Without SMOTE
X_train_no_smote = scaler.fit_transform(X_train_raw)
y_train_no_smote = y_train_raw.values

# Pipeline B: With SMOTE
smote = SMOTE(random_state=42)
X_train_smote_raw, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)
scaler_smote = StandardScaler()
X_train_smote = scaler_smote.fit_transform(X_train_smote_raw)

X_test_scaled_no = scaler.transform(X_test_raw)
X_test_scaled_smote = scaler_smote.transform(X_test_raw)

# Model definitions
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=51),
        "SVM (RBF)": SVC(kernel='rbf', C=0.1, probability=True),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.7),
        "MLP-ANN": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, early_stopping=True, random_state=42)
    }

ablation_results = []

print("Running SMOTE Ablation Study...")

for name, model in get_models().items():
    # Fit WITHOUT SMOTE
    model_no = get_models()[name]
    model_no.fit(X_train_no_smote, y_train_no_smote)
    pred_no = model_no.predict(X_test_scaled_no)
    prob_no = model_no.predict_proba(X_test_scaled_no)[:, 1]
    
    rec_no = recall_score(y_test, pred_no)
    spec_no = confusion_matrix(y_test, pred_no).ravel()[0] / (confusion_matrix(y_test, pred_no).ravel()[0] + confusion_matrix(y_test, pred_no).ravel()[1])
    f1_no = f1_score(y_test, pred_no)
    auc_no = roc_auc_score(y_test, prob_no)

    # Fit WITH SMOTE
    model_smote = get_models()[name]
    model_smote.fit(X_train_smote, y_train_smote)
    pred_smote = model_smote.predict(X_test_scaled_smote)
    prob_smote = model_smote.predict_proba(X_test_scaled_smote)[:, 1]
    
    rec_smote = recall_score(y_test, pred_smote)
    spec_smote = confusion_matrix(y_test, pred_smote).ravel()[0] / (confusion_matrix(y_test, pred_smote).ravel()[0] + confusion_matrix(y_test, pred_smote).ravel()[1])
    f1_smote = f1_score(y_test, pred_smote)
    auc_smote = roc_auc_score(y_test, prob_smote)

    ablation_results.append({
        "Model": name,
        "No-SMOTE Recall": rec_no,
        "SMOTE Recall": rec_smote,
        "Recall Diff": rec_smote - rec_no,
        "No-SMOTE F1": f1_no,
        "SMOTE F1": f1_smote,
        "F1 Diff": f1_smote - f1_no,
        "No-SMOTE ROC-AUC": auc_no,
        "SMOTE ROC-AUC": auc_smote
    })

ablation_df = pd.DataFrame(ablation_results)
print("\nSMOTE Ablation Study Results:")
print(ablation_df.to_string(index=False))

ablation_df.to_csv("smote_ablation_results.csv", index=False)
print("\nSaved SMOTE ablation results to 'smote_ablation_results.csv'")
