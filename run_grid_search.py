import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
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

leaky_cols = [c for c in df_encoded.columns if 'qchat' in c.lower() or 'score' in c.lower()]
drop_cols = list(set(['Class/ASD Traits'] + leaky_cols))

X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
y = df_encoded['Class/ASD Traits']

X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_raw, y_train_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)

# Define Hyperparameter Search Grids
param_grids = {
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5, 10, 15]
        }
    ),
    "SVM (RBF)": (
        SVC(kernel='rbf', probability=True, random_state=42),
        {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            'max_depth': [3, 5, 10],
            'min_samples_split': [10, 20],
            'ccp_alpha': [0.0, 0.005, 0.01]
        }
    ),
    "MLP-ANN": (
        MLPClassifier(max_iter=200, early_stopping=True, random_state=42),
        {
            'hidden_layer_sizes': [(32, 16), (64, 32), (32,)],
            'alpha': [0.0001, 0.001, 0.01]
        }
    )
}

tuning_summary = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running GridSearchCV Hyperparameter Tuning...")

for name, (model, p_grid) in param_grids.items():
    print(f"\nTuning {name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=p_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train_res)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best CV ROC-AUC: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Store string representation of params
    tuning_summary[name] = {
        "best_cv_roc_auc": round(float(best_score), 4),
        "best_params": {k: (str(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
        "search_grid": {k: [str(x) if isinstance(x, tuple) else x for x in v] for k, v in p_grid.items()}
    }

with open("hyperparameter_tuning_results.json", "w") as f:
    json.dump(tuning_summary, f, indent=4)

print("\nHyperparameter tuning complete. Results saved to 'hyperparameter_tuning_results.json'")
