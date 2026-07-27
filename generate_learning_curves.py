import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Toddler Autism dataset July 2018.csv")
df.drop(columns=['Case_No'], errors='ignore', inplace=True)
df.columns = df.columns.str.strip()

# Preprocessing
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

# Drop target leakage columns
leaky_cols = [c for c in df_encoded.columns if 'qchat' in c.lower() or 'score' in c.lower()]
drop_cols = list(set(['Class/ASD Traits'] + leaky_cols))

X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
y = df_encoded['Class/ASD Traits']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models to plot
models_to_plot = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP-ANN": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for ax, (name, model) in zip(axes, models_to_plot.items()):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y, cv=cv, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    ax.set_title(f"Learning Curve — {name}")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Accuracy Score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=300)
print("Learning curves saved to 'learning_curves.png'")
