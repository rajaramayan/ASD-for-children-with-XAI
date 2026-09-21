# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, log_loss)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

MODELS_DIR = "models"

# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("Toddler Autism dataset July 2018.csv")
df.drop(columns=['Case_No'], errors='ignore', inplace=True)
df.columns = df.columns.str.strip()

print("Shape:", df.shape)
print(df.info())
print(df.describe())

# ==========================================
# 3. PREPROCESSING
# ==========================================
df.drop_duplicates(inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Encode categorical variables (keep per-column encoders)
le_dict = {}
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    le_dict[col] = le

for col in df_encoded.columns:
    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
df_encoded.fillna(df_encoded.mean(), inplace=True)

# Split features and target — EXCLUDE TARGET LEAKAGE COLUMNS (Qchat-10-Score)
leaky_cols = [c for c in df_encoded.columns if 'qchat' in c.lower() or 'score' in c.lower()]
drop_cols = list(set(['Class/ASD Traits'] + leaky_cols))

X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
y = df_encoded['Class/ASD Traits']
feature_names = X.columns.tolist()

print(f"\nTarget leakage prevention applied: Removed {leaky_cols}")
print(f"Features selected for training ({len(feature_names)} features): {feature_names}")

# Update numeric and categorical column lists for metadata without target & dropped cols
numeric_cols = [c for c in numeric_cols if c in feature_names]
categorical_cols = [c for c in categorical_cols if c in feature_names]

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 5. HANDLE IMBALANCE (SMOTE)
# ==========================================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ==========================================
# 6. FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 7. DEFINE MODELS (same config as Streamlit app)
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=20, min_samples_leaf=10,
        ccp_alpha=0.005, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=15,
        min_samples_leaf=6, max_features='sqrt',
        min_impurity_decrease=0.001, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=51, weights='uniform', metric='minkowski', p=1),
    "SVM (Poly)": SVC(kernel='poly', degree=2, C=0.1, gamma='scale', probability=True),
    "SVM (RBF)": SVC(kernel='rbf', C=0.1, gamma='scale', probability=True),
    "Naive Bayes": GaussianNB(var_smoothing=1e-8),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=0.7)
}

# ==========================================
# 8. TRAIN + EVALUATE CLASSICAL MODELS
# ==========================================
results = []
roc_data = []
trained_models = {}

columns = ["Model", "Train Accuracy", "Accuracy", "Gap",
           "Precision", "Recall", "Specificity", "F1 Score", "ROC-AUC", "Log Loss"]

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_prob)
    logloss  = log_loss(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    gap = train_acc - acc

    results.append([name, train_acc, acc, gap, prec, rec, specificity, f1, roc_auc, logloss])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data.append((name, fpr, tpr, roc_auc))

# ==========================================
# 9. ANN MODEL (MLPClassifier — same as Streamlit)
# ==========================================
print("\nTraining ANN (MLPClassifier)...")
ann = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
ann.fit(X_train_scaled, y_train)

train_acc_ann = accuracy_score(y_train, ann.predict(X_train_scaled))
y_prob_ann = ann.predict_proba(X_test_scaled)[:, 1]
y_pred_ann = ann.predict(X_test_scaled)

acc  = accuracy_score(y_test, y_pred_ann)
prec = precision_score(y_test, y_pred_ann)
rec  = recall_score(y_test, y_pred_ann)
f1   = f1_score(y_test, y_pred_ann)
roc_auc  = roc_auc_score(y_test, y_prob_ann)
logloss  = log_loss(y_test, y_prob_ann)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ann).ravel()
specificity = tn / (tn + fp)
gap_ann = train_acc_ann - acc

results.append(["ANN", train_acc_ann, acc, gap_ann, prec, rec, specificity, f1, roc_auc, logloss])

fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)
roc_data.append(("ANN", fpr_ann, tpr_ann, roc_auc))

# ==========================================
# 10. CREATE RESULTS DATAFRAME
# ==========================================
results_df = pd.DataFrame(results, columns=columns)
results_df_sorted = results_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

print("\nFinal Model Comparison (Sorted by Test ROC-AUC):\n")
print(results_df_sorted.to_string(index=False))

# ==========================================
# 11. 10-FOLD STRATIFIED CROSS-VALIDATION & OVERFITTING DIAGNOSTICS
# ==========================================
print("\nRunning 10-Fold Stratified Cross-Validation...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_summary = []

all_eval_models = dict(models)
all_eval_models["ANN"] = ann

for name, model in all_eval_models.items():
    scores = cross_validate(model, X_train_scaled, y_train, cv=skf, scoring=['accuracy', 'f1', 'roc_auc'], return_train_score=True)
    train_acc_mean = np.mean(scores['train_accuracy'])
    val_acc_mean = np.mean(scores['test_accuracy'])
    val_acc_std = np.std(scores['test_accuracy'])
    val_f1_mean = np.mean(scores['test_f1'])
    val_auc_mean = np.mean(scores['test_roc_auc'])
    cv_gap = train_acc_mean - val_acc_mean
    
    cv_summary.append({
        "Model": name,
        "CV Train Acc": train_acc_mean,
        "CV Val Acc Mean": val_acc_mean,
        "CV Val Acc Std": val_acc_std,
        "CV F1": val_f1_mean,
        "CV ROC-AUC": val_auc_mean,
        "CV Gap": cv_gap
    })

cv_df = pd.DataFrame(cv_summary).sort_values(by="CV ROC-AUC", ascending=False)
print("\n10-Fold Stratified Cross-Validation Results:")
print(cv_df.to_string(index=False))

# ==========================================
# 11b. STATISTICAL SIGNIFICANCE TESTING (McNemar's Test)
# ==========================================
print("\nStatistical Significance Testing (McNemar's Test vs Best Classical Model):")
top_classical = results_df_sorted[results_df_sorted["Model"] != "ANN"].iloc[0]["Model"]
top_classical_pred = trained_models[top_classical].predict(X_test_scaled)

b = np.sum((y_pred_ann == y_test) & (top_classical_pred != y_test))  # ANN correct, Classical wrong
c = np.sum((y_pred_ann != y_test) & (top_classical_pred == y_test))  # ANN wrong, Classical correct

n_disc = b + c
if n_disc > 0:
    p_value = 2 * binom.cdf(min(b, c), n_disc, 0.5)
    p_value = min(1.0, p_value)
else:
    p_value = 1.0

print(f"Comparison Pair: ANN vs {top_classical}")
print(f"Disagreements: ANN correct/Classical wrong = {b}, ANN wrong/Classical correct = {c}")
print(f"McNemar exact p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"Conclusion: Performance difference between ANN and {top_classical} IS statistically significant (p < 0.05).")
else:
    print(f"Conclusion: Performance difference between ANN and {top_classical} IS NOT statistically significant (p >= 0.05).")

# ==========================================
# 12. ROC CURVE (ALL MODELS)
# ==========================================
line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-']
colors = plt.cm.tab10(np.linspace(0, 0.9, 9))
fpr_grid = np.linspace(0, 1, 300)

plt.figure(figsize=(10, 8))
for i, (name, fpr, tpr, auc) in enumerate(roc_data):
    _, first_idx = np.unique(fpr, return_index=True)
    tpr_smooth = np.interp(fpr_grid, fpr[first_idx], tpr[first_idx])
    plt.plot(fpr_grid, tpr_smooth,
             label=f"{name} (AUC={auc:.4f})",
             linewidth=2,
             linestyle=line_styles[i % len(line_styles)],
             color=colors[i])
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (Target-Leakage-Free Models)")
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()

# ==========================================
# 13. BAR PLOT — PERFORMANCE COMPARISON
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))
results_df_sorted.set_index("Model")[["Accuracy", "F1 Score", "ROC-AUC"]].plot(kind='bar', ax=ax)
ax.set_title("Model Performance Comparison (Target-Leakage-Free Models)")
ax.set_ylabel("Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("model_comparison_bar.png", dpi=300)
plt.close()

# ==========================================
# 14. CONFUSION MATRIX (BEST MODEL)
# ==========================================
best_model_name = results_df_sorted.iloc[0]["Model"]

if best_model_name == "ANN":
    y_pred_best = y_pred_ann
else:
    y_pred_best = trained_models[best_model_name].predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix_best.png", dpi=300)
plt.close()

# ==========================================
# 15. FINAL BEST MODEL SUMMARY
# ==========================================
print("\nBEST MODEL:")
print(results_df_sorted.iloc[0])

# ==========================================
# 15.5 EXPLAINABLE AI (XAI) WITH SHAP
# ==========================================
print("\nGenerating SHAP Explainable AI (XAI) Analysis...")
try:
    import shap
    
    # Let's generate SHAP explanations for the best classical model. 
    # If ANN is the best, we'll explain the top classical model (or Random Forest) 
    # since tree/linear models are more straightforward with SHAP.
    model_to_explain_name = best_model_name
    if model_to_explain_name == "ANN":
        model_to_explain_name = results_df_sorted[results_df_sorted["Model"] != "ANN"].iloc[0]["Model"]
        print(f"  Note: Best model is ANN. Using {model_to_explain_name} for SHAP global explanation.")
    
    model_to_explain = trained_models[model_to_explain_name]
    
    if model_to_explain_name in ['Random Forest', 'Decision Tree']:
        explainer = shap.TreeExplainer(model_to_explain)
        shap_values = explainer(X_test_scaled)
        if len(shap_values.shape) > 2:
            # Multi-class output (ASD Positive is class 1)
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values
    else:
        # Linear/Logistic Regression, SVM, etc.
        explainer = shap.Explainer(model_to_explain, X_train_scaled)
        sv = explainer(X_test_scaled)
        
    sv.feature_names = feature_names
    
    # 1. Generate Global Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300)
    plt.close()
    
    # 2. Generate Feature Importance Bar Plot
    plt.figure(figsize=(10, 8))
    shap.plots.bar(sv, show=False)
    plt.tight_layout()
    plt.savefig("shap_feature_importance_bar.png", dpi=300)
    plt.close()
    
    print("  -> Saved 'shap_summary_plot.png'")
    print("  -> Saved 'shap_feature_importance_bar.png'")
    
except ImportError:
    print("  -> SHAP library not found. Skipping XAI plots. (Run: pip install shap)")
except Exception as e:
    print(f"  -> Could not generate SHAP explanation: {e}")

# ==========================================
# 16. SAVE ALL MODELS & ARTIFACTS TO DISK
# ==========================================
os.makedirs(MODELS_DIR, exist_ok=True)

with open(os.path.join(MODELS_DIR, "trained_models.pkl"), "wb") as f:
    pickle.dump(trained_models, f)

with open(os.path.join(MODELS_DIR, "ann.pkl"), "wb") as f:
    pickle.dump(ann, f)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODELS_DIR, "le_dict.pkl"), "wb") as f:
    pickle.dump(le_dict, f)

with open(os.path.join(MODELS_DIR, "results_df.pkl"), "wb") as f:
    pickle.dump(results_df, f)

with open(os.path.join(MODELS_DIR, "roc_data.pkl"), "wb") as f:
    pickle.dump(roc_data, f)

metadata = {
    "feature_names": feature_names,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
}
with open(os.path.join(MODELS_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print(f"\nAll models and artifacts saved to '{MODELS_DIR}/' folder.")
print("   You can now run the Streamlit app -- it will load these pre-trained models.")