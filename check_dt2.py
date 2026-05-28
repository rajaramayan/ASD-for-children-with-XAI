import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text
from imblearn.over_sampling import SMOTE

# Load saved model and scaler
with open('models/trained_models.pkl', 'rb') as f:
    trained_models = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Reproduce preprocessing
df = pd.read_csv('Toddler Autism dataset July 2018.csv')
df.drop(columns=['Case_No'], inplace=True)
df.drop_duplicates(inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

df_encoded = df.copy()
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_s, y_train_s = SMOTE(random_state=42).fit_resample(X_train, y_train)

dt = trained_models['Decision Tree']

print("=== Decision Tree Structure ===")
print(export_text(dt, feature_names=feature_names))

print("=== Split feature index:", dt.tree_.feature[0])
split_feature = feature_names[dt.tree_.feature[0]]
split_threshold = dt.tree_.threshold[0]
print(f"Split feature : {split_feature}")
print(f"Split threshold: {split_threshold:.4f}  (i.e., {split_feature} <= {split_threshold:.4f} → left)")

print()
print("=== Class distribution by Qchat-10-Score (original data) ===")
print(df.groupby(['Qchat-10-Score', 'Class/ASD Traits '])['Qchat-10-Score'].count().unstack(fill_value=0))

print()
print("=== Is Qchat-10-Score the sum of A1-A10? ===")
a_cols = [c for c in df.columns if c.startswith('A') and c[1:].isdigit()]
print(f"A-columns found: {a_cols}")
df['computed_score'] = df[a_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
match = (df['computed_score'] == df['Qchat-10-Score']).all()
print(f"computed sum == Qchat-10-Score for ALL rows: {match}")

print()
print("=== Cutoff verification ===")
print("Score < threshold -> ASD- (No), Score >= threshold -> ASD+ (Yes)")
print(df.groupby('Class/ASD Traits ')['Qchat-10-Score'].agg(['min','max','mean']))
