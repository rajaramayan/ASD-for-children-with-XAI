import pickle, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

with open('models/trained_models.pkl','rb') as f: models = pickle.load(f)
with open('models/ann.pkl','rb') as f: ann = pickle.load(f)
with open('models/results_df.pkl','rb') as f: rdf = pickle.load(f)
with open('models/scaler.pkl','rb') as f: scaler = pickle.load(f)

MODEL_PRIORITY = {'ANN':1,'Random Forest':2,'Logistic Regression':3,'SVM (RBF)':4,'QDA':5,'KNN':6,'Naive Bayes':7,'SVM (Poly)':8,'Decision Tree':9}
rdf['_p'] = rdf['Model'].map(MODEL_PRIORITY)
rdf = rdf.sort_values(by=['ROC-AUC','_p'],ascending=[False,True]).drop(columns=['_p']).reset_index(drop=True)

df = pd.read_csv('Toddler Autism dataset July 2018.csv')
df.drop(columns=['Case_No'], errors='ignore', inplace=True)
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_enc = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
for col in df_enc.columns:
    df_enc[col] = pd.to_numeric(df_enc[col], errors='coerce')
df_enc.fillna(df_enc.mean(), inplace=True)
leaky_cols = [c for c in df_enc.columns if 'qchat' in c.lower() or 'score' in c.lower()]
drop_cols = list(set(['Class/ASD Traits'] + leaky_cols))
X = df_enc.drop(columns=[c for c in drop_cols if c in df_enc.columns])
y = df_enc['Class/ASD Traits']
_, X_test_raw, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_scaled = scaler.transform(X_test_raw)

n_pos = int(y_test.sum())
n_neg = int((y_test == 0).sum())
print(f"Test size: {len(y_test)}, ASD+(Yes)={n_pos}, ASD-(No)={n_neg}")
print()

all_models = {**models, 'ANN': ann}
for _, row in rdf.iterrows():
    name = row['Model']
    m = all_models[name]
    y_pred = m.predict(X_test_scaled)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"{name}:")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Acc={row['Accuracy']:.4f} Recall={row['Recall']:.4f} Spec={row['Specificity']:.4f}")
    print(f"  Prec={row['Precision']:.4f} F1={row['F1 Score']:.4f} AUC={row['ROC-AUC']:.4f} LL={row['Log Loss']:.6f}")
    print()
