# 🧠 Autism Spectrum Disorder (ASD) Screening Prediction System

A Machine Learning and Artificial Neural Network (ANN) based system to predict the likelihood of Autism Spectrum Disorder in children, built as part of a thesis entitled "PREDICTION OF AUTISM SPECTRUM DISORDER IN CHILDREN USING MACHINE LEARNING TECHNIQUES" in partial fulfillment of the requirements for the Degree of Master of Science in Information System Engineering at Purbanchal University School of Engineering, Nepal. The thesis is currently in progress and will be defended soon.


---

## 👤 Author
**Author:** Mrs. Chhayachabbi Jha  
**Thesis Supervisor:** Prof. Raj Kumar Thakur
**Email of Thesis Supervisor:** rajkshiva1@gmail.com 
GitHub: [rajaramayan](https://github.com/rajaramayan)

---

## 📌 Project Overview

This project trains and evaluates multiple ML models and an ANN on ASD screening data. The best-performing model can then be used for real-time predictions via an interactive Streamlit web application. Additionally, it integrates **Explainable AI (XAI)** using SHAP (SHapley Additive exPlanations) to provide transparent, feature-level insights into how the models arrive at their predictions both globally and per-patient.

---

## 🗂️ Project Structure

```
├── ASD in Children using Machine Learning and ANN (1).py   # Local training script (with SHAP XAI)
├── streamlit_app.py                                         # Streamlit web app (with SHAP XAI)
├── Toddler Autism dataset July 2018.csv                           # Dataset (1054 records)
├── models/                                                  # Saved model artifacts
│   ├── trained_models.pkl                                   # All 8 ML models
│   ├── ann.pkl                                              # ANN (MLPClassifier)
│   ├── scaler.pkl                                           # StandardScaler
│   ├── le_dict.pkl                                          # Label Encoders
│   ├── results_df.pkl                                       # Metrics results
│   ├── roc_data.pkl                                         # ROC curve data
│   └── metadata.pkl                                         # Feature metadata
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🤖 Models Used

| Model | Type |
|---|---|
| Logistic Regression | Classical ML |
| Decision Tree | Classical ML |
| Random Forest | Classical ML |
| K-Nearest Neighbors (KNN) | Classical ML |
| SVM (Polynomial Kernel) | Classical ML |
| SVM (RBF Kernel) | Classical ML |
| Naive Bayes | Classical ML |
| QDA | Classical ML |
| ANN (MLPClassifier) | Neural Network |

---

## 📊 Model Results (Held-Out Test Set, n=195)

All models are evaluated after target-leakage feature removal (`Qchat-10-Score`) and SMOTE training set rebalancing.

| Rank | Model | Test Accuracy | Precision | Recall | Specificity | F1 Score | ROC-AUC | Overfitting Status |
|---|---|---|---|---|---|---|---|---|
| 1 | **Logistic Regression** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | None (0.00% gap) |
| 2 | **SVM (RBF)** | 94.87% | 0.9324 | 1.0000 | 0.8246 | 0.9650 | 0.9986 | Low (3.86% gap) |
| 3 | **MLP-ANN** | 96.41% | 0.9645 | 0.9855 | 0.9123 | 0.9749 | 0.9968 | Low (2.41% gap) |
| 4 | **QDA** | 96.92% | 0.9853 | 0.9710 | 0.9649 | 0.9781 | 0.9963 | Low (1.36% gap) |
| 5 | **KNN** | 93.33% | 1.0000 | 0.9058 | 1.0000 | 0.9506 | 0.9962 | Low (1.69% gap) |
| 6 | **Random Forest** | 93.85% | 0.9375 | 0.9783 | 0.8421 | 0.9574 | 0.9886 | Moderate (4.53% gap) |
| 7 | **Naïve Bayes** | 92.31% | 0.9020 | 1.0000 | 0.7368 | 0.9485 | 0.9849 | Low (2.99% gap) |
| 8 | **Decision Tree** | 89.74% | 0.9097 | 0.9493 | 0.7719 | 0.9291 | 0.9265 | Moderate (4.47% gap) |
| 9 | **SVM (Poly)** | 75.38% | 0.9327 | 0.7029 | 0.8772 | 0.8017 | 0.8931 | Higher (8.16% gap) |

> **Best Model:** Logistic Regression — **100.00% Accuracy, 100.00% Precision, 100.00% Recall, ROC-AUC: 1.0000**  
> Dataset: `Toddler Autism dataset July 2018.csv` (1,054 records total; 195 test records)

---

## 🚀 How to Run

### 1. Navigate to Project Directory & Install Dependencies
```bash
cd autism-spectrum-detector2
pip install -r requirements.txt
```

### 2. Train Models Locally
```bash
python "ASD in Children using Machine Learning and ANN (1).py"
```
This trains all 9 models, prints evaluation metrics, shows plots, generates global SHAP Explainable AI charts (`shap_summary_plot.png`, `shap_feature_importance_bar.png`), and saves all model artifacts to the `models/` folder.

### 3. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```
Go to **🤖 Model Training** → click **📂 Load Pre-trained Models**.

---

## 🌐 Streamlit App Features

- **🏠 Home** — Project overview and usage guide
- **🤖 Model Training** — Load pre-trained models or train in-browser
- **🔮 Make Prediction** — Enter patient screening values and get ASD prediction, accompanied by a **SHAP Waterfall Plot** to explain the reasoning behind the specific prediction.
- **📊 Model Comparison** — ROC curves, bar charts, performance metrics table

---

## 🧪 Dataset

- **File:** `Toddler Autism dataset July 2018.csv`
- **Records:** 1,054 (toddlers)
- **Features:** 18 (A1–A10 screening questions, Age_Mons, Qchat-10-Score, Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who completed the test)
- **Target:** `Class/ASD Traits` (Yes / No)
- **Imbalance handling:** SMOTE (Synthetic Minority Over-sampling Technique)

---

## 🛠️ Tech Stack

- Python 3.10
- scikit-learn, imbalanced-learn, shap (SHAP)
- Streamlit
- NumPy, Pandas, Matplotlib, Seaborn

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It is not a substitute for professional medical diagnosis.
