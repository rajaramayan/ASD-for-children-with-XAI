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

This project trains and evaluates multiple ML models and an ANN on ASD screening data. The best-performing model can then be used for real-time predictions via an interactive Streamlit web application.

---

## 🗂️ Project Structure

```
├── ASD in Children using Machine Learning and ANN (1).py   # Local training script
├── streamlit_app.py                                         # Streamlit web app
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

## 📊 Model Results (Test Set)

| Model | Accuracy | F1 Score | ROC-AUC | Overfitting |
|---|---|---|---|---|
| **Logistic Regression** | **100.00%** | **1.0000** | **1.0000** | None |
| **Decision Tree** | **100.00%** | **1.0000** | **1.0000** | None |
| **Random Forest** | **100.00%** | **1.0000** | **1.0000** | None |
| **ANN** | **100.00%** | **1.0000** | **1.0000** | None |
| SVM (RBF) | 97.95% | 0.9857 | 0.9994 | None |
| QDA | 96.92% | 0.9781 | 0.9978 | None |
| KNN | 95.38% | 0.9663 | 0.9980 | None |
| Naive Bayes | 94.36% | 0.9617 | 0.9905 | Mild |
| SVM (Poly) | 69.74% | 0.7354 | 0.8781 | Moderate |

> **Best Models: Logistic Regression, Decision Tree, Random Forest, ANN** — ROC-AUC: 1.0000, Accuracy: 100.00%  
> Dataset: `Toddler Autism dataset July 2018.csv` (1,054 records)

---

## 🚀 How to Run

is lesser thn ### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models Locally
```bash
python "ASD in Children using Machine Learning and ANN (1).py"
```
This trains all 9 models, prints evaluation metrics, shows plots, and saves all model artifacts to the `models/` folder.

### 3. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```
Go to **🤖 Model Training** → click **📂 Load Pre-trained Models**.

---

## 🌐 Streamlit App Features

- **🏠 Home** — Project overview and usage guide
- **🤖 Model Training** — Load pre-trained models or train in-browser
- **🔮 Make Prediction** — Enter patient screening values and get ASD prediction
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
- scikit-learn, imbalanced-learn
- Streamlit
- NumPy, Pandas, Matplotlib, Seaborn

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It is not a substitute for professional medical diagnosis.
