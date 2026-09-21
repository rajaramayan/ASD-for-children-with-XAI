import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import base64
import tempfile
import plotly.express as px
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

try:
    import shap
except ImportError:
    shap = None

try:
    import lime
    import lime.lime_tabular
except ImportError:
    lime = None

from sklearn.model_selection import train_test_split
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

MODELS_DIR = "models"

# Import TensorFlow only when needed (lazy loading)

# Set page configuration
st.set_page_config(
    page_title="ASD Screening Prediction System",
    page_icon="🧠",
    layout="wide"
)

# Premium UI CSS Injection
st.markdown("""
<style>
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Elegant Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(118, 75, 162, 0.3);
        color: white;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800;
        color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Autism Spectrum Disorder (ASD) Screening Prediction System")
st.markdown("---")

# Sidebar for navigation
page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home", 
        "🔍 Dataset Insights", 
        "🤖 Model Training", 
        "🔮 Make Prediction", 
        "🚀 Batch Prediction", 
        "📊 Model Comparison"
    ]
)

# ---- Close App ----
if "app_closed" not in st.session_state:
    st.session_state.app_closed = False

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ App Controls")
if st.sidebar.button("🔴 Close App", help="End your session", use_container_width=True):
    st.session_state.app_closed = True

if st.session_state.app_closed:
    st.balloons()
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <h1>👋 Thank You!</h1>
        <h3>Thank you for using the ASD Screening Prediction System.</h3>
        <br>
        <p style="font-size:18px;">Your session has ended.</p>
        <hr>
        <p style="color:gray;">You can now <strong>close this browser tab</strong>.<br>
        To fully stop the server, press <kbd>Ctrl+C</kbd> in the terminal window.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

@st.cache_data
def load_data():
    """Load the CSV dataset - cached for performance"""
    df = pd.read_csv("Toddler Autism dataset July 2018.csv")
    df.drop(columns=['Case_No'], errors='ignore', inplace=True)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def prepare_data(df):
    """Preprocess the data - cached for performance"""
    df = df.drop_duplicates()
    df = df.fillna(df.mode().iloc[0])
    
    # Identify numeric and categorical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    le_dict = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le
    
    # Ensure all columns are numeric
    for col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    
    df_encoded.fillna(df_encoded.mean(), inplace=True)
    
    return df_encoded, le_dict, numeric_cols, categorical_cols

def create_pdf_report(patient_data, prediction_text, confidence):
    if FPDF is None:
        return None
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Autism Spectrum Disorder (ASD) Screening Report", 0, 1, 'C')
        
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "---------------------------------------------------------", 0, 1, 'C')
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Diagnosis: {prediction_text} (Confidence: {confidence})", 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Patient Inputs:", 0, 1, 'L')
        
        pdf.set_font("Arial", '', 10)
        for key, val in patient_data.items():
            pdf.cell(0, 8, f"{key}: {val}", 0, 1, 'L')
            
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 10, "Note: AI-generated report for educational purposes only. Not a medical diagnosis.", 0, 1, 'C')
        
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_pdf.name)
        return temp_pdf.name
    except Exception as e:
        return None

def pretrained_models_exist():
    """Check if all required model files are present in the models/ folder"""
    required = ["trained_models.pkl", "ann.pkl", "scaler.pkl",
                "le_dict.pkl", "results_df.pkl", "roc_data.pkl", "metadata.pkl"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


def load_pretrained_models():
    """Load all models and artifacts saved by the local training script"""
    def _load(fname):
        with open(os.path.join(MODELS_DIR, fname), "rb") as f:
            return pickle.load(f)

    trained_models = _load("trained_models.pkl")
    ann            = _load("ann.pkl")
    scaler         = _load("scaler.pkl")
    le_dict        = _load("le_dict.pkl")
    results_df     = _load("results_df.pkl")
    roc_data       = _load("roc_data.pkl")
    metadata       = _load("metadata.pkl")

    st.session_state.trained_models  = trained_models
    st.session_state.ann             = ann
    st.session_state.scaler          = scaler
    st.session_state.le_dict         = le_dict
    st.session_state.results_df      = results_df
    st.session_state.roc_data        = roc_data
    st.session_state.feature_names   = metadata["feature_names"]
    st.session_state.numeric_cols    = metadata["numeric_cols"]
    st.session_state.categorical_cols = metadata["categorical_cols"]

    # Load encoded data for slider ranges and re-derive test set
    df = load_data()
    df_encoded, _, _, _ = prepare_data(df)
    st.session_state.df_encoded = df_encoded

    # Re-derive X_test_scaled and y_test using identical pipeline (excluding target leakage columns)
    leaky_cols = [c for c in df_encoded.columns if 'qchat' in c.lower() or 'score' in c.lower()]
    drop_cols = list(set(['Class/ASD Traits'] + leaky_cols))

    X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
    y = df_encoded['Class/ASD Traits']
    _, X_test_raw, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    st.session_state.X_test_scaled = scaler.transform(X_test_raw)
    st.session_state.y_test = y_test.values

def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train all ML models"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            ccp_alpha=0.005,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=6,
            max_features='sqrt',
            min_impurity_decrease=0.001,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=51,
            weights='uniform',
            metric='minkowski',
            p=1
        ),
        "SVM (Poly)": SVC(kernel='poly', degree=2, C=0.1, gamma='scale', probability=True),
        "SVM (RBF)": SVC(kernel='rbf', C=0.1, gamma='scale', probability=True),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.7)
    }
    
    results = []
    roc_data = []
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        gap = train_acc - acc

        results.append([name, train_acc, acc, gap, prec, rec, specificity, f1, roc_auc, logloss])
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((name, fpr, tpr, roc_auc))
    
    results_df = pd.DataFrame(results, columns=["Model", "Train Accuracy", "Accuracy", "Gap",
                                                "Precision", "Recall", "Specificity",
                                                "F1 Score", "ROC-AUC", "Log Loss"])
    
    return results_df, roc_data, trained_models

def train_ann(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train ANN model using sklearn MLPClassifier"""
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

    y_prob_ann = ann.predict_proba(X_test_scaled)[:, 1]
    y_pred_ann = ann.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, ann.predict(X_train_scaled))

    acc = accuracy_score(y_test, y_pred_ann)
    prec = precision_score(y_test, y_pred_ann)
    rec = recall_score(y_test, y_pred_ann)
    f1 = f1_score(y_test, y_pred_ann)
    roc_auc = roc_auc_score(y_test, y_prob_ann)
    logloss = log_loss(y_test, y_prob_ann)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ann).ravel()
    specificity = tn / (tn + fp)
    gap = train_acc - acc

    fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)

    return ann, [train_acc, acc, gap, prec, rec, specificity, f1, roc_auc, logloss], (fpr_ann, tpr_ann, roc_auc), None


# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "🏠 Home":
    st.subheader("Welcome to the ASD Screening Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### About This Application
        
        This application uses **Machine Learning** and **Artificial Neural Networks (ANN)** 
        to predict the likelihood of Autism Spectrum Disorder (ASD) based on screening data.
        
        **Features:**
        - Train multiple ML models
        - Compare model performance
        - Make predictions on new data
        - Visualize model performance metrics
        """)
    
    with col2:
        st.success("""
        ### ✅ Recommended Workflow
        
        **Step 1 — Train Locally:**
        Run the local training script to train and save all models:
        ```
        python "ASD in Children using Machine Learning and ANN (1).py"
        ```
        
        **Step 2 — Load in App:**
        Go to **🤖 Model Training** → click **📂 Load Pre-trained Models**
        
        **Step 3 — Predict & Compare:**
        Use **🔮 Make Prediction** and **📊 Model Comparison**
        
        > You can also train directly in the browser as a fallback.
        """)
    
    st.markdown("---")
    st.markdown("**💡 Tip:** Run the local training script first, then load models via the Model Training tab!")


# ==========================================
# PAGE 2: DATASET INSIGHTS (EDA)
# ==========================================
elif page == "🔍 Dataset Insights":
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    st.write("Understand the underlying patterns in the screening dataset before modeling.")
    
    df = load_data()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("ASD Positive Cases", len(df[df['Class/ASD Traits'] == 'Yes']))
    col3.metric("ASD Negative Cases", len(df[df['Class/ASD Traits'] == 'No']))
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Demographics", "Q-Chat Scores", "Correlations"])
    
    with tab1:
        st.write("### Age Distribution")
        fig_age = px.histogram(df, x="Age_Mons", color="Class/ASD Traits", barmode="group",
                               title="Age Distribution by ASD Traits",
                               color_discrete_sequence=['#764ba2', '#667eea'])
        st.plotly_chart(fig_age, use_container_width=True)
        
    with tab2:
        st.write("### Q-Chat-10 Score Distribution")
        fig_qchat = px.box(df, x="Class/ASD Traits", y="Qchat-10-Score", color="Class/ASD Traits",
                           title="Q-Chat-10 Scores by Diagnosis",
                           color_discrete_sequence=['#764ba2', '#667eea'])
        st.plotly_chart(fig_qchat, use_container_width=True)
        
    with tab3:
        st.write("### Feature Correlation Heatmap")
        df_numeric, _, _, _ = prepare_data(df)
        corr = df_numeric.corr()
        fig_corr = px.imshow(corr, text_auto=False, aspect="auto",
                             color_continuous_scale="Purples",
                             title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# PAGE 3: MODEL TRAINING
# ==========================================
elif page == "🤖 Model Training":
    st.subheader("🤖 Train Models")

    # ---- Load pre-trained models from disk (preferred workflow) ----
    if pretrained_models_exist():
        st.success(
            f"✅ Pre-trained models found in `{MODELS_DIR}/` folder. "
            "Load them instantly or re-train in-browser below."
        )
        if st.button("📂 Load Pre-trained Models", key="load_pretrained"):
            with st.spinner("Loading pre-trained models..."):
                load_pretrained_models()
            st.success("✅ Pre-trained models loaded successfully!")
        st.markdown("---")
    else:
        st.info(
            "ℹ️ No pre-trained models found. "
            "Run the local training script first to save models, "
            "or use the **Train in Browser** button below."
        )
        st.markdown("---")

    # ---- In-browser training (fallback) ----
    st.markdown("#### Train in Browser")
    try:
        df = load_data()
        df_encoded, le_dict, numeric_cols, categorical_cols = prepare_data(df)
        
        # Split features and target — EXCLUDE TARGET LEAKAGE COLUMNS (Qchat-10-Score)
        leaky_cols = [c for c in df_encoded.columns if 'qchat' in c.lower() or 'score' in c.lower()]
        drop_cols = list(set(['Class/ASD Traits'] + leaky_cols))

        X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
        y = df_encoded['Class/ASD Traits']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if st.button("🚀 Train All Models", key="train_button"):
            with st.spinner("Training models... This may take a moment..."):
                # Train classical models
                results_df, roc_data, trained_models = train_models(
                    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
                )
                
                # Train ANN
                ann, ann_metrics, ann_roc, history = train_ann(
                    X_train_scaled, X_test_scaled, y_train, y_test
                )
                
                # Add ANN to results
                results_df.loc[len(results_df)] = [
                    "ANN", ann_metrics[0], ann_metrics[1], ann_metrics[2],
                    ann_metrics[3], ann_metrics[4], ann_metrics[5],
                    ann_metrics[6], ann_metrics[7], ann_metrics[8]
                ]
                
                roc_data.append(("ANN", ann_roc[0], ann_roc[1], ann_roc[2]))
                
                st.success("✅ Training completed!")
                
                # Store in session state
                st.session_state.results_df = results_df
                st.session_state.roc_data = roc_data
                st.session_state.trained_models = trained_models
                st.session_state.ann = ann
                st.session_state.scaler = scaler
                st.session_state.le_dict = le_dict
                st.session_state.feature_names = X.columns.tolist()
                st.session_state.numeric_cols = numeric_cols
                st.session_state.categorical_cols = categorical_cols
                st.session_state.df_encoded = df_encoded
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.y_test = y_test.values
        
        # Display results if available
        if 'results_df' in st.session_state:
            st.subheader("📈 Model Performance Results")

            MODEL_PRIORITY = {
                "Logistic Regression": 1, "SVM (RBF)": 2, "ANN": 3,
                "Random Forest": 4, "QDA": 5, "KNN": 6,
                "Naive Bayes": 7, "Decision Tree": 8, "SVM (Poly)": 9
            }
            _df = st.session_state.results_df.copy()
            _df["_priority"] = _df["Model"].map(MODEL_PRIORITY).fillna(10)
            results_sorted = _df.sort_values(
                by=["ROC-AUC", "_priority"], ascending=[False, True]
            ).drop(columns=["_priority"]).reset_index(drop=True)
            st.dataframe(results_sorted.style.highlight_max(axis=0), use_container_width=True)

            # Best model
            best_model = results_sorted.iloc[0]
            msg = (f"🏆 Best Model: **{best_model['Model']}** "
                   f"with ROC-AUC: {best_model['ROC-AUC']:.4f} "
                   f"| Log Loss: {best_model['Log Loss']:.6f}")
            st.success(msg)

            # Overfitting Gap Table
            st.subheader("📊 Overfitting Analysis (Train vs Test Accuracy Gap)")
            gap_df = results_sorted[["Model", "Train Accuracy", "Accuracy", "Gap"]].copy()
            gap_df = gap_df.rename(columns={"Accuracy": "Test Accuracy"})

            def gap_status(gap):
                if gap < 0.02:
                    return "✅ No overfitting"
                elif gap < 0.05:
                    return "✅ Mild — acceptable"
                elif gap < 0.10:
                    return "⚠️ Moderate — needs justification"
                else:
                    return "❌ Severe overfitting"

            gap_df["Status"] = gap_df["Gap"].apply(gap_status)
            gap_df["Train Accuracy"] = gap_df["Train Accuracy"].map("{:.6f}".format)
            gap_df["Test Accuracy"] = gap_df["Test Accuracy"].map("{:.6f}".format)
            gap_df["Gap"] = gap_df["Gap"].map("{:.6f}".format)
            st.dataframe(gap_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error during training: {e}")

# ==========================================
# PAGE 3: MAKE PREDICTION
# ==========================================
elif page == "🔮 Make Prediction":
    st.subheader("🔮 Make Prediction")
    
    if 'scaler' not in st.session_state or 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train the models first on the 'Model Training' page")
    else:
        try:
            # Get feature names and data
            feature_names = st.session_state.feature_names
            df_encoded = st.session_state.df_encoded
            numeric_cols = st.session_state.numeric_cols
            categorical_cols = [col for col in st.session_state.categorical_cols if col != df_encoded.columns[-1]]
            le_dict = st.session_state.le_dict
            
            st.write("Enter screening values for the patient:")

            # Display labels (rename internal column names to user-friendly text)
            FEATURE_LABELS = {
                "Age_Mons": "Age_Months",
            }

            # Tooltip descriptions for Q-CHAT-10 behavioural items and score
            FEATURE_HELP = {
                "A1":  "Does your child look at you when you call his/her name?",
                "A2":  "How easy is it for you to get eye contact with your child?",
                "A3":  "Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
                "A4":  "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
                "A5":  "Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
                "A6":  "Does your child follow where you're looking?",
                "A7":  "If you or someone else in the family is visibly upset, "
                       "does your child show signs of wanting to comfort them?",
                "A8":  "Would you describe your child's first words as typical for their age?",
                "A9":  "Does your child use simple gestures? (e.g. wave goodbye)",
                "A10": "Does your child stare at nothing with no apparent purpose?",
                "Qchat-10-Score": "Total Q-CHAT-10 score (sum of A1–A10 responses, range 0–10). "
                                  "Higher scores indicate more ASD-associated behaviours.",
            }

            # Create input form
            user_input = {}
            cols = st.columns(3)

            for idx, feature in enumerate(feature_names):
                with cols[idx % 3]:
                    label = FEATURE_LABELS.get(feature, feature)
                    help_text = FEATURE_HELP.get(feature, None)
                    if feature in categorical_cols and feature in le_dict:
                        classes = list(le_dict[feature].classes_)
                        if len(classes) == 2:
                            # Binary categorical (Sex, Jaundice, Family_mem_with_ASD) → radio
                            selected_label = st.radio(
                                label, options=classes, horizontal=True, help=help_text
                            )
                        elif feature == "Ethnicity":
                            # Ethnicity has 11 options — vertical radio to avoid dropdown clipping
                            selected_label = st.radio(label, options=classes, help=help_text)
                        else:
                            # Other multi-class categorical (Who completed the test) → selectbox
                            selected_label = st.selectbox(label, options=classes, help=help_text)
                        user_input[feature] = int(le_dict[feature].transform([selected_label])[0])
                    else:
                        min_val = int(df_encoded[feature].min())
                        max_val = int(df_encoded[feature].max())
                        default_val = (min_val + max_val) // 2
                        if min_val == 0 and max_val == 1:
                            # Binary numeric (A1–A10) → radio with Yes/No labels
                            user_input[feature] = st.radio(
                                label, options=[0, 1], horizontal=True,
                                format_func=lambda x: "Yes" if x == 1 else "No",
                                help=help_text
                            )
                        else:
                            # Integer numeric (Age_Months, Qchat-10-Score) → integer slider
                            user_input[feature] = st.slider(
                                label,
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                step=1,
                                help=help_text
                            )
            
            if st.button("🎯 Predict", key="predict_button"):
                # Prepare input DataFrame with feature names to avoid UserWarning
                input_df = pd.DataFrame([{f: float(user_input[f]) for f in feature_names}], columns=feature_names)
                input_scaled = st.session_state.scaler.transform(input_df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Classical Model Predictions")
                    # Get best classical model (ranked by ROC-AUC, then model priority)
                    MODEL_PRIORITY = {
                        "Logistic Regression": 1, "SVM (RBF)": 2, "ANN": 3,
                        "Random Forest": 4, "QDA": 5, "KNN": 6,
                        "Naive Bayes": 7, "Decision Tree": 8, "SVM (Poly)": 9
                    }
                    _df2 = st.session_state.results_df.copy()
                    _df2["_priority"] = _df2["Model"].map(MODEL_PRIORITY).fillna(10)
                    results_sorted = _df2.sort_values(
                        by=["ROC-AUC", "_priority"], ascending=[False, True]
                    ).drop(columns=["_priority"])
                    best_classical = results_sorted[results_sorted['Model'] != 'ANN'].iloc[0]
                    
                    best_model = st.session_state.trained_models[best_classical['Model']]
                    prediction = best_model.predict(input_scaled)[0]
                    probability = best_model.predict_proba(input_scaled)[0][1]
                    
                    st.write(f"**Model:** {best_classical['Model']}")
                    st.write(f"**Prediction:** {'🔴 ASD Positive' if prediction == 1 else '🟢 ASD Negative'}")
                    st.write(f"**Confidence:** {probability*100:.2f}%")
                
                with col2:
                    st.subheader("ANN Model Prediction")
                    ann_prob = st.session_state.ann.predict_proba(input_scaled)[0][1]
                    ann_pred = 1 if ann_prob > 0.5 else 0
                    
                    st.write("**Model:** Artificial Neural Network")
                    st.write(f"**Prediction:** {'🔴 ASD Positive' if ann_pred == 1 else '🟢 ASD Negative'}")
                    st.write(f"**Confidence:** {ann_prob*100:.2f}%")
                
                # XAI Section
                st.markdown("---")
                st.subheader("🧠 Explainable AI (XAI) Analysis")
                
                xai_tab1, xai_tab2 = st.tabs(["SHAP Explanation", "LIME Explanation"])
                
                with xai_tab1:
                    if shap is None:
                        st.warning("⚠️ The `shap` library is not installed. To see feature explanations, stop the app and run: `pip install shap`")
                    elif best_classical['Model'] not in ['Logistic Regression', 'Random Forest', 'Decision Tree']:
                        st.info(f"ℹ️ SHAP explanations in this demo are optimized for Random Forest, Decision Tree, and Logistic Regression. The current best model is {best_classical['Model']}.")
                    else:
                        try:
                            with st.spinner("Generating SHAP explanations..."):
                                model_name = best_classical['Model']
                                if model_name in ['Random Forest', 'Decision Tree']:
                                    explainer = shap.TreeExplainer(best_model)
                                    shap_values = explainer(input_scaled)
                                    if len(shap_values.shape) > 2:
                                        sv = shap_values[:, :, 1]
                                    else:
                                        sv = shap_values
                                else:
                                    explainer = shap.Explainer(best_model, st.session_state.X_test_scaled)
                                    sv = explainer(input_scaled)
                                
                                sv.feature_names = feature_names
                                
                                st.write("### Feature Contributions (Waterfall Plot)")
                                st.write("Features in **red** pushed the probability higher (towards ASD Positive), while features in **blue** pushed it lower.")
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                shap.plots.waterfall(sv[0], show=False)
                                st.pyplot(fig)
                                plt.clf()
                        except Exception as e:
                            st.error(f"Could not generate SHAP explanation: {e}")

                with xai_tab2:
                    if lime is None:
                        st.warning("⚠️ The `lime` library is not installed. To see LIME explanations, stop the app and run: `pip install lime`")
                    else:
                        try:
                            with st.spinner("Generating LIME explanations..."):
                                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                                    st.session_state.X_test_scaled,
                                    feature_names=feature_names,
                                    class_names=['ASD Negative', 'ASD Positive'],
                                    mode='classification'
                                )
                                exp = lime_explainer.explain_instance(
                                    input_scaled[0], 
                                    best_model.predict_proba, 
                                    num_features=10,
                                    num_samples=1000
                                )
                                st.write("### Local Interpretable Model-agnostic Explanations (LIME)")
                                st.write("This plot shows the specific feature values for this patient and how much they contributed to the final prediction.")
                                
                                # Render LIME as a matplotlib plot instead of HTML to avoid browser freeze
                                fig = exp.as_pyplot_figure()
                                fig.set_size_inches(10, 6)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                        except Exception as e:
                            st.error(f"Could not generate LIME explanation: {e}")
                
                # PDF Generation
                st.markdown("---")
                st.subheader("📄 Download Clinical Report")
                if FPDF is not None:
                    pred_label = 'ASD Positive' if prediction == 1 else 'ASD Negative'
                    conf_str = f"{probability*100:.2f}%"
                    pdf_path = create_pdf_report(user_input, pred_label, conf_str)
                    
                    if pdf_path:
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_file,
                                file_name="ASD_Screening_Report.pdf",
                                mime="application/pdf"
                            )
                else:
                    st.info("The `fpdf2` library is required to generate PDF reports. (Run: pip install fpdf2)")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")

# ==========================================
# PAGE 4: BATCH PREDICTION
# ==========================================
elif page == "🚀 Batch Prediction":
    st.subheader("🚀 Batch Prediction via CSV")
    
    if 'scaler' not in st.session_state or 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train the models first on the 'Model Training' page")
    else:
        st.write("Upload a CSV file containing multiple patient records. Ensure the columns match the required screening features.")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(batch_df)} records.")
                
                # Check for required columns
                missing_cols = [col for col in st.session_state.feature_names if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns in uploaded CSV: {missing_cols}")
                    st.info(f"Required columns: {st.session_state.feature_names}")
                else:
                    # Keep only needed columns in correct order
                    batch_data = batch_df[st.session_state.feature_names].copy()
                    
                    if st.button("Predict Batch"):
                        with st.spinner("Processing predictions..."):
                            # The batch data must be encoded similar to training data.
                            # For simplicity, if it's already encoded, we scale it.
                            # If not encoded, we'd need to encode. Since this is a demo, we assume encoded or numeric.
                            # Let's enforce numeric conversion as a safety measure.
                            for col in batch_data.columns:
                                batch_data[col] = pd.to_numeric(batch_data[col], errors='coerce').fillna(0)
                                
                            batch_scaled = st.session_state.scaler.transform(batch_data)
                            
                            best_classical_name = st.session_state.results_df.iloc[0]['Model']
                            if best_classical_name == "ANN":
                                best_classical_name = st.session_state.results_df[st.session_state.results_df['Model'] != 'ANN'].iloc[0]['Model']
                                
                            best_model = st.session_state.trained_models[best_classical_name]
                            
                            preds = best_model.predict(batch_scaled)
                            probs = best_model.predict_proba(batch_scaled)[:, 1]
                            
                            batch_df['Prediction'] = ['ASD Positive' if p == 1 else 'ASD Negative' for p in preds]
                            batch_df['Confidence'] = [f"{prob*100:.2f}%" for prob in probs]
                            
                            st.success("Batch prediction complete!")
                            st.dataframe(batch_df)
                            
                            csv_output = batch_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=csv_output,
                                file_name='batch_predictions_results.csv',
                                mime='text/csv',
                            )
                            
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

# ==========================================
# PAGE 5: MODEL COMPARISON
# ==========================================
elif page == "📊 Model Comparison":
    st.subheader("📊 Model Comparison & Visualization")
    
    if 'results_df' not in st.session_state:
        st.warning("⚠️ Please train the models first on the 'Model Training' page")
    else:
        try:
            _df3 = st.session_state.results_df.copy()
            MODEL_PRIORITY = {
                "Logistic Regression": 1, "SVM (RBF)": 2, "ANN": 3,
                "Random Forest": 4, "QDA": 5, "KNN": 6,
                "Naive Bayes": 7, "Decision Tree": 8, "SVM (Poly)": 9
            }
            _df3["_priority"] = _df3["Model"].map(MODEL_PRIORITY).fillna(10)
            results_df = _df3.sort_values(
                by=["ROC-AUC", "_priority"], ascending=[False, True]
            ).drop(columns=["_priority"]).reset_index(drop=True)
            roc_data = st.session_state.roc_data
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Performance Metrics", "ROC Curves", "Bar Chart",
                "Model Ranking", "Confusion Matrices"
            ])
            
            with tab1:
                st.subheader("Detailed Metrics Table")
                st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
            
            with tab2:
                st.subheader("ROC Curve Comparison")
                fig, ax = plt.subplots(figsize=(10, 8))

                line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-']
                colors = plt.cm.tab10(np.linspace(0, 0.9, 9))
                fpr_grid = np.linspace(0, 1, 300)

                for i, (name, fpr, tpr, auc) in enumerate(roc_data):
                    # Use first occurrence of each FPR value so curves start at (0,0)
                    _, first_idx = np.unique(fpr, return_index=True)
                    tpr_smooth = np.interp(fpr_grid, fpr[first_idx], tpr[first_idx])
                    ax.plot(fpr_grid, tpr_smooth,
                            label=f"{name} (AUC={auc:.4f})",
                            linewidth=2,
                            linestyle=line_styles[i % len(line_styles)],
                            color=colors[i])

                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
                ax.set_xlabel("False Positive Rate", fontsize=12)
                ax.set_ylabel("True Positive Rate", fontsize=12)
                ax.set_title("ROC Curve Comparison (All Models)", fontsize=14, fontweight='bold')
                ax.legend(loc='lower right')
                ax.grid(alpha=0.3)

                st.pyplot(fig)
            
            with tab3:
                st.subheader("Model Performance Comparison")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                metrics_to_plot = ["Accuracy", "F1 Score", "ROC-AUC"]
                results_df.set_index("Model")[metrics_to_plot].plot(kind='bar', ax=ax)
                
                ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
                ax.set_ylabel("Score", fontsize=12)
                ax.set_xlabel("Model", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
            
            with tab4:
                st.subheader("Model Ranking by ROC-AUC")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sorted_results = results_df.sort_values("ROC-AUC", ascending=True)
                colors = [
                    '#2ecc71' if i == len(sorted_results) - 1 else '#3498db'
                    for i in range(len(sorted_results))
                ]
                
                ax.barh(sorted_results['Model'], sorted_results['ROC-AUC'], color=colors)
                ax.set_xlabel("ROC-AUC Score", fontsize=12)
                ax.set_title("Model Ranking by ROC-AUC", fontsize=14, fontweight='bold')
                ax.set_xlim([0, 1])
                
                for i, v in enumerate(sorted_results['ROC-AUC']):
                    ax.text(v + 0.02, i, f'{v:.4f}', va='center')
                
                st.pyplot(fig)

            with tab5:
                st.subheader("Confusion Matrices")

                if 'X_test_scaled' not in st.session_state or 'y_test' not in st.session_state:
                    st.warning("⚠️ Test set not available. Please re-load or re-train models.")
                else:
                    X_test_scaled = st.session_state.X_test_scaled
                    y_test = st.session_state.y_test

                    trained_models = st.session_state.trained_models
                    ann = st.session_state.ann

                    all_models = dict(trained_models)
                    all_models["ANN"] = ann
                    model_names = list(all_models.keys())

                    view_mode = st.radio(
                        "Display mode",
                        ["Single model", "All models (grid)"],
                        horizontal=True
                    )

                    def plot_cm(ax, y_true, y_pred, title, best=False):
                        cm = confusion_matrix(y_true, y_pred)
                        group_labels = ["TN", "FP", "FN", "TP"]
                        group_counts = [f"{v}" for v in cm.flatten()]
                        annot = np.array(
                            [f"{lbl}\n{cnt}" for lbl, cnt in zip(group_labels, group_counts)]
                        ).reshape(2, 2)
                        cmap = "Greens" if best else "Blues"
                        sns.heatmap(
                            cm, annot=annot, fmt="", cmap=cmap, ax=ax,
                            xticklabels=["Pred ASD−", "Pred ASD+"],
                            yticklabels=["Actual ASD−", "Actual ASD+"],
                            linewidths=0.5, cbar=False
                        )
                        tn, fp, fn, tp = cm.ravel()
                        acc = (tn + tp) / cm.sum()
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                        border = " ★" if best else ""
                        ax.set_title(f"{title}{border}\nAcc={acc:.2%}  Recall={rec:.2%}",
                                     fontsize=9, fontweight='bold' if best else 'normal')
                        ax.set_xlabel("Predicted", fontsize=8)
                        ax.set_ylabel("Actual", fontsize=8)

                    best_model_name = results_df.iloc[0]["Model"]

                    if view_mode == "Single model":
                        selected = st.selectbox("Select model", model_names,
                                                index=model_names.index(best_model_name)
                                                if best_model_name in model_names else 0)
                        y_pred = all_models[selected].predict(X_test_scaled)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        plot_cm(ax, y_test, y_pred, selected, best=(selected == best_model_name))
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Derived metrics table
                        cm = confusion_matrix(y_test, y_pred)
                        tn, fp, fn, tp = cm.ravel()
                        st.markdown(f"""
| Metric | Value |
|---|---|
| True Negatives (TN) | {tn} |
| False Positives (FP) | {fp} |
| False Negatives (FN) | {fn} |
| True Positives (TP) | {tp} |
| Accuracy | {(tn+tp)/cm.sum():.4f} |
| Recall (Sensitivity) | {tp/(tp+fn) if (tp+fn)>0 else 0:.4f} |
| Specificity | {tn/(tn+fp) if (tn+fp)>0 else 0:.4f} |
| Precision | {tp/(tp+fp) if (tp+fp)>0 else 0:.4f} |
""")

                    else:  # All models grid
                        ncols = 3
                        nrows = -(-len(model_names) // ncols)  # ceiling division
                        fig, axes = plt.subplots(nrows, ncols,
                                                 figsize=(5 * ncols, 4 * nrows))
                        axes = axes.flatten()
                        for i, name in enumerate(model_names):
                            y_pred = all_models[name].predict(X_test_scaled)
                            plot_cm(axes[i], y_test, y_pred, name,
                                    best=(name == best_model_name))
                        # Hide any unused subplots
                        for j in range(len(model_names), len(axes)):
                            axes[j].set_visible(False)
                        plt.suptitle("Confusion Matrices — All Models (Test Set)",
                                     fontsize=13, fontweight='bold', y=1.01)
                        plt.tight_layout()
                        st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error during visualization: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='font-size: 12px; color: gray;'>
    Autism Spectrum Disorder Screening Prediction System | Machine Learning & ANN | Educational Purpose Only
    </p>
</div>
""", unsafe_allow_html=True)
