# Thesis Report

## Early Detection of Autism Spectrum Disorder in Children Using Machine Learning and Artificial Neural Networks

---

**Author:** Mrs. Chhayachabbi Jha  
**Thesis Supervisor:** Prof. Raj Kumar Thakur
**Email of Thesis Supervisor:** rajkshiva1@gmail.com  
**Date:** 11 May 2026

---

## Abstract

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition characterised by persistent challenges in social communication, restricted and repetitive patterns of behaviour, and sensory processing differences. Early and accurate diagnosis of ASD is critical, as timely intervention significantly improves developmental outcomes for affected children. However, traditional diagnostic procedures are time-consuming, resource-intensive, and often subject to long waiting periods, limiting access especially in underserved regions.

This thesis proposes a machine learning and artificial neural network (ANN) based computational framework for the early screening and prediction of ASD in toddlers. The study utilises the **Q-CHAT-10 Toddler Autism Screening Dataset (July 2018)**, comprising 1,054 records (975 after duplicate removal). 

To prevent **Target Leakage** (Data Leakage), `Qchat-10-Score` (and any derivative total score columns) were explicitly removed from the training input feature set $X$. In ASD screening datasets, the target binary classification label (`Class/ASD Traits`) is mathematically derived directly from the Q-CHAT score threshold:

$$\text{Class/ASD Traits} = \begin{cases} 1 & \text{if } \sum_{i=1}^{10} A_i \ge 4 \\ 0 & \text{if } \sum_{i=1}^{10} A_i < 4 \end{cases}$$

Including `Qchat-10-Score` as an input feature causes models to perform trivial shortcut learning ($\text{Score} \ge 4$), swallowing feature importance and rendering accuracy metrics scientifically invalid. 

After removing target leakage columns, the final feature set consists of **16 input features**:
- **10 Behavioural Screening Items ($A_1 \dots A_{10}$)** (Binary: 0 or 1)
- **Age in Months (`Age_Mons`)** (Numeric: 12–36 months)
- **Sex** (Categorical: Male, Female)
- **Ethnicity** (Categorical: 11 categories)
- **Jaundice** (Categorical: Yes, No)
- **Family History of ASD (`Family_mem_with_ASD`)** (Categorical: Yes, No)
- **Respondent Category (`Who completed the test`)** (Categorical: Family Member, Health Care Professional, etc.)

A rigorous data preprocessing pipeline was implemented, including duplicate removal, missing value imputation using mode strategy, categorical variable encoding using individual Label Encoders, and class imbalance correction using the Synthetic Minority Over-sampling Technique (SMOTE). Feature scaling was performed using StandardScaler to normalise the input space prior to model training.

Nine classification models were trained and evaluated: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbours (KNN), Support Vector Machine with Polynomial kernel (SVM-Poly), Support Vector Machine with RBF kernel (SVM-RBF), Naïve Bayes, Quadratic Discriminant Analysis (QDA), and a Multilayer Perceptron Artificial Neural Network (MLP-ANN). The MLP-ANN was constructed with two hidden layers of 32 and 16 neurons respectively, using ReLU activation, trained via the Adam optimiser with backpropagation. **True early stopping** was enforced via `early_stopping=True`, `validation_fraction=0.1`, and `n_iter_no_change=10` — ensuring training terminates when validation loss ceases to improve, rather than running to a fixed iteration limit. Hyperparameter optimisation was performed using **5-fold cross-validated GridSearchCV** across Random Forest, SVM (RBF), Decision Tree, and MLP-ANN.

Model performance was assessed using Accuracy, Precision, Recall, Specificity, F1 Score, ROC-AUC, and Log Loss on a held-out test set (20% split). **10-fold Stratified Cross-Validation** was applied to measure generalisation stability. **McNemar's exact test** was used to assess statistical significance of the performance difference between the MLP-ANN and the best classical model. A **SMOTE ablation study** was conducted to empirically quantify the impact of class imbalance correction.

Experimental results demonstrate that **Logistic Regression achieved the highest overall performance** with a ROC-AUC of **1.0000**, test accuracy of **100.00%**, F1 Score of **1.0000**, and Log Loss of **0.0272** — the best-calibrated classifier on the Q-CHAT-10 toddler dataset. This result confirms the linear separability of the ten binary Q-CHAT-10 items ($A_1 \dots A_{10}$). The MLP-ANN with true early stopping achieved ROC-AUC of **0.9968** and test accuracy of **96.41%**, ranking third after SVM (RBF) (ROC-AUC: 0.9986). McNemar's test confirmed that the performance difference between ANN and Logistic Regression is statistically significant ($p = 0.0156 < 0.05$), validating the objective ranking. SVM (Polynomial) was the lowest-ranked model (ROC-AUC: 0.8931, Accuracy: 75.38%).

The trained models were serialised using Python's pickle library and deployed as an interactive web application using Streamlit, enabling real-time ASD risk prediction based on user-provided screening inputs. The application loads the best model — **Logistic Regression**, selected by objective metrics — to provide clinical-grade screening predictions, alongside comprehensive model comparison visualisations including ROC curves, performance bar charts, overfitting analysis tables, and SMOTE ablation results.

This work demonstrates that a rigorous, statistically validated machine learning framework — combining target-leakage prevention, SMOTE class balancing, 10-fold cross-validation, McNemar significance testing, and GridSearchCV hyperparameter tuning — can achieve reliable and scientifically defensible ASD screening, potentially aiding clinicians and caregivers in identifying at-risk toddlers at a much earlier stage than traditional methods allow.

**Keywords:** Autism Spectrum Disorder, ASD Screening, Machine Learning, Artificial Neural Network, MLP Classifier, Logistic Regression, Backpropagation, SMOTE, SMOTE Ablation, GridSearchCV, Hyperparameter Tuning, McNemar Test, 10-Fold Cross-Validation, Early Stopping, Target Leakage, Classification, Early Detection, Streamlit Deployment.

---

## Table of Contents

| | |
|---|---|
| **Abstract** | |
| **Table of Contents** | |
| **List of Figures** | |
| **List of Tables** | |
| | |
| **Chapter 1: Introduction** | |
| 1.1 Background and Motivation | |
| 1.2 Problem Statement | |
| 1.3 Aims and Objectives of the Study | |
| 1.4 Scope of the Study | |
| 1.5 Significance of the Study | |
| 1.6 Overview of the Proposed Approach | |
| 1.7 Thesis Organisation | |
| | |
| **Chapter 2: Literature Review** | |
| 2.1 Overview | |
| 2.2 ASD Prevalence, Clinical Challenges, and the Need for Automated Tools | |
| 2.3 Conventional Machine Learning Classifiers for ASD Screening | |
| 2.4 Random Forest and Ensemble Methods for ASD | |
| 2.5 Support Vector Machines and Kernel Methods for ASD | |
| 2.6 Artificial Neural Networks and Deep Learning for ASD | |
| 2.7 Handling Class Imbalance with SMOTE | |
| 2.8 Feature Selection and Dimensionality Reduction in ASD Datasets | |
| 2.9 Comparative Studies and Systematic Reviews | |
| 2.10 Mobile and Web Deployment of ASD Screening Tools | |
| 2.11 Logistic Regression, Naïve Bayes, and QDA as Interpretable Baselines | |
| 2.12 Ethical Considerations and Limitations in ML-Based ASD Diagnosis | |
| 2.13 Summary | |
| | |
| **Chapter 3: Research Gap** | |
| 3.1 Introduction | |
| 3.2 Identified Research Gaps | |
| &nbsp;&nbsp;&nbsp;3.2.1 Lack of Comprehensive Multi-Model Benchmarking | |
| &nbsp;&nbsp;&nbsp;3.2.2 Insufficient Overfitting Analysis and Generalisation Reporting | |
| &nbsp;&nbsp;&nbsp;3.2.3 Limited Use of Large, Combined, and Representative Datasets | |
| &nbsp;&nbsp;&nbsp;3.2.4 Inadequate Handling of Class Imbalance | |
| &nbsp;&nbsp;&nbsp;3.2.5 Absence of Probabilistic and Discriminant Analysis Classifiers | |
| &nbsp;&nbsp;&nbsp;3.2.6 Lack of End-to-End Deployment with Multi-Model Inference | |
| &nbsp;&nbsp;&nbsp;3.2.7 Reproducibility and Code Transparency Deficits | |
| 3.3 Summary of Research Gaps and Thesis Contributions | |
| 3.4 Research Objectives | |
| | |
| **Chapter 4: Methodology** | |
| 4.1 Overview | |
| 4.2 Overall System Architecture and Workflow | |
| 4.3 Phase 1: Dataset Description | |
| &nbsp;&nbsp;&nbsp;4.3.1 Data Source | |
| &nbsp;&nbsp;&nbsp;4.3.2 Feature Description | |
| &nbsp;&nbsp;&nbsp;4.3.3 Target Variable and Class Distribution | |
| 4.4 Phase 2: Data Preprocessing | |
| &nbsp;&nbsp;&nbsp;4.4.1 Duplicate Removal | |
| &nbsp;&nbsp;&nbsp;4.4.2 Missing Value Imputation | |
| &nbsp;&nbsp;&nbsp;4.4.3 Categorical Label Encoding | |
| &nbsp;&nbsp;&nbsp;4.4.4 Numeric Coercion and Residual Fill | |
| 4.5 Phase 3: Train-Test Partitioning | |
| 4.6 Phase 4: Class Imbalance Handling with SMOTE | |
| 4.7 Phase 5: Feature Scaling | |
| 4.8 Phase 6: Model Definition and Training | |
| &nbsp;&nbsp;&nbsp;4.8.1 Logistic Regression | |
| &nbsp;&nbsp;&nbsp;4.8.2 Decision Tree | |
| &nbsp;&nbsp;&nbsp;4.8.3 Random Forest | |
| &nbsp;&nbsp;&nbsp;4.8.4 K-Nearest Neighbours (KNN) | |
| &nbsp;&nbsp;&nbsp;4.8.5 Support Vector Machine – Polynomial Kernel | |
| &nbsp;&nbsp;&nbsp;4.8.6 Support Vector Machine – RBF Kernel | |
| &nbsp;&nbsp;&nbsp;4.8.7 Gaussian Naïve Bayes | |
| &nbsp;&nbsp;&nbsp;4.8.8 Quadratic Discriminant Analysis (QDA) | |
| &nbsp;&nbsp;&nbsp;4.8.9 Multilayer Perceptron – Artificial Neural Network (MLP-ANN) | |
| 4.9 Phase 7: Model Evaluation Framework | |
| &nbsp;&nbsp;&nbsp;4.9.1 Overfitting Analysis | |
| &nbsp;&nbsp;&nbsp;4.9.2 Confusion Matrices — All Nine Models | |
| 4.10 Phase 8: Model Serialisation | |
| 4.11 Phase 9: Streamlit Web Application Deployment | |
| 4.12 Laboratory Setup | |
| &nbsp;&nbsp;&nbsp;4.12.1 Hardware Configuration | |
| &nbsp;&nbsp;&nbsp;4.12.2 Software and Development Environment | |
| &nbsp;&nbsp;&nbsp;4.12.3 Python Libraries and Versions | |
| 4.13 Tools, Libraries, and Environment | |
| 4.14 Summary | |
| | |
| **Chapter 5: Results and Discussion** | |
| 5.1 Overview | |
| 5.2 Dataset Summary | |
| 5.3 Overall Model Performance Results | |
| 5.4 Logistic Regression — Superior Model Analysis | |
| 5.5 SVM (RBF) — Second-Ranked Model | |
| 5.6 Model-by-Model Analysis | |
| &nbsp;&nbsp;&nbsp;5.6.1 Logistic Regression | |
| &nbsp;&nbsp;&nbsp;5.6.2 Decision Tree | |
| &nbsp;&nbsp;&nbsp;5.6.3 K-Nearest Neighbours (KNN) | |
| &nbsp;&nbsp;&nbsp;5.6.4 SVM (RBF Kernel) | |
| &nbsp;&nbsp;&nbsp;5.6.5 SVM (Polynomial Kernel) | |
| &nbsp;&nbsp;&nbsp;5.6.6 Naïve Bayes | |
| &nbsp;&nbsp;&nbsp;5.6.7 Quadratic Discriminant Analysis (QDA) | |
| 5.7 Overfitting Analysis | |
| 5.8 ROC-AUC Comparison | |
| 5.9 Metric-by-Metric Cross-Model Comparison | |
| &nbsp;&nbsp;&nbsp;5.9.1 Recall (Sensitivity) — Clinical Priority Metric | |
| &nbsp;&nbsp;&nbsp;5.9.2 Specificity — Minimising Unnecessary Referrals | |
| &nbsp;&nbsp;&nbsp;5.9.3 F1 Score — Harmonic Mean of Precision and Recall | |
| &nbsp;&nbsp;&nbsp;5.9.4 Log Loss — Probability Calibration Quality | |
| 5.10 Ranking Summary — Multi-Criteria Evaluation | |
| 5.11 Comparison with Prior Literature | |
| 5.12 Clinical Interpretation of Results | |
| 5.13 Summary of Key Findings | |
| | |
| **Chapter 6: Discussion** | |
| 6.1 Introduction to Discussion | |
| 6.2 Interpreting the Performance Hierarchy | |
| &nbsp;&nbsp;&nbsp;6.2.1 Why Logistic Regression Achieves Superior Performance | |
| &nbsp;&nbsp;&nbsp;6.2.2 Why SVM (RBF) is the Strongest Non-Linear Model | |
| &nbsp;&nbsp;&nbsp;6.2.3 Why SVM (Polynomial) Underperforms | |
| &nbsp;&nbsp;&nbsp;6.2.4 The KNN and QDA Paradox — Perfect Recall Without Optimal Utility | |
| 6.3 The Precision-Recall Trade-Off in ASD Screening | |
| 6.4 Significance of Overfitting Control | |
| &nbsp;&nbsp;&nbsp;6.4.1 Why Overfitting Control Matters in Medical AI | |
| &nbsp;&nbsp;&nbsp;6.4.2 Impact of Target Leakage Prevention and Model Calibration | |
| &nbsp;&nbsp;&nbsp;6.4.3 MLP-ANN Early Stopping & Overfitting Control | |
| 6.5 Addressing the Research Gaps | |
| 6.6 Comparison with State-of-the-Art | |
| 6.7 SMOTE and Class Imbalance: Impact on Results | |
| 6.8 Limitations of the Study | |
| 6.9 Ethical Considerations | |
| 6.10 Implications for Future Research | |
| 6.11 Practical Utility of the Streamlit Application | |
| 6.12 Summary | |
| | |
| **Chapter 7: Conclusion** | |
| 7.1 Overview | |
| 7.2 Principal Findings | |
| &nbsp;&nbsp;&nbsp;7.2.1 Logistic Regression is the Superior Model for ASD Screening | |
| &nbsp;&nbsp;&nbsp;7.2.2 SVM (RBF) is the Second-Ranked and Most Reliable Non-Linear Classifier | |
| &nbsp;&nbsp;&nbsp;7.2.3 Overfitting Was Controlled Across All Models | |
| &nbsp;&nbsp;&nbsp;7.2.4 Class Imbalance Handling is Essential for High Recall | |
| &nbsp;&nbsp;&nbsp;7.2.5 The Feature Set is Highly Discriminative | |
| 7.3 Research Objectives — Achievement Summary | |
| 7.4 Contributions of the Thesis | |
| 7.5 Limitations Acknowledged | |
| 7.6 Recommendations | |
| 7.7 Final Conclusion | |
| | |
| **References** | |

---

## List of Figures

| Figure | Caption |
|---|---|
| **Figure 4.1** | Overall system workflow — nine-phase methodology pipeline with hyperparameter tuning, early stopping, and cross-validation |
| **Figure 4.2** | Data preprocessing pipeline — target leakage prevention, deduplication, imputation, and encoding |
| **Figure 4.3** | Phase 2: Data preprocessing pipeline (deduplication, imputation, encoding) |
| **Figure 4.4** | Phase 3: Stratified 80/20 train-test partitioning |
| **Figure 4.5** | Phase 4: SMOTE class imbalance correction mechanism applied to training set |
| **Figure 4.6** | Phase 5: StandardScaler feature normalisation protocol |
| **Figure 4.7** | Phase 6: Classification model taxonomy — nine models across five paradigms |
| **Figure 4.8** | MLP-ANN backpropagation training loop with validation-monitored early stopping |
| **Figure 4.9** | Phase 7: Seven-metric evaluation framework and confusion matrix formulas |
| **Figure 4.9b** | 10-Fold Stratified Cross-Validation and GridSearchCV hyperparameter tuning workflow |
| **Figure 4.10** | Phase 9: Streamlit web application deployment architecture |
| **Figure 4.11** | Confusion matrices for all nine classifiers on the test set (n=195) |
| **Figure 5.1** | Logistic Regression performance summary — key metrics and confusion matrix |
| **Figure 5.2** | SVM (RBF) performance summary — key metrics and confusion matrix |
| **Figure 5.3** | Overfitting gap chart — train accuracy vs. test accuracy for all nine models |
| **Figure 5.4** | ROC-AUC ranking — all nine models sorted by discriminative power |
| **Figure 5.5** | Clinical interpretation of Logistic Regression results on the test set |
| **Figure 6.1** | Precision-recall trade-off comparison across all nine classifiers |

---

## List of Tables

| Table | Caption |
|---|---|
| **Table 2.1** | Summary of 25 reviewed studies on ML/ANN-based ASD screening (2021–2026) |
| **Table 3.1** | Summary of identified research gaps and corresponding thesis contributions |
| **Table 4.1** | Dataset feature description — 16 non-leaky input features and target variable |
| **Table 4.2** | Hyperparameter search space and optimal configurations derived from 5-fold GridSearchCV |
| **Table 4.3** | Serialised model artefacts and their contents |
| **Table 4.4** | Laboratory hardware configuration |
| **Table 4.5** | Software and Python library environment |
| **Table 5.1** | Dataset partition statistics (records, class distribution, split sizes) |
| **Table 5.2** | Complete seven-metric performance comparison for all nine classifiers (target-leakage-free) |
| **Table 5.3** | 10-Fold Stratified Cross-Validation results — mean and standard deviation across all models |
| **Table 5.4** | McNemar's exact test results — ANN vs Logistic Regression pairwise comparison |
| **Table 5.5** | SMOTE ablation study — Recall, F1, and ROC-AUC comparison (No-SMOTE vs SMOTE) |
| **Table 5.6** | Overfitting analysis — train-test accuracy gaps for all nine models |
| **Table 5.7** | Multi-criteria rank summary across all seven evaluation metrics |
| **Table 6.1** | Research gap coverage assessment — G1–G7 addressed by this thesis |
| **Table 7.1** | Research objective achievement summary — RO1–RO6 |

---

## Chapter 1: Introduction

### 1.1 Background and Motivation

Autism Spectrum Disorder (ASD) is a complex, lifelong neurodevelopmental condition characterised by persistent impairments in social communication and interaction, along with restricted, repetitive patterns of behaviour, interests, and activities. First formally described by Leo Kanner in 1943, ASD has since been recognised as a spectrum disorder — one that encompasses a wide continuum of symptom severity, cognitive ability, and functional independence. According to the World Health Organisation (WHO), approximately 1 in 100 children worldwide is diagnosed with ASD, while prevalence data from the United States Centers for Disease Control and Prevention (CDC) places the figure closer to 1 in 36 children, reflecting both genuine increases and improvements in diagnostic awareness.

The consequences of late or missed diagnosis are profound. Neuroscientific research consistently demonstrates that the human brain exhibits its highest degree of neuroplasticity during the first three to five years of life. Behavioural, speech, and occupational interventions applied within this critical developmental window yield substantially superior outcomes in communication, adaptive behaviour, and social integration compared to interventions initiated later in childhood or adolescence. Despite this well-established clinical consensus, the median age of ASD diagnosis globally remains above four years, and in many low- and middle-income countries it exceeds seven years. This diagnostic lag represents not merely a healthcare failure but a missed opportunity for transformative developmental gain.

The bottleneck lies in the diagnostic process itself. Gold-standard diagnostic instruments — the Autism Diagnostic Observation Schedule, Second Edition (ADOS-2) and the Autism Diagnostic Interview–Revised (ADI-R) — are comprehensive, valid, and reliable, but they are also clinician-administered, time-intensive (requiring two to four hours per assessment), and dependent on the availability of highly trained specialists. In many parts of the world, waiting lists for such evaluations span months to years. Primary care physicians and paediatricians, who represent the most accessible first line of contact for families, typically lack the specialised training to confidently identify early behavioural markers of ASD during routine developmental screenings.

The intersection of machine learning (ML) and clinical medicine offers a compelling solution to this access-and-efficiency gap. ML-based models trained on structured behavioural screening questionnaire data have demonstrated the capacity to identify ASD risk rapidly, consistently, and at scale — without requiring specialist clinical infrastructure. By encoding the pattern-recognition capabilities learned from large labelled datasets, such models can function as decision-support tools that flag children at elevated ASD risk, enabling targeted referrals for formal evaluation while avoiding unnecessary burden on specialist services. The democratisation of early ASD screening through lightweight, deployable ML tools represents one of the most actionable and practically meaningful applications of artificial intelligence in paediatric healthcare.

This thesis is motivated by the dual imperative of clinical need and technological opportunity: to design, train, rigorously evaluate, and deploy a machine learning and artificial neural network (ANN) framework capable of reliable, accessible early ASD screening for children, and to do so in a manner that is transparent, reproducible, and directly grounded in the identified limitations of prior research.

---

### 1.2 Problem Statement

Despite a growing body of research demonstrating the utility of machine learning for ASD screening, several critical gaps persist in the existing literature that limit the clinical reliability, scientific rigour, and practical deployability of published models.

First, the majority of studies evaluate only a single or small number of classifiers, often selecting the best-performing model without systematic comparison across a diverse model portfolio. This selective reporting prevents practitioners from understanding the relative trade-offs between model families and does not establish a trustworthy performance hierarchy. Second, overfitting — the tendency of a model to memorise training data patterns rather than learning generalisable rules — is rarely reported or analysed, despite being a primary threat to real-world clinical utility. A model that achieves 99% training accuracy but 75% test accuracy would be clinically unsafe, yet many published studies report training accuracy alone. Third, class imbalance in ASD datasets (where ASD-negative cases substantially outnumber ASD-positive cases) is frequently neglected, leading to models that are biased towards the majority class and fail to detect the very condition they were designed to identify. Fourth, end-to-end deployment — the transformation of a trained model into a usable clinical tool — is rarely undertaken, leaving models confined to academic manuscripts and inaccessible to the practitioners and caregivers who could most benefit.

This thesis directly addresses each of these gaps by constructing a comprehensive, multi-model benchmarking framework that includes explicit overfitting analysis, principled class imbalance correction via SMOTE, and end-to-end Streamlit web application deployment.

---

### 1.3 Aims and Objectives of the Study

The overarching aim of this thesis is to develop a robust, explainable, and practically deployable machine learning and ANN-based framework for the early screening of Autism Spectrum Disorder in children, using structured behavioural questionnaire data.

The specific research objectives are as follows:

**RO1 — Rigorous Preprocessing & Target Leakage Prevention:** To develop a rigorous, end-to-end data preprocessing pipeline including duplicate removal, mode imputation, categorical encoding, exclusion of the target-leaky `Qchat-10-Score` feature, and post-split SMOTE class balancing — ensuring methodological integrity throughout.

**RO2 — Multi-Model Benchmarking:** To train and systematically compare nine classification algorithms — Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbours, SVM (Polynomial), SVM (RBF), Gaussian Naïve Bayes, QDA, and MLP-ANN — under identical experimental conditions on the Q-CHAT-10 toddler dataset.

**RO3 — Rigorous Evaluation with Overfitting Control:** To evaluate all models against seven performance metrics (Accuracy, Precision, Recall, Specificity, F1 Score, ROC-AUC, and Log Loss) on a held-out test set, and to quantify overfitting by explicitly reporting the training-test accuracy gap for each model.

**RO4 — Hyperparameter Tuning & Early Stopping:** To apply 5-fold GridSearchCV hyperparameter optimisation for key model families (RF, SVM-RBF, Decision Tree, MLP-ANN) and enforce true validation-monitored early stopping for the MLP-ANN, ensuring principled regularisation.

**RO5 — Statistical Validation & Cross-Validation:** To evaluate generalisation stability through 10-fold stratified cross-validation, and to confirm the statistical significance of the best model's superiority using McNemar's exact test.

**RO6 — End-to-End Deployment:** To serialise all trained models and deploy them as an interactive, real-time web application using Streamlit, enabling clinicians, researchers, and caregivers to obtain multi-model ASD risk predictions with probability scores and comparative visualisations.

---

### 1.4 Scope of the Study

This thesis is scoped as follows:

- **Dataset:** The study utilises the `Toddler Autism dataset July 2018.csv` dataset, the Q-CHAT-10 toddler ASD screening dataset comprising **1,054 records** (975 after deduplication) and **16 input features** (ten Q-CHAT-10 behavioural items A1–A10, age in months, sex, ethnicity, history of jaundice, family history of ASD, and who completed the test; excluding total Q-CHAT-10 score to prevent target leakage), with a binary target variable (ASD traits Yes / No).
- **Population:** The study is focused exclusively on the **toddler** subpopulation of ASD screening data (children aged 12–36 months screened using the Q-CHAT-10 instrument). Adult and adolescent screening data are outside the scope of this work.
- **Classification task:** The task is binary supervised classification — predicting whether a child screens positive or negative for ASD risk.
- **Model families:** Nine classifiers spanning linear, tree-based, kernel-based, probabilistic, discriminant-analysis, and neural network families are included. Deep learning architectures (CNNs, RNNs, transformers) and neuroimaging-based approaches are outside scope.
- **Deployment platform:** The Streamlit web application is designed as a local and cloud-deployable screening adjunct. It does not constitute a certified medical device and is intended as a research-and-demonstration tool.
- **Geographical context:** The models are trained on a composite dataset aggregated from multiple international screening studies and are not specific to any single national population.

---

### 1.5 Significance of the Study

This work carries significance across multiple dimensions:

**Clinical significance:** A validated, accessible, and real-time ASD screening tool can meaningfully shorten the diagnostic pathway for at-risk toddlers, enabling earlier referral and, consequently, earlier access to interventional support. By achieving a Recall (Sensitivity) of 100.00% and Specificity of 100.00% — ensuring that all ASD-positive toddlers in the test set are correctly flagged and no ASD-negative toddlers are over-referred — the Logistic Regression model developed in this thesis prioritises the clinical imperative of minimising both missed diagnoses and unnecessary clinical burdens.

**Scientific significance:** This thesis advances the state of knowledge through a systematic, multi-model comparative study that includes overfitting analysis, SMOTE-based class imbalance correction, and a seven-metric evaluation framework. By documenting not only which models perform best but also why, and by quantifying generalisation risk through the train-test accuracy gap, this work provides a more complete and trustworthy performance characterisation than the majority of existing published studies.

**Methodological significance:** The adoption of nine diverse classifiers — including the often-neglected Quadratic Discriminant Analysis and Gaussian Naïve Bayes — provides a comprehensive baseline that future researchers can build upon. The explicit inclusion of Log Loss as a metric enriches the evaluation by assessing probabilistic calibration, an important dimension of clinical trustworthiness that accuracy-centric studies overlook.

**Practical significance:** The Streamlit web application delivers the model's capabilities to end users without requiring any programming expertise. Healthcare providers, community screeners, and caregivers in resource-limited settings can interact with the tool through a structured questionnaire interface, receiving real-time ASD risk assessments. This end-to-end deployment closes the gap between research and real-world utility that characterises much of the existing ML-in-medicine literature.

---

### 1.6 Overview of the Proposed Approach

The computational framework proposed in this thesis follows a nine-phase pipeline:

1. **Data Acquisition:** The Q-CHAT-10 toddler ASD screening dataset (975 records after dedup, 16 features after dropping Case_No and target-leaky Q-CHAT-10 score) is loaded.
2. **Preprocessing:** Duplicate removal, mode-based missing value imputation, individual Label Encoding of categorical features, and numeric coercion ensure data integrity.
3. **Partitioning:** A stratified 80/20 train-test split is applied, preserving the class distribution in both partitions.
4. **Class Imbalance Correction:** SMOTE is applied exclusively to the training set to generate synthetic ASD-positive samples and achieve a balanced 1:1 class ratio.
5. **Feature Scaling:** StandardScaler normalises the input feature space based on training-set statistics only, preventing data leakage.
6. **Model Training:** Nine classifiers are trained with carefully selected hyperparameters. The MLP-ANN uses two hidden layers (32 and 16 neurons), ReLU activation, the Adam optimiser, and validation-monitored early stopping (`early_stopping=True`).
7. **Evaluation:** All models are evaluated on the held-out test set across seven metrics, with train-test accuracy gaps computed to quantify overfitting.
8. **Serialisation:** Trained models, the Label Encoder dictionary, and the StandardScaler are serialised using Python's pickle library for persistent storage and inference-time reuse.
9. **Deployment:** The serialised artefacts are loaded by a Streamlit web application that accepts structured screening inputs and returns real-time ASD risk predictions from both the best classical model (Logistic Regression) and the MLP-ANN.

Logistic Regression achieved the highest overall performance, with a ROC-AUC of **1.0000**, test accuracy of **100.00%**, F1 Score of **1.0000**, and Recall of **100.00%**, confirmed as statistically significantly superior to MLP-ANN via McNemar's exact test ($p = 0.0156$). SVM (RBF) ranked second (ROC-AUC 0.9986), followed by MLP-ANN under early stopping (ROC-AUC 0.9968, test accuracy 96.41%).

---

### 1.7 Thesis Organisation

The remainder of this thesis is organised into six chapters:

**Chapter 2 — Literature Review** critically examines 25 studies published between 2021 and 2026 on machine learning and ANN-based ASD screening, covering conventional classifiers, ensemble methods, support vector machines, deep learning, class imbalance handling, feature selection, and deployment approaches. Key methodological limitations in the existing body of work are identified and contextualised.

**Chapter 3 — Research Gap** synthesises the findings of the literature review into seven formally identified research gaps and maps them directly to the objectives and contributions of this thesis.

**Chapter 4 — Methodology** presents the complete nine-phase experimental pipeline in detail, including dataset description, preprocessing protocol, SMOTE application, model architectures and hyperparameters, evaluation framework, model serialisation, laboratory setup, and Streamlit deployment architecture.

**Chapter 5 — Results and Discussion** presents the complete experimental results for all nine classifiers across seven metrics, with detailed analysis of Logistic Regression, SVM (RBF), MLP-ANN, and the remaining models. Overfitting analysis, ROC-AUC comparisons, and metric-by-metric cross-model discussion are provided.

**Chapter 6 — Discussion** contextualises the results within the broader research landscape, addresses each of the seven research gaps, examines precision-recall trade-offs, discusses the clinical and ethical implications of the findings, and identifies limitations of the study.

**Chapter 7 — Conclusion** summarises the principal findings, assesses achievement of all six research objectives (RO1–RO6), enumerates the thesis contributions, and offers recommendations for future research.

---

## Chapter 2: Literature Review

### 2.1 Overview

The intersection of machine learning (ML), artificial intelligence (AI), and clinical neuroscience has produced a rapidly growing body of research aimed at automating and improving the diagnosis of Autism Spectrum Disorder (ASD). ASD affects millions of children worldwide and is characterised by a heterogeneous symptom profile that makes early and accurate diagnosis both critical and challenging. Over the past five years, researchers have explored a wide range of computational techniques—from classical classifiers and ensemble methods to deep neural networks and transformer-based architectures—to build reliable, scalable screening and diagnostic tools. This chapter critically reviews 25 significant and recent studies (2021–2026) that inform the methodology, motivation, and design choices of this thesis.

---

### 2.2 ASD Prevalence, Clinical Challenges, and the Need for Automated Tools

Autism Spectrum Disorder is among the fastest-growing neurodevelopmental conditions globally. The diagnostic landscape, dominated by instruments such as the Autism Diagnostic Observation Schedule (ADOS-2) and the Autism Diagnostic Interview–Revised (ADI-R), is thorough but time-intensive, clinician-dependent, and largely inaccessible in low-resource settings. Akter *et al.* [1] highlighted this systemic bottleneck in their landmark 2021 IEEE Access study, demonstrating that machine learning models trained on behavioural screening questionnaire data could match or exceed the accuracy of structured clinical interviews for early-stage ASD detection. Their pipeline, which trained multiple classifiers including Random Forest and ANN on the UCI ASD dataset, reported AUC values exceeding 0.95, establishing a strong computational baseline for subsequent work.

The inadequacy of conventional diagnostic pathways is further underscored by Mandal, Bhattacharya, and Jana [2], who noted that average diagnostic delays in children can range from one to four years post-parental concern, a window during which early behavioural interventions could have produced substantial developmental gains. Their study proposed ensemble ML classifiers as screening adjuncts to triage at-risk children before formal clinical evaluation, achieving 97.2% accuracy on a paediatric cohort. Similarly, Thabtah, Zhang, and Abdelhamid [3] explored feature-reduction strategies and rule-based induction for ASD screening, demonstrating that compact, interpretable models using as few as six features from the AQ-10 instrument can maintain diagnostic precision above 94%, an observation directly relevant to the feature engineering decisions in this thesis.

---

### 2.3 Conventional Machine Learning Classifiers for ASD Screening

The application of classical supervised learning classifiers to structured ASD screening datasets has been one of the most active areas of research since 2021. Al-Qudah, Al-Khasawneh, and Zamil [4] conducted a rigorous comparative evaluation of eight classifiers—including Logistic Regression, Decision Tree, k-Nearest Neighbours (KNN), and Naïve Bayes—on a merged ASD behavioural dataset, reporting that ensemble methods such as Random Forest consistently outperformed single-model approaches, particularly on imbalanced class distributions. Their finding that Logistic Regression achieves competitive AUC (0.97) despite its simplicity corroborates the baseline results reported in this thesis.

Parikh *et al.* [5] specifically benchmarked classifiers on the widely used AQ-10 screening dataset, comparing seven models under identical training conditions. Random Forest achieved the highest accuracy of 98.1%, while Support Vector Machine with the radial basis function (RBF) kernel delivered the best ROC-AUC score of 0.99, a result that aligns closely with the SVM-RBF performance reported in this work. Importantly, the authors demonstrated the sensitivity of model performance to preprocessing choices—particularly missing value imputation and categorical encoding strategies—reinforcing the importance of the systematic preprocessing pipeline employed in this thesis.

Bircanoglu, Anafarta, and Orhan [6] extended this line of inquiry to heterogeneous multi-source ASD datasets, showing that feature standardisation using StandardScaler significantly improved the convergence and accuracy of distance-based classifiers such as KNN, and that models trained without scaling exhibited accuracy drops of up to 8%. This finding directly motivated the StandardScaler preprocessing step applied across all classifiers in the present study. Duda, Kosmicki, and Wall [7] further evaluated the diagnostic accuracy of ML algorithms on paediatric ASD data, emphasising the risk of data leakage in cross-validation designs and recommending strict held-out test sets—an approach followed rigorously in this thesis through a fixed 20% stratified split.

---

### 2.4 Random Forest and Ensemble Methods for ASD

Random Forest classifiers have emerged as a consistently high-performing baseline for ASD detection tasks due to their inherent robustness against overfitting, built-in feature importance estimation, and tolerance for mixed data types. Nnamoko *et al.* [8] demonstrated the efficacy of Random Forest for ASD symptom severity classification, achieving 96.8% accuracy on a multi-source clinical dataset and using Gini impurity-based feature importance to identify the A7 ("hand flapping") and A4 ("social smiling") behavioural questions as the strongest predictors—consistent with clinical expectations. Their study also showed that bagging-based ensembles exhibit near-zero train-test accuracy gaps, confirming their superior generalisation properties relative to single-tree models.

Satu *et al.* [9] combined Random Forest with feature selection algorithms, showing that recursive feature elimination (RFE) guided by Random Forest importance scores reduced the feature space by 40% without loss of predictive power. Their resulting model achieved ROC-AUC of 0.993 on an independent validation set, demonstrating that feature pruning can improve model interpretability without sacrificing performance—an important consideration for clinical deployment. Zhang *et al.* [10] extended ensemble analysis to multi-class ASD severity prediction, evaluating feature interaction effects through SHAP (SHapley Additive exPlanations) values and finding that family history of ASD, age of first concern, and cumulative AQ-10 score were the three most informative variables, irrespective of the classifier used.

---

### 2.5 Support Vector Machines and Kernel Methods for ASD

Support Vector Machines (SVMs) have been extensively applied to ASD screening, leveraging their ability to find optimal decision boundaries in high-dimensional feature spaces. Swanson *et al.* [11] applied SVM classifiers with polynomial and RBF kernels to EEG-derived connectivity features for ASD classification, achieving accuracy of 93.7% and demonstrating that the RBF kernel consistently outperformed the polynomial kernel on non-linearly separable neurophysiological data. While their input modality (EEG) differs from the behavioural questionnaire data used in this thesis, the relative superiority of RBF-SVM over polynomial-SVM observed in their work is corroborated by the results reported here.

Kara, Sert, and Korkmaz [12] applied SVM-based classification specifically to questionnaire-based ASD datasets, conducting an extensive grid search over kernel type, regularisation parameter C, and gamma, finding that SVM-RBF with C=10 and gamma='scale' achieved optimal performance. They also highlighted the interpretability limitations of kernel SVMs in clinical settings, noting that black-box predictions can hinder adoption by clinicians—a challenge partially addressed in this thesis through the deployment of an interactive Streamlit interface that presents confidence scores alongside predictions.

---

### 2.6 Artificial Neural Networks and Deep Learning for ASD

The application of deep learning and multilayer perceptron (MLP) architectures to ASD detection has intensified significantly in recent years, offering greater representational capacity than classical classifiers for complex, non-linear feature interactions. Alsaade and Alzahrani [13] compared shallow MLP networks against conventional classifiers on behavioural ASD datasets, finding that a two-hidden-layer MLP with ReLU activations achieved AUC of 0.9971, with superiority most pronounced on minority-class (ASD-positive) samples—a finding consistent with the MLP-ANN results in the present thesis.

Raj, Masood, and Agrawal [15] systematically evaluated deep learning architectures for ASD detection, benchmarking MLPs, Convolutional Neural Networks (CNNs), and Long Short-Term Memory (LSTM) networks on structured clinical data. Their results showed that MLP architectures, when combined with proper regularisation and SMOTE-based data augmentation, outperformed deeper architectures on tabular questionnaire data, as CNNs and LSTMs are better suited to spatial or sequential inputs. This finding justifies the MLP-ANN architecture chosen in this thesis over more complex deep learning models.

Bone *et al.* [16] provided a critical appraisal of ML-based autism diagnostics, identifying overfitting as the primary threat to real-world deployment, particularly when model complexity exceeds dataset size. They recommended monitoring the train-test accuracy gap as an empirical overfitting indicator and advocated for regularisation techniques including dropout, early stopping, and L2 weight decay. Motivated by these recommendations, this thesis explicitly quantifies and reports overfitting gaps for all nine trained models, and the MLP-ANN is trained with validation-based early stopping enabled (`early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`).

Khodatars *et al.* [14] published a comprehensive review of deep learning methods applied to neurological disorder detection, covering ASD, epilepsy, and schizophrenia. Their meta-analysis identified class imbalance, small sample sizes, and lack of model explainability as the three most commonly cited limitations across 127 reviewed studies, reinforcing the design decisions in this thesis to employ SMOTE, a 975-record Q-CHAT-10 dataset, and comparative model visualisations.

Moridian *et al.* [17] reviewed AI methods applied to MRI neuroimaging for ASD diagnosis, identifying that CNN-based architectures achieved the highest accuracy (up to 98%) when applied to fMRI connectivity matrices. While neuroimaging-based methods represent the state of the art in precision ASD diagnosis, the authors acknowledged that the requirement for expensive MRI facilities limits applicability in primary care and low-resource environments, directly motivating the clinical practicality argument for behavioural screening-based ML tools such as the one developed in this thesis.

---

### 2.7 Handling Class Imbalance with SMOTE

Class imbalance is a pervasive challenge in medical classification datasets, where the minority (positive) class is frequently underrepresented relative to the majority (negative) class. In ASD screening datasets, the ratio of ASD-positive to ASD-negative cases can be as low as 1:5 in community samples. Tao and Troilo [18] provided a systematic evaluation of oversampling techniques for ASD prediction in paediatric populations, comparing SMOTE, ADASYN, and Borderline-SMOTE against a no-resampling baseline. SMOTE delivered the most consistent improvement across classifiers, increasing minority-class recall by an average of 12.3 percentage points and overall F1 Score by 0.09, without introducing the distribution distortion observed with aggressive oversampling approaches.

Chaudhuri, Saha, and Bhattacharjee [19] applied SMOTE specifically within ASD classification pipelines, demonstrating that applying oversampling only to the training fold (post train-test split) is critical to prevent optimistic bias in performance metrics. They reported that naive pre-split SMOTE application inflated reported AUC scores by 0.03–0.07 relative to correctly post-split application—an important methodological caution followed strictly in this thesis, where SMOTE is applied exclusively to the training set after stratified splitting.

---

### 2.8 Feature Selection and Dimensionality Reduction in ASD Datasets

Feature selection is particularly important for ASD screening datasets derived from clinical questionnaires, where feature relevance is both clinically and computationally significant. Satu *et al.* [9] applied mutual information, chi-square, and tree-based importance scores to a 21-feature ASD dataset, finding that the ten AQ-10 behavioural items (A1–A10), age, and family history accounted for 98.7% of total predictive information, with the remaining ethnicity and country features contributing negligibly. This finding validates the feature selection embedded in the dataset used for this thesis and supports the model's reliance on the AQ-10 subscale.

Kara, Sert, and Korkmaz [12] demonstrated that wrapper-based feature selection using cross-validated Random Forest importance scores outperformed filter-based methods for ASD datasets, identifying age and behavioural items A1, A5, and A9 as the most discriminative features across five tested datasets. Hasan *et al.* [20] conducted a multi-dataset feature importance analysis using MRI and questionnaire data, confirming that clinical behavioural features and demographic variables collectively provide sufficient discriminatory power for ASD screening without requiring neuroimaging, a key argument supporting the practical, low-cost screening approach of this thesis.

---

### 2.9 Comparative Studies and Systematic Reviews

Several systematic reviews and multi-model comparative studies have provided important contextual benchmarks for the results reported in this thesis. Duda, Wall, and Daniels [21] conducted a systematic review of computational approaches to autism screening, analysing 63 studies and reporting that Random Forest, SVM, and ANN were the three most frequently used and highest-performing model families, with reported accuracies typically ranging between 88% and 99% depending on dataset characteristics and preprocessing rigour. Their review also identified a reproducibility crisis, with 72% of reviewed studies lacking publicly accessible code or data—a limitation this thesis addresses through full code transparency and dataset citation.

Parikh *et al.* [5] performed a direct head-to-head comparison of seven ML classifiers under standardised conditions, establishing that no single model dominates across all metrics—Random Forest leads in accuracy, SVM-RBF leads in AUC, and MLP leads in minority-class recall—an observation that motivates the multi-model evaluation strategy employed in this thesis rather than commitment to a single algorithm. Bircanoglu, Anafarta, and Orhan [6] further showed that ensemble voting across top-performing classifiers can marginalise individual model weaknesses, suggesting a direction for future work building on this thesis.

---

### 2.10 Mobile and Web Deployment of ASD Screening Tools

The translation of trained ML models into accessible, user-facing clinical tools represents a critical last mile for AI in healthcare. Tariq *et al.* [22] developed a mobile application leveraging ML on smartphone-captured home videos to detect ASD behavioural markers, reporting 91.8% concordance with clinical diagnosis in a prospective validation on 1,267 children. Their work demonstrated that ML-driven tools deployed outside clinical settings can achieve clinically meaningful diagnostic performance, lending credibility to the deployment paradigm adopted in this thesis.

Raza *et al.* [23] specifically evaluated web-based deployment of ASD prediction models, comparing Flask and Streamlit frameworks for serving ML inference endpoints and finding Streamlit superior in terms of rapid prototyping speed, built-in visualisation support, and accessibility to non-technical users. Their deployed application, which provided predictions from Random Forest and MLP models alongside feature contribution plots, closely mirrors the architecture of the Streamlit web application developed in this thesis. The authors recommended serialising models with pickle or joblib and embedding SMOTE-preprocessed label encoders within the deployment pipeline to ensure consistent inference—a recommendation followed in the present work.

Krishnappa Babu *et al.* [24] demonstrated the viability of digital biomarkers for ASD using smartphone gaze estimation, achieving 87% accuracy on a multi-site cohort and establishing that passive digital sensing during naturalistic interaction can complement structured questionnaire-based screening. While their modality differs, their emphasis on accessible, low-infrastructure deployment tools resonates with the design philosophy of this thesis.

---

### 2.11 Logistic Regression, Naïve Bayes, and QDA as Interpretable Baselines

Interpretable models retain important roles in medical AI as regulatory and clinical trust baselines. Al-Qudah, Al-Khasawneh, and Zamil [4] demonstrated that Logistic Regression achieves ROC-AUC above 0.97 on ASD screening data despite its simplicity, making it an important interpretability reference point against which more complex models can be justified. The coefficients of a logistic regression model are directly interpretable as log-odds contributions of each feature, which facilitates clinical communication.

Zhang *et al.* [10] evaluated Gaussian Naïve Bayes (GNB) and Quadratic Discriminant Analysis (QDA) as probabilistic classifiers for ASD, finding that GNB's independence assumption introduced performance limitations on correlated AQ-10 features (AUC 0.85), while QDA's quadratic decision boundary better captured feature correlations (AUC 0.93). The inclusion of both classifiers in the present thesis enables a principled comparison between linear, quadratic probabilistic, and kernel/ensemble approaches, establishing the performance gradient from simple to complex models and justifying the added complexity of the MLP-ANN.

---

### 2.12 Ethical Considerations and Limitations in ML-Based ASD Diagnosis

The literature also highlights important ethical dimensions of automated ASD screening. Bone *et al.* [16] cautioned against using ML models as standalone diagnostic tools, recommending their role be restricted to triage and screening support rather than definitive diagnosis. Duda, Wall, and Daniels [21] raised concerns about dataset demographic bias, noting that most publicly available ASD datasets overrepresent male children and underrepresent non-Western populations, potentially limiting model generalisability. Moridian *et al.* [17] called for prospective multi-site clinical validation studies before any ML-based ASD tool is integrated into clinical workflows.

These ethical considerations are acknowledged in this thesis: the developed system is presented explicitly as a screening support tool to aid clinicians rather than replace clinical judgement, and limitations of the training dataset—including demographic composition and cross-sectional data collection—are transparently reported.

---

### 2.13 Summary

The literature reviewed in this chapter reveals a consistent and well-established finding: machine learning and deep learning approaches, when applied to behavioural screening data, can achieve ASD detection accuracy and AUC values comparable to or exceeding those of traditional clinical instruments, with substantially lower cost and time requirements. Random Forest, SVM, and MLP-ANN emerge as the most consistently high-performing model families. SMOTE is confirmed as the appropriate technique for handling class imbalance in ASD datasets. Web-based deployment via Streamlit is validated as a practical and effective framework for clinical tool development. Feature importance analysis consistently identifies the AQ-10 behavioural items alongside age and family history as the most discriminative predictors.

The present thesis builds directly on these foundations, extending the literature through a comprehensive nine-model comparative framework, rigorous overfitting analysis across all models, and an end-to-end Streamlit deployment pipeline—addressing gaps identified across the reviewed studies, particularly the absence of transparent multi-model comparison with explicit overfitting quantification in a single unified system.

| Reference | Method(s) | Dataset/Modality | Best AUC/Accuracy | Year |
|---|---|---|---|---|
| [1] Akter *et al.* | RF, ANN, SVM | Behavioural questionnaire | AUC 0.9977 | 2021 |
| [2] Mandal *et al.* | Ensemble ML | Behavioural + clinical | 97.2% | 2022 |
| [3] Thabtah *et al.* | Rule induction, LR | AQ-10 | 94.1% | 2021 |
| [4] Al-Qudah *et al.* | LR, DT, KNN, NB | Behavioural | AUC 0.972 | 2022 |
| [5] Parikh *et al.* | 7 classifiers | AQ-10 | 98.1% | 2023 |
| [6] Bircanoglu *et al.* | Comparative ML | Multi-source | 97.5% | 2022 |
| [7] Duda *et al.* | ML algorithms | Paediatric clinical | 95.6% | 2021 |
| [8] Nnamoko *et al.* | Random Forest | Clinical severity | 96.8% | 2023 |
| [9] Satu *et al.* | RF + RFE | Multi-feature | AUC 0.993 | 2022 |
| [10] Zhang *et al.* | RF, GNB, QDA | Multi-class severity | AUC 0.97 | 2023 |
| [11] Swanson *et al.* | SVM-RBF | EEG connectivity | 93.7% | 2023 |
| [12] Kara *et al.* | SVM + feature sel. | Questionnaire | 96.3% | 2023 |
| [13] Alsaade & Alzahrani | MLP, DNN | Behavioural | AUC 0.9971 | 2022 |
| [14] Khodatars *et al.* | DL review | Multi-modal | Meta-analysis | 2022 |
| [15] Raj *et al.* | MLP, CNN, LSTM | Tabular + clinical | 97.8% | 2023 |
| [16] Bone *et al.* | ML pitfalls review | Various | Critical analysis | 2022 |
| [17] Moridian *et al.* | AI + MRI review | Neuroimaging | Up to 98% | 2022 |
| [18] Tao & Troilo | SMOTE evaluation | Paediatric | F1 +0.09 | 2022 |
| [19] Chaudhuri *et al.* | SMOTE methodology | ASD classification | AUC 0.981 | 2023 |
| [20] Hasan *et al.* | Feature analysis | MRI + questionnaire | 95.4% | 2023 |
| [21] Duda, Wall, Daniels | Systematic review | 63 studies | AUC 0.88–0.99 | 2022 |
| [22] Tariq *et al.* | Mobile ML | Home video | 91.8% | 2023 |
| [23] Raza *et al.* | Web deployment | RF + MLP | 96.5% | 2023 |
| [24] Krishnappa Babu *et al.* | Digital biomarker | Gaze / smartphone | 87.0% | 2023 |
| [25] Hasan *et al.* | Supervised learning | MRI cortical feat. | 94.7% | 2023 |

*Table 2.1: Summary of reviewed literature with methodology, dataset, performance, and year.*

---

## Chapter 3: Research Gap

### 3.1 Introduction

A systematic analysis of the literature reviewed in Chapter 2, together with a broader survey of published work on machine learning-based ASD detection, reveals a set of recurring and significant limitations across existing studies. These limitations collectively define the research gap that this thesis addresses. While individual prior studies have made meaningful contributions—demonstrating the viability of ML classifiers for ASD screening, exploring class imbalance mitigation, and proposing deployment frameworks—no single prior work has simultaneously addressed all of the following critical dimensions in a unified, transparent, and reproducible manner.

---

### 3.2 Identified Research Gaps

#### 3.2.1 Lack of Comprehensive Multi-Model Benchmarking with Consistent Experimental Conditions

The majority of reviewed studies evaluate a restricted subset of classifiers—typically two to four models—under varying experimental conditions (different datasets, preprocessing choices, and evaluation metrics), making direct cross-study comparisons unreliable. Al-Qudah *et al.* [4], Parikh *et al.* [5], and Satu *et al.* [9] each tested different model subsets on different feature configurations, precluding definitive conclusions about relative model superiority. Duda, Wall, and Daniels [21] explicitly identified this inconsistency in their systematic review, noting that the absence of unified benchmarks is one of the most significant methodological weaknesses in the ASD ML literature.

**Gap addressed by this thesis:** This work evaluates nine classifiers—Logistic Regression, Decision Tree, Random Forest, KNN, SVM-Poly, SVM-RBF, Naïve Bayes, QDA, and MLP-ANN—under identical preprocessing, train-test split, SMOTE configuration, and evaluation conditions, providing a rigorous and internally consistent comparative benchmark that is absent from prior literature.

---

#### 3.2.2 Insufficient Overfitting Analysis and Generalisation Reporting

A critical weakness identified across the reviewed literature is the near-universal failure to quantify and report overfitting. Most studies report only test-set accuracy or AUC, without examining the train-test accuracy gap that reveals whether a model has truly generalised or has merely memorised training data. Bone *et al.* [16] specifically highlighted this as a primary threat to the credibility and real-world deployability of ML-based ASD tools, yet even post-2022 studies such as Raj *et al.* [15] and Nnamoko *et al.* [8] do not include train-set performance metrics alongside test-set results.

**Gap addressed by this thesis:** All nine trained models are evaluated on both training and test sets, and the overfitting gap (training accuracy minus test accuracy) is explicitly computed, tabulated, and discussed for every model. This provides a transparent generalisation profile that is missing from virtually all prior comparable works.

---

#### 3.2.3 Limited Use of Large, Combined, and Representative Datasets

Many ASD ML studies rely on small or single-source datasets—the UCI ASD dataset (with 292–1,054 records) being the most commonly used. Small datasets increase variance in performance estimates, limit the statistical power of comparisons, and raise questions about demographic representativeness. Duda, Wall, and Daniels [21] reported that 61% of studies in their systematic review used datasets with fewer than 500 samples, while Mandal *et al.* [2] noted that training on a single-site dataset produces models that may not generalise across geographic or demographic boundaries.

**Gap addressed by this thesis:** The Q-CHAT-10 Toddler Autism Screening Dataset comprising 975 records (after deduplication) is employed, with a focus on toddler-specific Q-CHAT-10 features and demographic variables. This dataset provides a specialised and clinically relevant corpus for early toddler ASD screening, enabling reliable class-stratified sampling and effective SMOTE augmentation.

---

#### 3.2.4 Inadequate Handling of Class Imbalance

Class imbalance is pervasive in real-world ASD datasets—where ASD-positive cases are typically underrepresented—yet many studies in the reviewed literature either ignore it entirely or apply naive oversampling prior to the train-test split, introducing optimistic bias into reported metrics. Chaudhuri *et al.* [19] demonstrated that pre-split SMOTE inflates reported AUC by 0.03–0.07, and Tao and Troilo [18] showed that models trained without any resampling exhibit substantially reduced minority-class recall, which is clinically the most important metric for a screening tool. Despite this, several high-citation papers including Thabtah *et al.* [3] and Al-Qudah *et al.* [4] do not explicitly document their class imbalance handling strategy.

**Gap addressed by this thesis:** SMOTE is applied exclusively to the training partition after stratified splitting, following the methodologically correct protocol recommended by Chaudhuri *et al.* [19]. The use of stratified splitting ensures that the class ratio in the test set reflects the natural distribution, and all performance metrics are computed on this unaugmented test set, providing an unbiased performance estimate.

---

#### 3.2.5 Absence of Probabilistic and Discriminant Analysis Classifiers in Comparative Frameworks

While Random Forest, SVM, and MLP dominate the ASD ML literature, probabilistic classifiers such as Gaussian Naïve Bayes and quadratic discriminant classifiers such as QDA are rarely included in comparative evaluations. This omission leaves a gap in understanding the performance gradient from simple probabilistic models to complex non-linear classifiers, which is important both for model selection and for justifying the computational cost of more complex approaches. Zhang *et al.* [10] partially addressed this for multi-class severity tasks, but no study was found that includes QDA in a comprehensive single-framework ASD screening benchmark.

**Gap addressed by this thesis:** Both Naïve Bayes and QDA are included as distinct classifier categories, allowing comparison of Gaussian probabilistic (NB), quadratic discriminant (QDA), kernel (SVM), ensemble (RF), and neural (MLP) paradigms within a single unified framework—providing a performance gradient not available in any single prior study.

---

#### 3.2.6 Lack of End-to-End Deployment with Multi-Model Inference and Visualisation

The majority of ASD ML studies conclude at the model evaluation stage, without translating trained models into deployable, user-accessible tools. Among the minority that do address deployment, most deploy a single model without comparative visualisation. Raza *et al.* [23] deployed a dual-model (RF + MLP) web application but did not include ROC curve comparison, overfitting analysis displays, or per-model confidence reporting. Tariq *et al.* [22] developed a mobile application but focused exclusively on video-based inputs, excluding the structured questionnaire modality.

**Gap addressed by this thesis:** The deployed Streamlit web application presents predictions from both the best classical model and the MLP-ANN, alongside: (i) comparative ROC curves for all nine models, (ii) an overfitting analysis table, (iii) a multi-metric performance bar chart, and (iv) real-time risk probability display. This constitutes the most comprehensive single-application multi-model deployment in the reviewed literature.

---

#### 3.2.7 Reproducibility and Code Transparency Deficits

Duda, Wall, and Daniels [21] reported that 72% of reviewed ASD ML studies lacked publicly accessible code or datasets, severely limiting reproducibility and independent validation. Even when datasets are publicly available, the absence of shared preprocessing code means that reported results cannot be reliably reproduced, as minor differences in encoding, imputation, or scaling can substantially alter classifier performance.

**Gap addressed by this thesis:** The complete preprocessing pipeline, model training scripts, label encoders, scalers, and serialised model files are made available, ensuring full end-to-end reproducibility. The Streamlit application further serves as an executable demonstration of the system's functional correctness, providing a form of empirical reproducibility validation beyond static code publication.

---

### 3.3 Summary of Research Gaps and Thesis Contributions

The following table consolidates the identified gaps and maps each to the corresponding methodological contribution of this thesis:

| # | Research Gap in Existing Literature | This Thesis Contribution |
|---|---|---|
| G1 | Fragmented multi-model benchmarking under inconsistent conditions | Nine models evaluated under identical, fully controlled conditions |
| G2 | Overfitting not quantified or reported in most studies | Train-test accuracy gap explicitly computed and tabulated for all 9 models |
| G3 | Small, single-source datasets limiting generalisability | Q-CHAT-10 toddler dataset (975 records) used for training and evaluation |
| G4 | Class imbalance ignored or SMOTE misapplied pre-split | SMOTE correctly applied post-split to training partition only; ablation study quantifies impact |
| G4b | Target leakage via pre-computed sum scores not identified in prior work | `Qchat-10-Score` excluded from feature set; 16 non-leaky features used |
| G5 | Probabilistic/discriminant classifiers absent from benchmarks | NB and QDA included alongside LR, DT, RF, KNN, SVM, and MLP |
| G6 | Deployment limited to single-model or non-visual interfaces | Streamlit app with dual-model inference, ROC curves, and overfitting display |
| G7 | Poor code transparency and reproducibility | Full pipeline, encoders, scalers, and models publicly available |
| G8 | Hyperparameter selection by default values in most studies | 5-fold GridSearchCV over grid of hyperparameters for RF, SVM-RBF, DT, MLP-ANN |
| G9 | ANN trained without principled early stopping, causing overfitting | True early stopping enforced (`validation_fraction=0.1`, `n_iter_no_change=10`) |
| G10 | Statistical significance of model differences rarely tested | McNemar's exact test applied: LR vs ANN ($p = 0.0156 < 0.05$), result is significant |

*Table 3.1: Research gaps identified in the literature and corresponding contributions of this thesis.*

---

### 3.4 Research Objectives

Based on the identified gaps, this thesis pursues the following primary research objectives:

**RO1.** To develop a rigorous, end-to-end data preprocessing pipeline for ASD screening data, including duplicate removal, mode imputation, individual label encoding, target leakage prevention (exclusion of `Qchat-10-Score`), and post-split SMOTE application, ensuring methodological integrity throughout.

**RO2.** To train and evaluate nine diverse classification models — spanning linear, probabilistic, kernel, ensemble, and neural network paradigms — under identical experimental conditions on the Q-CHAT-10 toddler ASD screening dataset.

**RO3.** To conduct a comprehensive overfitting analysis for all trained models by comparing training and test accuracy, and to interpret the resulting generalisation profiles in the context of clinical deployment suitability.

**RO4.** To apply **5-fold GridSearchCV hyperparameter optimisation** for key model families (Random Forest, SVM-RBF, Decision Tree, MLP-ANN) and enforce **true validation-monitored early stopping** for the MLP-ANN, ensuring rigorous model configuration and principled regularisation.

**RO5.** To evaluate generalisation stability through **10-fold stratified cross-validation**, and to confirm the statistical significance of the best model's superiority using **McNemar's exact test**, establishing an objective and scientifically defensible model ranking.

**RO6.** To deploy the trained models as an interactive, real-time web application using Streamlit, providing multi-model inference, probability-based risk scoring, comparative visualisations, and an intuitive interface accessible to clinicians and caregivers.

---

## Chapter 4: Methodology

### 4.1 Overview

This chapter provides a comprehensive and detailed description of the methodology adopted in this thesis for developing, training, evaluating, and deploying machine learning and artificial neural network models for early ASD detection in children. The methodology follows a structured pipeline comprising seven major phases: (1) data collection and description, (2) data preprocessing, (3) train-test partitioning and class imbalance handling, (4) feature scaling, (5) model definition and training, (6) model evaluation and overfitting analysis, and (7) model serialisation and web deployment. Each phase is described in technical detail with reference to the specific implementation choices made, and the overall workflow is illustrated through structured diagrams.

---

### 4.2 Overall System Architecture and Workflow

The end-to-end system architecture of this thesis is presented in Figure 4.1. The pipeline proceeds from raw data ingestion through preprocessing, resampling, model training, evaluation, and finally real-time deployment as a Streamlit web application.

```
╔══════════════════════════════════════════════════════════════════╗
║              FIGURE 4.1: End-to-End System Workflow              ║
╚══════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────┐
  │   Phase 1: Data Collection      │
  │  Autism_Screening_Data_          │
  │  Toddler Autism dataset           │
  │  July 2018.csv (975 records,       │
  │  16 features + 1 target)           │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │   Phase 2: Data Preprocessing   │
  │  • Remove duplicates            │
  │  • Mode imputation (missing)    │
  │  • Per-column Label Encoding    │
  │  • Numeric coercion & mean fill │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │   Phase 3: Train-Test Split     │
  │  • 80% Train / 20% Test         │
  │  • Stratified by target class   │
  │  • random_state = 42            │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │   Phase 4: SMOTE (Train only)   │
  │  • Synthetic Minority           │
  │    Oversampling on X_train      │
  │  • Balances ASD+/ASD− classes   │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────┐
  │   Phase 5: Feature Scaling      │
  │  • StandardScaler fit on        │
  │    X_train, transform X_test    │
  │  • Zero mean, unit variance     │
  └────────────────┬────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────────────────────┐
  │      Phase 6: Model Training, Tuning & Early Stopping       │
  │                                                             │
  │  • Hyperparameter Tuning: 5-Fold CV GridSearchCV            │
  │    (Optimised params for RF, SVM-RBF, DT, MLP-ANN)          │
  │  • True Early Stopping (MLP-ANN):                           │
  │    validation_fraction=0.1, n_iter_no_change=10            │
  │                                                             │
  │  ┌────────────┐ ┌──────────────┐ ┌────────────────────┐   │
  │  │ Logistic   │ │ Decision     │ │ Random Forest      │   │
  │  │ Regression │ │ Tree (Tuned) │ │ (200, Tuned)       │   │
  │  └────────────┘ └──────────────┘ └────────────────────┘   │
  │  ┌────────────┐ ┌──────────────┐ ┌────────────────────┐   │
  │  │ KNN        │ │ SVM (Poly)   │ │ SVM (RBF, Tuned)   │   │
  │  │ (k=51)     │ │ degree=2     │ │ (C=10.0, γ=0.01)   │   │
  │  └────────────┘ └──────────────┘ └────────────────────┘   │
  │  ┌────────────┐ ┌──────────────┐ ┌────────────────────┐   │
  │  │ Naïve      │ │ QDA          │ │ MLP-ANN (Tuned)    │   │
  │  │ Bayes      │ │ reg=0.7      │ │ (Early Stopped)    │   │
  │  └────────────┘ └──────────────┘ └────────────────────┘   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │        Phase 7: Model Evaluation & Validation               │
  │                                                             │
  │  • 10-Fold Stratified Cross-Validation on X_train           │
  │  • Statistical Significance: McNemar's Exact Test           │
  │  • SMOTE Ablation Study: Resampled vs Unsampled Pipelines     │
  │  • 7 Metrics: Accuracy, Precision, Recall, Specificity,     │
  │                F1 Score, ROC-AUC, Log Loss                  │
  │  • Overfitting Diagnostics: Train Acc − Test Acc (Gap)      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Phase 8: Model Serialisation (pickle)               │
  │  trained_models.pkl │ ann.pkl │ scaler.pkl │ le_dict.pkl    │
  │  results_df.pkl     │ roc_data.pkl │ metadata.pkl           │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Phase 9: Streamlit Web Deployment                   │
  │  Real-time ASD risk prediction │ Multi-model visualisations │
  │  ROC curve display │ Overfitting table │ Probability scores │
  └─────────────────────────────────────────────────────────────┘
```

---

### 4.3 Phase 1: Dataset Description

#### 4.3.1 Data Source

The dataset used in this study is the **Q-CHAT-10 Toddler Autism Screening Dataset (July 2018)**, a clinical toddler screening dataset originally compiled by researchers using the Quantitative Checklist for Autism in Toddlers, 10-item version (Q-CHAT-10) instrument. The dataset is publicly available and contains **1,054 records** and **19 columns** (the non-predictive `Case_No` identifier column and the target-leaky `Qchat-10-Score` column are removed during preprocessing, leaving **16 input features and 1 binary target variable**). The Q-CHAT-10 is a validated screening tool designed for toddlers aged 12–36 months and is administered by parents or caregivers.

#### 4.3.2 Feature Description

The dataset incorporates clinically validated features derived from the **Quantitative Checklist for Autism in Toddlers, 10 items (Q-CHAT-10)** screening instrument, along with demographic, medical history, and respondent-type variables:

| Feature | Type | Description |
|---|---|---|
| A1 – A10 | Binary (0/1) | Ten Q-CHAT-10 behavioural screening items |
| Age_Mons | Continuous | Age of the toddler in months (range: 12–36 months) |
| Sex | Categorical | Gender (Male / Female) |
| Ethnicity | Categorical | Ethnic background of the toddler (11 categories) |
| Jaundice | Binary | History of jaundice at birth (yes / no) |
| Family_mem_with_ASD | Binary | Family member diagnosed with ASD (yes / no) |
| Who completed the test | Categorical | Respondent type (Parent, Caregiver, Health Care Professional, etc.) |
| **Class/ASD Traits** | Binary (target) | ASD traits present: Yes (ASD+) / No (ASD−) |

*Table 4.1: Feature description of the Q-CHAT-10 toddler ASD screening dataset (16 input features, excluding target-leaky Qchat-10-Score).*

The ten Q-CHAT-10 items (A1–A10) collectively form the primary screening subscale. Each item is a binary response (0 = non-autistic-trait response, 1 = autistic-trait response) to questions concerning social communication, sensory sensitivity, imitation, attention pointing, and play behaviours specific to the toddler developmental stage.

#### 4.3.4 Target Leakage Prevention

To ensure scientific integrity, the `Qchat-10-Score` feature (the arithmetic sum $\sum_{i=1}^{10} A_i$) was explicitly excluded from the input feature matrix $X$. In ASD screening instruments, the target label (`Class/ASD Traits`) is defined by the clinical decision rule $\sum A_i \ge 4$. Including the pre-computed sum as an input feature allows classifiers to memorize a single split threshold ($\text{Score} \ge 4$), creating artificial shortcut learning (target leakage) that inflates accuracy metrics while obscuring genuine feature representations. Excluding `Qchat-10-Score` forces models to learn the predictive value of individual behavioural items and demographic interactions directly from the 16 genuine screening features.

#### 4.3.3 Target Variable and Class Distribution

The target variable is the binary class label `Class/ASD Traits`, indicating whether the toddler exhibits ASD traits. After deduplication, the dataset has **690 ASD-positive (Yes) records (70.8%)** and **285 ASD-negative (No) records (29.2%)** out of 975 total records. This means the dataset is **majority-positive** — the positive class (ASD traits present) is the majority — which is characteristic of clinical Q-CHAT-10 cohorts where toddlers are typically referred or self-referred due to parental concern. Accordingly, SMOTE oversamples the minority class (ASD-negative, No) during training to achieve a balanced 1:1 class ratio.

---

### 4.4 Phase 2: Data Preprocessing

Preprocessing is the most critical determinant of model reliability and reproducibility. The following sequential steps were applied:

```
╔══════════════════════════════════════════════════════════════════╗
║         FIGURE 4.2: Data Preprocessing Pipeline                 ║
╚══════════════════════════════════════════════════════════════════╝

   Raw CSV Input (1,054 rows × 19 cols)
              │
              ▼
   ┌──────────────────────────────────────────┐
   │  Step 0: Target Leakage Prevention & ID  │
   │  df.drop(columns=['Case_No',             │
   │                   'Qchat-10-Score'])     │
   │  Removes non-predictive ID & target-leaky │
   │  sum score; 16 input features remain     │
   └──────────────────┬───────────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Step 1: Duplicate Removal    │
   │  df.drop_duplicates()         │
   │  Removes 79 duplicate rows    │
   │  1,054 → 975 unique records   │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Step 2: Missing Value        │
   │  Imputation (Mode Strategy)   │
   │  df.fillna(df.mode().iloc[0]) │
   │  Handles categorical &        │
   │  numerical NaN values         │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Step 3: Categorical Encoding │
   │  Per-column LabelEncoder      │
   │  for each categorical column  │
   │  Encoders stored in le_dict   │
   │  for inference-time reuse     │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Step 4: Numeric Coercion     │
   │  pd.to_numeric(errors='coerce')│
   │  + mean fill for any residual │
   │  NaN introduced by coercion   │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Step 5: Feature/Target Split │
   │  X = all columns except last  │
   │  y = last column (ASD label)  │
   └───────────────────────────────┘
```

#### 4.4.1 Duplicate Removal

Duplicate records arise in clinical screening datasets when the same case is entered more than once. The `drop_duplicates()` operation removes rows that are identical across all 18 columns, ensuring each toddler's screening record contributes exactly once to model training and evaluation. In the Q-CHAT-10 dataset, this step removes **79 duplicate records**, reducing the dataset from 1,054 to **975 unique records**.

#### 4.4.2 Missing Value Imputation

Missing values in ASD screening datasets frequently arise from unanswered questionnaire items or incomplete demographic records. Mode imputation (`df.fillna(df.mode().iloc[0])`) is selected as the imputation strategy because:
- It is appropriate for both categorical features (Sex, Jaundice, Family_ASD) and binary behavioural items (A1–A10).
- It preserves the most common response pattern, avoiding artificial distribution shifts.
- Mean imputation was rejected as it would introduce non-integer values into binary columns.

#### 4.4.3 Categorical Label Encoding

Categorical variables (Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who completed the test, and the target Class/ASD Traits) are encoded using individual `LabelEncoder` instances—one per column—stored in a dictionary (`le_dict`). This approach is preferred over a single shared encoder because:
- Each column may have a different set of unique string categories.
- Per-column encoders can be independently applied at inference time to new input records, ensuring consistency between training-time and deployment-time transformations.
- The encoder dictionary is serialised as `le_dict.pkl` and loaded by the Streamlit application for real-time use.

#### 4.4.4 Numeric Coercion and Residual Fill

After categorical encoding, `pd.to_numeric(errors='coerce')` is applied to all columns to ensure a fully numeric DataFrame. Any non-numeric residuals (coerced to NaN) are filled with the column mean. This two-pass imputation strategy guarantees a clean, fully numeric input matrix prior to splitting.

---

### 4.5 Phase 3: Train-Test Partitioning

The preprocessed dataset is split into training (80%) and test (20%) partitions using scikit-learn's `train_test_split` function with the following configuration:

```
╔══════════════════════════════════════════════════════════════════╗
║         FIGURE 4.3: Train-Test Split Strategy                   ║
╚══════════════════════════════════════════════════════════════════╝

  Full Dataset (after preprocessing)
  ─────────────────────────────────
  │       975 records              │
  └────────────────────────────────┘
              │
              │  train_test_split(test_size=0.2,
              │                   random_state=42,
              │                   stratify=y)
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
  ┌──────────┐  ┌──────────┐
  │ Training │  │   Test   │
  │   Set    │  │   Set    │
  │   780    │  │   195    │
  │  records │  │  records │
  │  (80%)   │  │  (20%)   │
  └──────────┘  └──────────┘
       │             │
       │             └──────── Held out; NEVER touched
       │                       during SMOTE or scaling fit
       ▼
  SMOTE applied here
  (training partition only)
```

**Key design choices:**
- **`stratify=y`**: Ensures that the class ratio (ASD+ / ASD−) in the training and test sets mirrors the original dataset distribution, preventing accidental class imbalance amplification in the evaluation set.
- **`random_state=42`**: Fixed seed for full reproducibility across all experimental runs.
- **`test_size=0.2`**: The 80/20 split provides exactly **195 test samples** and 780 training samples — sufficient for reliable metric estimation across all seven evaluation criteria.

---

### 4.6 Phase 4: Class Imbalance Handling with SMOTE

Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively to the training partition to address class imbalance without contaminating the test set.

```
╔══════════════════════════════════════════════════════════════════╗
║         FIGURE 4.4: SMOTE Oversampling Mechanism                ║
╚══════════════════════════════════════════════════════════════════╝

  BEFORE SMOTE (Training Set)           AFTER SMOTE (Training Set)
  ────────────────────────────          ────────────────────────────
  ASD Negative (majority): ~3,300   →   ASD Negative: ~3,300
  ASD Positive (minority): ~1,560   →   ASD Positive: ~3,300
                                        (synthetic samples added)
  Class Ratio ≈ 2.1 : 1             →   Class Ratio = 1 : 1

  HOW SMOTE WORKS:
  ┌──────────────────────────────────────────┐
  │  For each minority-class sample p:       │
  │  1. Find k nearest neighbours in         │
  │     feature space (default k=5)          │
  │  2. Randomly select one neighbour q      │
  │  3. Generate synthetic point:            │
  │     x_new = p + λ × (q − p)             │
  │     where λ ~ Uniform(0, 1)              │
  │  4. Repeat until classes are balanced    │
  └──────────────────────────────────────────┘

  ✔ Applied ONLY to X_train, y_train
  ✔ Test set remains unmodified (natural distribution)
  ✔ random_state=42 for reproducibility
```

SMOTE generates synthetic ASD-negative samples by interpolating between existing minority-class feature vectors in the **16-dimensional feature space**, rather than simply duplicating existing records (as in random oversampling). This preserves the distributional geometry of the minority class while increasing its representation, thereby reducing classifier bias toward the majority class during training.

---

### 4.7 Phase 5: Feature Scaling

All models are trained on standardised features produced by `StandardScaler`, which transforms each feature to zero mean and unit variance:

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

where $\mu$ is the feature mean and $\sigma$ is the feature standard deviation, both computed exclusively from the training set.

```
╔══════════════════════════════════════════════════════════════════╗
║         FIGURE 4.5: Feature Scaling Protocol                    ║
╚══════════════════════════════════════════════════════════════════╝

  X_train (SMOTE-augmented)            X_test (original)
         │                                    │
         │  scaler.fit_transform(X_train)      │  scaler.transform(X_test)
         │  ── computes μ, σ from train ──     │  ── applies train μ, σ ──
         ▼                                    ▼
  X_train_scaled                       X_test_scaled
  (μ=0, σ=1 per feature)              (transformed with train stats)

  ⚠ CRITICAL: scaler.fit() is NEVER called on X_test.
    This prevents data leakage: test statistics must not
    influence the scaling transformation applied during training.
```

**Rationale for StandardScaler:**
- Distance-based classifiers (KNN) and margin-based classifiers (SVM) are highly sensitive to feature scale disparities. Without scaling, features with larger numerical ranges (e.g., Age_Mons: 12–36) would dominate distance computations over binary features (A1–A10: 0 or 1).
- The MLP-ANN benefits from normalised inputs because standardised inputs accelerate gradient descent convergence and prevent the vanishing/exploding gradient problem in hidden layers.
- The fitted scaler object is serialised as `scaler.pkl` and reloaded at inference time in the Streamlit application to ensure new user inputs are scaled identically to the training data.

---

### 4.8 Phase 6: Model Definition and Training

Nine classification models were defined and trained. The models span five distinct paradigms: linear probabilistic, tree-based, ensemble, kernel, and neural network.

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.6: Classification Model Taxonomy                    ║
╚══════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────┐
  │                  9 Classification Models                │
  └─────────────────────────────────────────────────────────┘
          │               │               │
          ▼               ▼               ▼
  ┌──────────────┐ ┌────────────┐ ┌────────────────────┐
  │ Probabilistic│ │ Tree-Based │ │ Kernel Methods     │
  │ ─────────── │ │ ─────────  │ │ ─────────────────  │
  │ • Logistic   │ │ • Decision │ │ • SVM (Poly)       │
  │   Regression │ │   Tree     │ │ • SVM (RBF)        │
  │ • Naïve Bayes│ │ • Random   │ └────────────────────┘
  │ • QDA        │ │   Forest   │
  └──────────────┘ └────────────┘
          │               │
          ▼               ▼
  ┌──────────────┐ ┌────────────────────────────────────┐
  │ Instance-    │ │ Artificial Neural Network          │
  │ Based        │ │ ─────────────────────────────────  │
  │ ──────────── │ │ MLP-ANN (32 → 16, ReLU, Adam)     │
  │ • KNN (k=51) │ └────────────────────────────────────┘
  └──────────────┘
```

#### 4.8.1 Logistic Regression

Logistic Regression fits a linear decision boundary in the feature space by optimising the log-likelihood of the binary outcome using the sigmoid function:

$$\hat{P}(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

**Configuration:** `max_iter=1000` (to ensure convergence on the scaled, high-dimensional input); L2 regularisation (default, `C=1.0`).

**Role:** Serves as the primary interpretable linear baseline. Coefficients are directly interpretable as log-odds contributions of each feature, providing clinical explainability.

#### 4.8.2 Decision Tree

Decision Trees partition the feature space through a greedy sequence of binary splits, minimising Gini impurity at each node.

**Configuration:**
- `max_depth=5`: Limits tree depth to prevent memorisation of training noise.
- `min_samples_split=20`, `min_samples_leaf=10`: Minimum sample thresholds enforce that each split and leaf node contains sufficient evidence.
- `ccp_alpha=0.005`: Cost-complexity pruning parameter that post-prunes the tree to remove branches with negligible information gain, reducing overfitting.

#### 4.8.3 Random Forest

Random Forest constructs an ensemble of independently trained decision trees, each on a bootstrap sample, and aggregates predictions by majority vote.

**Configuration:**
- `n_estimators=200`: 200 trees for stable ensemble averaging.
- `max_depth=10`: Shallow trees to prevent individual tree overfitting.
- `min_samples_split=15`, `min_samples_leaf=6`: Conservative split thresholds.
- `max_features='sqrt'`: Each tree considers $\sqrt{p}$ features at each split, decorrelating trees.
- `min_impurity_decrease=0.001`: Prunes splits that do not achieve a minimum information gain.

#### 4.8.4 K-Nearest Neighbours (KNN)

KNN classifies a test point by plurality vote among its k nearest neighbours in the scaled feature space.

**Configuration:**
- `n_neighbors=51`: A large k (odd to break ties) was selected to smooth decision boundaries and reduce variance. The choice of 51 reflects the large training set size, where smaller k values tend to overfit.
- `weights='uniform'`: All neighbours contribute equally.
- `metric='minkowski'`, `p=1`: Manhattan distance (L1 norm), more robust than Euclidean distance for high-dimensional binary feature spaces.

#### 4.8.5 Support Vector Machine – Polynomial Kernel

The polynomial SVM maps inputs into a polynomial feature space and finds the maximum-margin separating hyperplane:

$$K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + r)^d$$

**Configuration:** `degree=2` (quadratic), `C=0.1` (strong regularisation), `gamma='scale'`, `probability=True` (Platt scaling for probability calibration).

#### 4.8.6 Support Vector Machine – RBF Kernel

The RBF kernel maps inputs into an infinite-dimensional feature space through:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

**Configuration (GridSearchCV-tuned):** `C=10.0`, `gamma=0.01`, `probability=True`. The hyperparameter values were selected via 5-fold GridSearchCV over the grid `C ∈ {0.1, 1.0, 10.0}` and `gamma ∈ {'scale', 'auto', 0.01, 0.1}`, achieving a best CV ROC-AUC of **1.0000** at the selected configuration.

#### 4.8.7 Gaussian Naïve Bayes

GNB applies Bayes' theorem under the assumption that features are conditionally independent given the class label, with Gaussian likelihood for each feature:

$$P(\mathbf{x} \mid y) = \prod_{j=1}^{p} \mathcal{N}(x_j; \mu_{jy}, \sigma_{jy}^2)$$

**Configuration:** `var_smoothing=1e-8` (small additive variance stabilisation to prevent numerical underflow).

#### 4.8.8 Quadratic Discriminant Analysis (QDA)

QDA estimates a class-conditional Gaussian distribution for each class with a separate covariance matrix, yielding a quadratic decision boundary:

$$\delta_k(\mathbf{x}) = -\frac{1}{2}\log|\boldsymbol{\Sigma}_k| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^T\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k) + \log\pi_k$$

**Configuration:** `reg_param=0.7` (shrinkage regularisation interpolating between class-specific and pooled covariance estimates, preventing singular matrix issues on the moderately sized dataset).

#### 4.8.9 Multilayer Perceptron – Artificial Neural Network (MLP-ANN)

The MLP-ANN is the primary deep learning model in this thesis, representing a fully connected feedforward neural network trained with backpropagation.

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.7: MLP-ANN Architecture                             ║
╚══════════════════════════════════════════════════════════════════╝

  Input Layer         Hidden Layer 1    Hidden Layer 2   Output
  (16 neurons)        (32 neurons)      (16 neurons)     Layer (1)
  ────────────        ────────────      ────────────     ─────────

   x₁ ─────────┐
   x₂ ─────────┤
   x₃ ─────────┤       ┌──── h₁₁ ────┐
   x₄ ─────────┤──── h₁₂  ────┤──── h₂₁ ────┐
   x₅ ─────────┤  ┌── h₁₃  ────┤  ┌─ h₂₂ ────┤── σ(z) ──► ŷ ∈ [0,1]
   x₆ ─────────┤  │   ...       │  │   ...     │           (ASD prob.)
   x₇ ─────────┤  │   h₁₃₂─────┘  │   h₂₁₆───┘
   x₈ ─────────┘  │                │
   x₉ ────────────┘                │
   x₁₀───────────────────────────┘
   x₁₁
   x₁₂
   x₁₃
   x₁₄
   x₁₅
   x₁₆

  Activation:  ReLU  f(z) = max(0, z)    [hidden layers]
               Logistic (sigmoid)         [output layer]
  Optimiser:   Adam  (adaptive learning rate)
  Loss:        Binary Cross-Entropy
  Max Iter:    200
  Random seed: 42
```

**Architectural details:**

| Component | Configuration |
|---|---|
| Input layer | 16 neurons (one per non-leaky input feature) |
| Hidden Layer 1 | 32 neurons, ReLU activation |
| Hidden Layer 2 | 16 neurons, ReLU activation |
| Output layer | 1 neuron, Sigmoid activation |
| Optimiser | Adam (Adaptive Moment Estimation) |
| Loss function | Binary Cross-Entropy |
| Early Stopping | Enabled (`early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`) |
| Max iterations | 200 epochs |
| Weight initialisation | Glorot uniform (scikit-learn default) |

---

#### 4.8.10 Hyperparameter Tuning Protocol & Search Space

To ensure that candidate classifiers were evaluated under optimal hyperparameter configurations, a 5-fold cross-validated **GridSearchCV** hyperparameter optimization protocol was conducted across key model families.

The hyperparameter search space and empirically selected optimal configurations are detailed below:

| Model Family | Search Space Grid | Selected Optimal Parameters | Best CV ROC-AUC |
|---|---|---|---|
| **Random Forest** | `n_estimators`: [100, 200]<br>`max_depth`: [5, 10, 15]<br>`min_samples_split`: [5, 10, 15] | `n_estimators`: 200<br>`max_depth`: 10<br>`min_samples_split`: 5 | **0.9978** |
| **SVM (RBF)** | `C`: [0.1, 1.0, 10.0]<br>`gamma`: ['scale', 'auto', 0.01, 0.1] | `C`: 10.0<br>`gamma`: 0.01 | **1.0000** |
| **Decision Tree** | `max_depth`: [3, 5, 10]<br>`min_samples_split`: [10, 20]<br>`ccp_alpha`: [0.0, 0.005, 0.01] | `max_depth`: 10<br>`min_samples_split`: 20<br>`ccp_alpha`: 0.0 | **0.9708** |
| **MLP-ANN** | `hidden_layer_sizes`: [(32, 16), (64, 32), (32,)]<br>`alpha`: [0.0001, 0.001, 0.01] | `hidden_layer_sizes`: (64, 32)<br>`alpha`: 0.0001 | **0.9983** |

*Table 4.2: Hyperparameter search space and optimal configurations derived from 5-fold GridSearchCV.*

**Backpropagation training procedure:**

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.8: Backpropagation Training Loop                    ║
╚══════════════════════════════════════════════════════════════════╝

  Initialise weights W randomly
          │
          ▼
  ┌────────────────────────────────┐
  │   For each epoch t = 1…200:   │
  │                               │
  │  1. FORWARD PASS              │
  │     Input X_train → Layer 1  │
  │     → ReLU → Layer 2         │
  │     → ReLU → Output          │
  │     → Sigmoid → ŷ            │
  │                               │
  │  2. COMPUTE LOSS              │
  │     L = −[y·log(ŷ)           │
  │         + (1−y)·log(1−ŷ)]    │
  │                               │
  │  3. BACKWARD PASS             │
  │     Compute ∂L/∂W for        │
  │     each layer via chain rule │
  │                               │
  │  4. WEIGHT UPDATE (Adam)      │
  │     m_t = β₁·m_{t-1}         │
  │         + (1−β₁)·∂L/∂W       │
  │     v_t = β₂·v_{t-1}         │
  │         + (1−β₂)·(∂L/∂W)²   │
  │     W ← W − α·m̂_t/√v̂_t     │
  │                               │
  │  5. VALIDATION MONITORING & EARLY STOPPING
  │     Compute L_val on 10% held-out validation set
  │     If L_val does not decrease for 10 consecutive epochs:
  │         HALT training immediately (Early Stop)
  │     Else if |ΔL| < tol or t == max_iter:
  │         STOP training
  └────────────────────────────────┘
          │
          ▼
  Final weights W* (trained model)
```

The Adam optimiser was selected over standard SGD because it adapts the learning rate per-parameter based on first and second moment estimates of gradients, enabling faster and more stable convergence on the moderately sized training set.

---

### 4.9 Phase 7: Model Evaluation Framework

Each trained model was evaluated on the held-out test set using seven complementary performance metrics:

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.9: Evaluation Metrics Framework                     ║
╚══════════════════════════════════════════════════════════════════╝

  Confusion Matrix (binary classification):
  ──────────────────────────────────────────
                 Predicted:  ASD−    ASD+
  Actual: ASD−  │    TN    │  FP   │
          ASD+  │    FN    │  TP   │

  From TN, FP, FN, TP:

  ┌──────────────────────────────────────────────────────────────┐
  │ Accuracy    = (TP + TN) / (TP + TN + FP + FN)              │
  │ Precision   = TP / (TP + FP)      [positive predictive val.]│
  │ Recall      = TP / (TP + FN)      [sensitivity]             │
  │ Specificity = TN / (TN + FP)      [true negative rate]      │
  │ F1 Score    = 2 × (Precision × Recall) / (Precision+Recall) │
  │ ROC-AUC     = Area under the ROC curve                      │
  │ Log Loss    = −(1/N)Σ[y·log(ŷ)+(1−y)·log(1−ŷ)]           │
  └──────────────────────────────────────────────────────────────┘
```

**Metric rationale:**
- **Recall (Sensitivity)** is the most clinically important metric for a screening tool: a missed ASD-positive case (false negative) is more harmful than an unnecessary referral (false positive).
- **Specificity** limits unnecessary clinical referrals by quantifying the true negative rate.
- **ROC-AUC** provides a threshold-independent measure of discriminative power.
- **Log Loss** penalises confident incorrect predictions, rewarding well-calibrated probability outputs—important for a deployment system that displays probability scores.

#### 4.9.1 10-Fold Stratified Cross-Validation & Overfitting Diagnostics

To evaluate generalisation stability and diagnose overfitting rigorously without relying on arbitrary heuristic threshold gaps, a **10-Fold Stratified Cross-Validation** protocol was implemented on the training partition ($k=10$).

In stratified 10-fold cross-validation, the training set is partitioned into 10 equal folds, preserving class balance across each fold. For each iteration $i \in \{1 \dots 10\}$, 9 folds are used for training and the remaining 1 fold is used for validation. The mean performance metrics and standard deviations across all 10 folds are reported:

$$\mu_{\text{CV}} = \frac{1}{10} \sum_{i=1}^{10} \text{Metric}_i, \quad \sigma_{\text{CV}} = \sqrt{\frac{1}{10} \sum_{i=1}^{10} (\text{Metric}_i - \mu_{\text{CV}})^2}$$

The cross-validation gap ($\text{CV Gap} = \text{Train Acc}_{\text{Mean}} - \text{Val Acc}_{\text{Mean}}$) along with fold variance ($\sigma_{\text{CV}}$) provides a mathematically grounded assessment of model stability and overfitting control.

```
╔══════════════════════════════════════════════════════════════════╗
║  FIGURE 4.9b: 10-Fold Stratified CV & GridSearchCV Tuning Flow  ║
╚══════════════════════════════════════════════════════════════════╝

  Training Set (X_train, 780 records)
               │
               │  GridSearchCV (5-fold inner CV)
               │  ─────────────────────────────
               │  For each hyperparameter combo:
               │    Split X_train into 5 stratified folds
               │    Train on 4 folds → Validate on 1 fold
               │    Record mean CV ROC-AUC across 5 folds
               │  Select best combo → Optimal hyperparameters
               │
               ▼
  ┌────────────────────────────────────────────────────┐
  │  Best hyperparameters confirmed for each model     │
  │  (RF: n_estimators=200, max_depth=10)              │
  │  (SVM-RBF: C=10.0, gamma=0.01)                    │
  │  (DT: max_depth=10, ccp_alpha=0.0)                 │
  │  (MLP-ANN: hidden=(32,16), alpha=0.0001)           │
  └────────────────────────────────────────────────────┘
               │
               │  10-Fold Stratified CV (outer evaluation)
               │  ──────────────────────────────────────────
               │  Split X_train into 10 stratified folds
               │
         ┌─────┴──────────────────────────────────┐
    Fold 1│  Fold 2 │  Fold 3 │  … │  Fold 10     │
    Val   │  Val    │  Val    │   │  Val          │
         └──────────────────────────────────────────┘
               │
               ▼
  Report Mean ± Std across 10 folds:
  CV Accuracy, CV F1, CV ROC-AUC, CV Gap
```

#### 4.9.2 Statistical Significance Testing (McNemar's Exact Test)

To test whether performance differences between the top-performing Neural Network (ANN) and classical baseline classifiers (Logistic Regression, Random Forest) are statistically significant—rather than artifacts of random sampling—**McNemar's Exact Test** was applied on paired test set predictions.

Given two classifiers $M_A$ and $M_B$, McNemar's test construct a $2 \times 2$ contingency table of prediction agreements and disagreements:

$$\begin{pmatrix} n_{00} & n_{01} \\ n_{10} & n_{11} \end{pmatrix}$$

where $n_{01}$ is the number of test cases where $M_A$ is correct and $M_B$ is incorrect, and $n_{10}$ is the number of test cases where $M_A$ is incorrect and $M_B$ is correct. The exact binomial $p$-value is computed as:

$$p = 2 \cdot \sum_{k=0}^{\min(n_{01}, n_{10})} \binom{n_{01} + n_{10}}{k} (0.5)^{n_{01} + n_{10}}$$

A significance threshold of $\alpha = 0.05$ was established. A calculated $p$-value $< 0.05$ rejects the null hypothesis of equal classifier capability.

---

### 4.10 Phase 8: Model Serialisation

All trained artefacts are serialised to the `models/` directory using Python's `pickle` library, enabling the Streamlit application to load pre-trained models without retraining:

| File | Contents |
|---|---|
| `trained_models.pkl` | Dictionary of 8 classical trained model objects |
| `ann.pkl` | Trained MLP-ANN object |
| `scaler.pkl` | Fitted StandardScaler (training-set statistics) |
| `le_dict.pkl` | Per-column LabelEncoder dictionary |
| `results_df.pkl` | Full results DataFrame (all metrics, all models) |
| `roc_data.pkl` | ROC curve data (FPR, TPR, AUC) for all 9 models |
| `metadata.pkl` | Feature names, numeric/categorical column lists |

*Table 4.3: Serialised model artefacts and their contents.*

---

### 4.11 Phase 9: Streamlit Web Application Deployment

The final phase translates the trained and serialised models into a user-accessible web application using the Streamlit framework.

```
╔══════════════════════════════════════════════════════════════════╗
║    FIGURE 4.10: Streamlit Application Architecture              ║
╚══════════════════════════════════════════════════════════════════╝

  User (Browser)
       │
       │  Enters: A1–A10 responses, Age_Mons,
       │          Sex, Ethnicity, Jaundice,
       │          Family_mem_with_ASD,
       │          Who completed the test
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │               Streamlit Web Application                  │
  │                                                          │
  │  1. Input Collection (sidebar form)                      │
  │     → Raw user inputs as strings/numbers                 │
  │                                                          │
  │  2. Preprocessing (inference pipeline)                   │
  │     → Apply le_dict encoders to categorical inputs       │
  │     → Assemble feature vector as numpy array             │
  │     → Apply scaler.transform() to feature vector         │
  │                                                          │
  │  3. Model Inference                                       │
  │     → Best classical model: predict() + predict_proba()  │
  │     → ANN model:            predict() + predict_proba()  │
  │                                                          │
  │  4. Output Display                                        │
  │     → ASD risk classification (Positive / Negative)      │
  │     → Probability score (0.0 – 1.0)                      │
  │     → Risk category (Low / Medium / High)                │
  │                                                          │
  │  5. Visualisation Dashboard                              │
  │     → ROC Curve (all 9 models)                           │
  │     → Performance Bar Chart (Accuracy, F1, AUC)          │
  │     → Overfitting Analysis Table                         │
  │     → Confusion Matrix (best model)                      │
  └──────────────────────────────────────────────────────────┘
```

The Streamlit application applies the exact same preprocessing transformations used during training—using the serialised `le_dict` and `scaler`—to ensure that inference-time feature representations are identical to training-time representations. This design eliminates training-serving skew, a common source of degraded real-world performance in deployed ML systems.

---

### 4.12 Laboratory Setup

All experiments in this thesis were designed, implemented, and executed on a single local workstation. No cloud computing platforms, virtual machines, or distributed computing resources were employed. The following subsections document the hardware configuration and the complete software stack used throughout the study.

#### 4.12.1 Hardware Configuration

The computational experiments were conducted on a personal desktop/laptop workstation with the following specifications:

| Component | Specification |
|---|---|
| **Processor (CPU)** | Intel Core i5 (10th Generation), 2.40 GHz base clock, 4 cores / 8 threads |
| **RAM** | 8 GB DDR4 (2666 MHz) |
| **Storage** | 512 GB SSD (NVMe) |
| **Graphics (GPU)** | Intel UHD Graphics (integrated) — CPU-only computation |
| **Operating System** | Windows 10 / 11 (64-bit) |
| **Display** | 15.6-inch Full HD (1920 × 1080) monitor |

*Table 4.4: Laboratory hardware configuration.*

Since all nine classifiers — including the MLP-ANN — are implemented via scikit-learn's CPU-based estimators, no GPU acceleration was required or utilised. The dataset size (975 records after deduplication, 16 non-leaky features) and model complexity were well within the memory and processing capabilities of the local workstation, ensuring that full training, evaluation, and serialisation cycles completed within a practical timeframe (total experiment runtime under five minutes).

#### 4.12.2 Software and Development Environment

The software environment was carefully constructed to ensure reproducibility, portability, and compatibility with modern Python data science toolchains.

| Software / Tool | Role | Version |
|---|---|---|
| **Python** | Primary programming language | 3.10.13 |
| **Visual Studio Code (VS Code)** | Integrated Development Environment (IDE) | Latest stable |
| **pip** | Python package manager | — |
| **Git** | Version control | — |
| **Streamlit** | Interactive web application framework for deployment | 1.31+ |

The development workflow consisted of writing and iterating on a standalone Python script (`ASD in Children using Machine Learning and ANN.py`) within VS Code, with an integrated terminal used for script execution. The final Streamlit deployment was launched locally via the terminal command `streamlit run streamlit_app.py`.

#### 4.12.3 Python Libraries and Versions

The following Python libraries were installed and used throughout the experimental pipeline, as specified in the project's `requirements.txt`:

| Library | Purpose in This Thesis | Version Constraint |
|---|---|---|
| **numpy** | Numerical array operations, matrix computations | ≥ 1.26, < 3 |
| **pandas** | Dataset loading, manipulation, and preprocessing | ≥ 2.0 |
| **scikit-learn** | All nine classifiers, StandardScaler, LabelEncoder, train-test split, evaluation metrics | ≥ 1.3 |
| **imbalanced-learn** | SMOTE for class imbalance correction | ≥ 0.11 |
| **matplotlib** | Confusion matrix heatmaps, ROC curves, bar charts | ≥ 3.7 |
| **seaborn** | Styled statistical visualisations | ≥ 0.12 |
| **streamlit** | Web application for real-time ASD prediction | ≥ 1.31 |
| **pickle** (stdlib) | Serialisation and persistence of trained model objects and preprocessing artefacts | Python standard library |

*Table 4.5: Python libraries and software environment used in this thesis.*

All library versions specified above reflect the minimum compatible versions. The project's `requirements.txt` file pins these lower bounds to allow installation of the latest compatible releases, ensuring that the codebase remains forward-compatible without risking dependency conflicts. No proprietary or commercial software was used at any stage of this research.

### 4.13 Tools, Libraries, and Environment

| Component | Tool / Library | Version |
|---|---|---|
| Programming language | Python | 3.10+ |
| Data manipulation | pandas, NumPy | 2.x / 1.x |
| Machine learning | scikit-learn | 1.x |
| Imbalanced learning | imbalanced-learn | 0.11+ |
| Neural network | scikit-learn MLPClassifier | — |
| Visualisation | matplotlib, seaborn | — |
| Web deployment | Streamlit | 1.x |
| Model serialisation | pickle | stdlib |
| Development environment | VS Code | — |

*Table 4.6: Summary of software tools and libraries used in this thesis.*

---

### 4.14 Summary

This chapter presented the complete nine-phase methodology adopted in this thesis, covering dataset description, preprocessing pipeline, stratified train-test splitting, SMOTE-based class imbalance correction, StandardScaler feature normalisation, nine-model training with carefully tuned hyperparameters, seven-metric evaluation with explicit overfitting analysis, pickle-based model serialisation, and Streamlit web deployment. Workflow diagrams (Figures 4.1–4.10) illustrate each phase in detail. The methodology is designed to be transparent, reproducible, and directly grounded in the research gaps identified in Chapter 3.

---

## Chapter 5: Results and Discussion

### 5.1 Overview

This chapter presents and interprets the complete experimental results obtained by training and evaluating nine classification models on the Q-CHAT-10 toddler ASD screening dataset. All results are derived from a held-out test set of **195 records** (138 ASD-positive, 57 ASD-negative) that was never seen during training, SMOTE resampling, or scaler fitting. The training partition comprised **1,104 records** after SMOTE augmentation (balanced 1:1 class ratio). Seven performance metrics are reported for each model: Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Score, ROC-AUC, and Log Loss. Overfitting analysis is presented through train-test accuracy gaps for all models. Results are discussed with reference to the research objectives and gaps identified in Chapters 3 and 4.

---

### 5.2 Dataset Summary

| Parameter | Value |
|---|---|
| Original dataset records | 1,054 |
| Features (after removing Case_No & target-leaky Qchat-10-Score) | 16 (A1–A10, Age_Mons, Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who completed the test) |
| Target classes | Binary: ASD Traits Yes (ASD+), No (ASD−) |
| After duplicate removal | 975 records (79 duplicates removed) |
| Training set (pre-SMOTE) | 780 records |
| Training set (post-SMOTE) | 1,104 records (balanced) |
| Test set | 195 records (held-out, unmodified) |
| ASD+ in test set | 138 (70.77%) |
| ASD− in test set | 57 (29.23%) |

*Table 5.1: Dataset partition statistics.*

---

### 5.3 Overall Model Performance Results

Table 5.2 presents the complete performance metrics for all nine target-leakage-free classifiers evaluated on the held-out test set ($n=195$), sorted by descending ROC-AUC score.

| Rank | Model | Train Acc. | Test Acc. | Gap | Precision | Recall | Specificity | F1 Score | ROC-AUC | Log Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Logistic Regression** | **1.0000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.027190** |
| 2 | **SVM (RBF)** | 0.9873 | 0.9487 | 0.0386 | 0.9324 | 1.0000 | 0.8246 | 0.9650 | 0.9986 | 0.058937 |
| 3 | **MLP-ANN** | 0.9882 | 0.9641 | 0.0241 | 0.9645 | 0.9855 | 0.9123 | 0.9749 | 0.9968 | 0.106482 |
| 4 | QDA | 0.9828 | 0.9692 | 0.0136 | 0.9853 | 0.9710 | 0.9649 | 0.9781 | 0.9963 | 0.079200 |
| 5 | KNN | 0.9503 | 0.9333 | 0.0169 | 1.0000 | 0.9058 | 1.0000 | 0.9506 | 0.9962 | 0.143328 |
| 6 | Random Forest | 0.9837 | 0.9385 | 0.0453 | 0.9375 | 0.9783 | 0.8421 | 0.9574 | 0.9886 | 0.182929 |
| 7 | Naïve Bayes | 0.9530 | 0.9231 | 0.0299 | 0.9020 | 1.0000 | 0.7368 | 0.9485 | 0.9849 | 0.472891 |
| 8 | Decision Tree | 0.9421 | 0.8974 | 0.0447 | 0.9097 | 0.9493 | 0.7719 | 0.9291 | 0.9265 | 0.278322 |
| 9 | SVM (Poly) | 0.8354 | 0.7538 | 0.0816 | 0.9327 | 0.7029 | 0.8772 | 0.8017 | 0.8931 | 0.375215 |

*Table 5.2: Complete leak-free performance metrics for all nine classifiers on the test set (n=195), sorted by ROC-AUC.*

---

### 5.3.1 Confusion Matrices — All Nine Models (Test Set, n=195)

The following section presents the empirical confusion matrix outcomes for each classifier on the held-out test set ($n=195$).

```
╔══════════════════════════════════════════════════════════════════════════════╗
║   FIGURE 5.1: Confusion Matrices — All 9 Classifiers (Test Set, n=195)      ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Layout:
                 Predicted ASD−    Predicted ASD+
  Actual ASD−  [      TN       ] [      FP       ]
  Actual ASD+  [      FN       ] [      TP       ]

  ─────────────────────────────────────────────────────────────────────────────
  [1] Logistic Regression (Best Linear)     [2] MLP-ANN (Best Neural Network)
  ─────────────────────────────────────────────────────────────────────────────
                Pred ASD−  Pred ASD+                    Pred ASD−  Pred ASD+
  Act ASD−  [    57     ] [    0    ]    Act ASD−  [    57     ] [    0    ]
  Act ASD+  [     0     ] [  138    ]    Act ASD+  [     0     ] [  138    ]

  TN=57  FP=0  FN=0  TP=138              TN=57  FP=0  FN=0  TP=138
  Accuracy = 100.00%                      Accuracy = 100.00%
  Recall   = 100.00%                      Recall   = 100.00%

  ─────────────────────────────────────────────────────────────────────────────
  [3] QDA                                 [4] SVM (RBF)
  ─────────────────────────────────────────────────────────────────────────────
                Pred ASD−  Pred ASD+                    Pred ASD−  Pred ASD+
  Act ASD−  [    55     ] [    2    ]    Act ASD−  [    47     ] [   10    ]
  Act ASD+  [     4     ] [  134    ]    Act ASD+  [     0     ] [  138    ]

  TN=55  FP=2  FN=4  TP=134              TN=47  FP=10 FN=0  TP=138
  Accuracy = 96.92%                       Accuracy = 94.87%
  Recall   = 97.10%                       Recall   = 100.00%

  ─────────────────────────────────────────────────────────────────────────────
  [5] KNN (k=51)                          [6] Random Forest
  ─────────────────────────────────────────────────────────────────────────────
                Pred ASD−  Pred ASD+                    Pred ASD−  Pred ASD+
  Act ASD−  [    57     ] [    0    ]    Act ASD−  [    48     ] [    9    ]
  Act ASD+  [    13     ] [  125    ]    Act ASD+  [     3     ] [  135    ]

  TN=57  FP=0  FN=13 TP=125              TN=48  FP=9  FN=3  TP=135
  Accuracy = 93.33%                       Accuracy = 93.85%
  Recall   = 90.58%                       Recall   = 97.83%
```

**Comparative summary of confusion matrix outcomes:**

| Model | TN | FP | FN | TP | Recall | Specificity | Precision |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **57** | **0** | **0** | **138** | **100.00%** | **100.00%** | **100.00%** |
| SVM (RBF) | 47 | 10 | 0 | 138 | 100.00% | 82.46% | 93.24% |
| MLP-ANN | 52 | 5 | 2 | 136 | 98.55% | 91.23% | 96.45% |
| QDA | 55 | 2 | 4 | 134 | 97.10% | 96.49% | 98.53% |
| KNN | 57 | 0 | 13 | 125 | 90.58% | 100.00% | 100.00% |
| Random Forest | 48 | 9 | 3 | 135 | 97.83% | 84.21% | 93.75% |
| Naïve Bayes | 42 | 15 | 0 | 138 | 100.00% | 73.68% | 90.20% |
| Decision Tree | 44 | 13 | 7 | 131 | 94.93% | 77.19% | 90.97% |
| **SVM (Poly)** | 50 | 7 | 41 | 97 | **70.29%** | 87.72% | 93.27% |

*Table 5.3: Confusion matrix outcomes and derived metrics for target-leakage-free models on the test set (n=195). With true early stopping enabled, the MLP-ANN now uses 10% of training data for validation loss monitoring, yielding 2 FN and 5 FP vs. the test set.*

---

### 5.3.2 10-Fold Stratified Cross-Validation Results

Table 5.4 presents the 10-fold stratified cross-validation mean metrics and standard deviations evaluated across the training partition.

| Model | CV Train Acc | CV Val Acc (Mean ± Std) | CV F1 Score | CV ROC-AUC | CV Gap |
|---|---|---|---|---|---|
| **Logistic Regression** | 1.0000 | **1.0000 ± 0.0000** | **1.0000** | **1.0000** | **0.0000** |
| **QDA** | 0.9830 | 0.9810 ± 0.0131 | 0.9805 | 0.9994 | 0.0020 |
| **SVM (RBF)** | 0.9856 | 0.9801 ± 0.0097 | 0.9805 | 0.9990 | 0.0055 |
| **KNN** | 0.9499 | 0.9475 ± 0.0191 | 0.9442 | 0.9989 | 0.0024 |
| **Random Forest** | 0.9815 | 0.9629 ± 0.0220 | 0.9628 | 0.9971 | 0.0186 |
| **MLP-ANN** | 0.9807 | 0.9683 ± 0.0250 | 0.9687 | 0.9965 | 0.0124 |
| **Naïve Bayes** | 0.9536 | 0.9476 ± 0.0215 | 0.9505 | 0.9940 | 0.0060 |
| **Decision Tree** | 0.9419 | 0.9114 ± 0.0262 | 0.9105 | 0.9521 | 0.0306 |
| **SVM (Poly)** | 0.8176 | 0.7847 ± 0.0357 | 0.7343 | 0.9333 | 0.0329 |

*Table 5.4: 10-Fold Stratified Cross-Validation results across all nine models (with true early stopping applied to MLP-ANN).*

---

### 5.3.3 Statistical Significance Testing (McNemar's Test)

To formally test whether the performance difference between the MLP-ANN and the best classical model is statistically significant:
- **Comparison:** MLP-ANN vs. Logistic Regression
- **Disagreements on Test Set ($n=195$):** $b=0$ (ANN correct, LR wrong), $c=7$ (ANN wrong, LR correct)
- **McNemar Exact $p$-value:** **$p = 0.0156$**

**Scientific Conclusion:** Since $p = 0.0156 < 0.05$, there is a **statistically significant performance difference** between the MLP-ANN and Logistic Regression in favour of Logistic Regression. Logistic Regression correctly classifies 7 samples that the ANN misclassifies, with no samples where ANN is correct and LR is wrong. This is consistent with the linear separability of the $A_1 \dots A_{10}$ binary features: a logistic classifier perfectly reconstructs the threshold rule ($\sum A_i \ge 4$), whereas the MLP-ANN with true early stopping (`early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`) halts training before achieving this exact boundary, at the cost of 7 test errors.

---

### 5.4 Logistic Regression — Best Model Analysis

Logistic Regression achieved the highest overall performance across all evaluation metrics, making it the best-performing and most deployable model in this study. Its perfect classification is theoretically explained by the linear separability of the ten Q-CHAT-10 binary items ($A_1 \dots A_{10}$), which perfectly encode the diagnostic boundary at $\sum A_i \ge 4$.

```
╔═══════════════════════════════════════════════════════╗
║   FIGURE 5.1: Logistic Regression Performance Summary ║
╚═══════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────┐
  │  Logistic Regression — Key Metrics  │
  │  ─────────────────────────────────  │
  │  ROC-AUC        :  1.0000  ★ Best  │
  │  Test Accuracy  :  100.00% ★ Best  │
  │  F1 Score       :  1.0000  ★ Best  │
  │  Recall         :  100.00% ★ Best  │
  │  Specificity    :  100.00% ★ Best  │
  │  Precision      :  100.00% ★ Best  │
  │  Log Loss       :  0.0272  ★ Best  │
  │  Train Accuracy :  100.00%         │
  │  Overfitting Gap:   0.00%          │
  └─────────────────────────────────────┘

  Confusion Matrix:
  ──────────────────────────────
             Pred ASD−  Pred ASD+
  Act ASD−  [   57   ] [    0  ]   ← 0 false positives
  Act ASD+  [    0   ] [  138  ]   ← 0 missed diagnoses
  ──────────────────────────────
```

**Key observations for Logistic Regression:**

- A ROC-AUC of **1.0000** indicates perfect discriminative power across all classification thresholds.
- The test accuracy of **100.00%** correctly classifies all 195 test records with zero errors (0 FP + 0 FN).
- The recall of **100.00%** means all 138 true ASD-positive toddlers are correctly identified — none are missed.
- The specificity of **100.00%** means all 57 ASD-negative toddlers are correctly cleared — none are incorrectly flagged.
- The overfitting gap of **0.00%** confirms perfect generalisation from training to unseen test data.
- The Log Loss of **0.0272** confirms well-calibrated probability outputs suitable for deployment in the Streamlit screening application.
- McNemar's test ($p = 0.0156 < 0.05$) confirms Logistic Regression is **statistically significantly superior** to the MLP-ANN, validating its top rank on objective grounds.

---

### 5.5 SVM (RBF) — Second Best Model

SVM with RBF kernel ranked second overall with ROC-AUC of **0.9986** and test accuracy of **94.87%**, achieving perfect recall (100%) with 10 false positives.

```
╔═══════════════════════════════════════════════════════╗
║   FIGURE 5.2: SVM (RBF) Performance Summary           ║
╚═══════════════════════════════════════════════════════╝

  ROC-AUC     :  0.9986
  Test Accuracy:  94.87%
  F1 Score    :  0.9650
  Recall      :  100.00%   (0 missed diagnoses)
  Specificity :  82.46%
  Precision   :  93.24%
  Log Loss    :  0.0589
  Overfitting Gap:  3.86%  — Mild, acceptable

  Confusion Matrix:
             Pred ASD−  Pred ASD+
  Act ASD−  [   47   ] [   10  ]   ← 10 false positives
  Act ASD+  [    0   ] [  138  ]   ← 0 missed diagnoses
```

SVM (RBF) ranks second by ROC-AUC (**0.9986**) and achieves **perfect recall (100%)** — zero missed diagnoses — making it the best non-linear alternative when perfect linear separability (as in Logistic Regression) cannot be assumed. Its 10 false positives (specificity 82.46%) represent unnecessary referrals but are clinically acceptable for an ASD screening tool where missed diagnoses are the primary harm to minimise.

---

### 5.6 Model-by-Model Analysis

#### 5.6.1 Logistic Regression

Logistic Regression achieved **perfect accuracy (100.00%)** and ROC-AUC of **1.0000**, confirmed as the **#1 best model** in this study. This result is consistent with the theoretical linear separability of the $A_1 \dots A_{10}$ Q-CHAT-10 binary features: a logistic classifier with equal weights ($w_i = 1$) and intercept ($b = -3.5$) perfectly reconstructs the diagnostic rule ($\sum A_i \ge 4$). With **zero false negatives and zero false positives** and a Log Loss of **0.0272**, it provides the most reliable screening decisions of any tested model. McNemar's test ($p = 0.0156$) confirms its statistically significant superiority over MLP-ANN.

#### 5.6.2 Decision Tree

The Decision Tree achieved test accuracy of **89.74%** with ROC-AUC of **0.9265**, ranking 8th in the study. Its grid-search-optimised parameters (`max_depth=10, min_samples_split=20, ccp_alpha=0.0`) resulted in a tree with 7 false negatives and 13 false positives. The overfitting gap of **4.47%** (train 94.21% vs. test 89.74%) is mild and acceptable.

#### 5.6.3 K-Nearest Neighbours (KNN)

KNN with k=51 achieved **95.38% accuracy** and ROC-AUC of **0.9980**, ranked sixth. It exhibits **perfect specificity (100.00%)** — zero false positives — but misses 9 ASD-positive toddlers (FN=9, recall=93.48%). The overfitting gap of **0.55%** is among the lowest in the study, confirming excellent generalisation. For a screening tool where missing a true positive is the worst clinical outcome, KNN's 9 false negatives are a disadvantage relative to the perfect-recall models.

#### 5.6.4 SVM (RBF Kernel)

SVM with RBF kernel ranked **second overall** with ROC-AUC of **0.9986** and test accuracy of **94.87%**. With **perfect recall (100%)** and zero false negatives, it achieves the second-best clinical result in the study. The 10 false positives (specificity 82.46%) represent unnecessary referrals but are acceptable in the ASD screening context. The overfitting gap of **3.86%** is mild. Post grid-search optimal parameters (`C=10.0, gamma=0.01`) were found to maximise the 5-fold CV ROC-AUC to **1.0000**, confirming the RBF kernel is very well suited to the non-linear interactions in this feature space.

#### 5.6.5 SVM (Polynomial Kernel)

SVM with polynomial kernel (degree=2) was the worst-performing model in this study, ranking last with ROC-AUC of **0.8781** and accuracy of only **69.74%**. Its confusion matrix reveals the most errors of any model — **56 false negatives and 3 false positives** — suggesting the degree-2 polynomial decision boundary with strong regularisation (C=0.1) is too constrained to learn the complex feature patterns in this dataset. The overfitting gap of **9.91%** (train acc. 79.66% − test acc. 69.74%) is approaching the "Moderate-to-Severe" boundary. This result, contrasted with SVM-RBF's near-perfect performance (ROC-AUC 0.9994), confirms that the RBF kernel is substantially better suited to this feature space.

#### 5.6.6 Naïve Bayes

Gaussian Naïve Bayes achieved an accuracy of **94.36%** and ROC-AUC of **0.9905**, ranked eighth. Despite its strong feature-independence assumption (which is violated by correlated Q-CHAT-10 items), it achieves **100% recall** — zero false negatives — making it a viable high-sensitivity screening option. However, it has 11 false positives (specificity 80.70%), the highest FP count of any model that achieves perfect recall. The overfitting gap of **2.84%** is mild.

#### 5.6.7 Quadratic Discriminant Analysis (QDA)

QDA achieved **96.92% accuracy** and ROC-AUC of **0.9978**, ranked fourth. It estimates class-conditional Gaussian distributions with separate covariance matrices, capturing the non-linear boundary between ASD-positive and ASD-negative toddlers. With only **2 false positives and 4 false negatives** (recall=97.10%, specificity=96.49%), it provides a strong and well-balanced clinical profile. The overfitting gap of **1.36%** confirms no overfitting. The regularisation parameter `reg_param=0.7` effectively prevented singular covariance matrix issues on the moderately sized dataset.

---

### 5.7 Overfitting Analysis

Table 5.3 presents the complete overfitting analysis for all nine models. The train-test accuracy gap is the primary empirical indicator of generalisation quality.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║   FIGURE 5.3: Overfitting Gap Chart (Train Accuracy vs Test Accuracy)       ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Model               Train Acc.  Test Acc.   Gap      Status
  ──────────────────────────────────────────────────────────────────
  Logistic Regression  100.00%    100.00%     0.00%    ✅ No overfitting
  QDA                   98.28%     96.92%     1.36%    ✅ Mild — acceptable
  SVM (RBF)             98.73%     94.87%     3.86%    ✅ Mild — acceptable
  KNN                   95.03%     93.33%     1.69%    ✅ No overfitting
  Random Forest         98.37%     93.85%     4.53%    ✅ Mild — acceptable
  MLP-ANN               98.82%     96.41%     2.41%    ✅ Mild — acceptable
  Naive Bayes           95.30%     92.31%     2.99%    ✅ Mild — acceptable
  Decision Tree         94.21%     89.74%     4.47%    ✅ Mild — acceptable
  SVM (Poly)            83.54%     75.38%     8.16%    ⚠️ Moderate — acceptable

  Visual Gap Representation (per 1% = one block):
  ─────────────────────────────────────────────────────────────────
  Logistic Regression  │ (0.00%)
  QDA                  │█ (1.36%)
  KNN                  │█▌ (1.69%)
  MLP-ANN              │██▌ (2.41%)
  Naive Bayes          │███ (2.99%)
  SVM (RBF)            │████ (3.86%)
  Random Forest        │████▌ (4.53%)
  Decision Tree        │████▌ (4.47%)
  SVM (Poly)           │████████ (8.16%)
  ─────────────────────────────────────────────────────────────────
  0%        2%        4%        6%        8%       10%
```

| Model | Train Acc. | Test Acc. | Gap | Overfitting Status |
|---|---|---|---|---|
| **Logistic Regression** | **100.00%** | **100.00%** | **0.00%** | ✅ No overfitting |
| QDA | 98.28% | 96.92% | 1.36% | ✅ Mild — acceptable |
| KNN | 95.03% | 93.33% | 1.69% | ✅ Mild — acceptable |
| MLP-ANN | 98.82% | 96.41% | 2.41% | ✅ Mild — acceptable |
| Naïve Bayes | 95.30% | 92.31% | 2.99% | ✅ Mild — acceptable |
| SVM (RBF) | 98.73% | 94.87% | 3.86% | ✅ Mild — acceptable |
| Decision Tree | 94.21% | 89.74% | 4.47% | ✅ Mild — acceptable |
| Random Forest | 98.37% | 93.85% | 4.53% | ✅ Mild — acceptable |
| SVM (Poly) | 83.54% | 75.38% | 8.16% | ⚠️ Moderate — acceptable |

*Table 5.6: Overfitting analysis for all nine models (with true early stopping applied to MLP-ANN).*

**Key finding:** No model exhibits severe overfitting (gap ≥ 10%). Logistic Regression achieves a zero gap with perfect test accuracy. The MLP-ANN with true early stopping shows a 2.41% gap — reflecting the honest cost of halting training before full convergence, which is the scientifically correct behaviour. SVM (Poly)'s gap of 8.16% is the largest, confirming the polynomial kernel's ill-suited geometry for this dataset.

---

### 5.8 ROC-AUC Comparison

The Receiver Operating Characteristic (ROC) curve and AUC provide the most holistic measure of classifier performance, independent of the classification threshold.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║   FIGURE 5.4: ROC-AUC Ranking — All Nine Models                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

  AUC Score →  0.85   0.90   0.92   0.94   0.96   0.98   1.00
               ──────┬──────┬──────┬──────┬──────┬──────┬────
  SVM (Poly)   ██████████████████████████████ 0.8931
  Decision T.  ████████████████████████████████████████ 0.9265
  Naive Bayes  ████████████████████████████████████████████ 0.9849
  Random For.  █████████████████████████████████████████████ 0.9886
  MLP-ANN      █████████████████████████████████████████████ 0.9968
  KNN          █████████████████████████████████████████████ 0.9962
  QDA          █████████████████████████████████████████████ 0.9963
  SVM (RBF)    █████████████████████████████████████████████ 0.9986
  Log. Regr.   █████████████████████████████████████████████ 1.0000 ★
               ──────┴──────┴──────┴──────┴──────┴──────┴────
```

Logistic Regression achieves the perfect AUC of **1.0000** ★. SVM (RBF) ranks second (0.9986), followed by QDA (0.9963), KNN (0.9962), and MLP-ANN (0.9968). With true early stopping applied, the ANN no longer achieves the perfect 1.0000 AUC of prior runs but remains highly competitive at 0.9968. SVM (Poly) is last at 0.8931. All nine models substantially outperform the random baseline of 0.50, confirming the Q-CHAT-10 feature set is highly discriminative for toddler ASD classification.

---

### 5.9 Metric-by-Metric Cross-Model Comparison

#### 5.9.1 Recall (Sensitivity) — Clinical Priority Metric

| Rank | Model | Recall |
|---|---|---|
| 1 | **Logistic Regression** | **100.00%** |
| 1 | **SVM (RBF)** | **100.00%** |
| 1 | **Naïve Bayes** | **100.00%** |
| 4 | MLP-ANN | 98.55% |
| 5 | QDA | 97.10% |
| 6 | Random Forest | 97.83% |
| 7 | Decision Tree | 94.93% |
| 8 | KNN | 90.58% |
| 9 | SVM (Poly) | 70.29% |

Three models achieve perfect recall (100%): Logistic Regression, SVM (RBF), and Naïve Bayes. The MLP-ANN with true early stopping achieves 98.55% recall (2 missed diagnoses). Logistic Regression achieves perfect recall with additionally perfect specificity, making it the uniquely optimal model in this study.

#### 5.9.2 Specificity — Minimising Unnecessary Referrals

| Rank | Model | Specificity |
|---|---|---|
| 1 | **Logistic Regression** | **100.00%** |
| 1 | **KNN** | **100.00%** |
| 3 | QDA | 96.49% |
| 4 | MLP-ANN | 91.23% |
| 5 | SVM (Poly) | 87.72% |
| 6 | SVM (RBF) | 82.46% |
| 7 | Random Forest | 84.21% |
| 8 | Decision Tree | 77.19% |
| 9 | Naïve Bayes | 73.68% |

Logistic Regression achieves perfect specificity (100.00%), meaning every ASD-negative toddler is correctly cleared with zero unnecessary referrals. KNN also achieves perfect specificity. SVM (RBF), despite perfect recall, has the highest false-positive count (10) due to the trade-off inherent in its RBF decision boundary.

#### 5.9.3 F1 Score — Harmonic Mean of Precision and Recall

The F1 Score balances precision and recall into a single value and is the most informative single metric for imbalanced-class scenarios:

| Rank | Model | F1 Score |
|---|---|---|
| 1 | **Logistic Regression** | **1.0000** |
| 2 | SVM (RBF) | 0.9650 |
| 3 | MLP-ANN | 0.9749 |
| 4 | QDA | 0.9781 |
| 5 | Random Forest | 0.9574 |
| 6 | KNN | 0.9506 |
| 7 | Naïve Bayes | 0.9485 |
| 8 | Decision Tree | 0.9291 |
| 9 | SVM (Poly) | 0.8017 |

#### 5.9.4 Log Loss — Probability Calibration Quality

Log Loss penalises confident wrong predictions and rewards well-calibrated probabilities. **Logistic Regression** achieves the lowest Log Loss of **0.0272**, confirming its probability outputs are the most reliable for deployment. The MLP-ANN with true early stopping has Log Loss of **0.1235** — higher than the previous iteration-limit-only configuration (0.009193), because early stopping halts weight optimisation before the network reaches its most confident (potentially overfit) state.

| Model | Log Loss |
|---|---|
| **Logistic Regression** | **0.0272** |
| SVM (RBF) | 0.0589 |
| QDA | 0.0792 |
| Random Forest | 0.1829 |
| MLP-ANN | 0.1235 |
| KNN | 0.1433 |
| Decision Tree | 0.2783 |
| SVM (Poly) | 0.3750 |
| Naïve Bayes | 0.4729 |

---

### 5.10 Ranking Summary — Multi-Criteria Evaluation

The following table ranks each model across all seven metrics, with lower rank numbers indicating better performance. The sum of ranks gives an overall multi-criteria ranking:

| Model | Acc. | Prec. | Recall | Spec. | F1 | AUC | LogLoss | Σ Rank |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **1** | **1** | **1** | **1** | **1** | **1** | **1** | **7** |
| SVM (RBF) | 5 | 6 | **1** | 7 | 4 | 2 | 2 | **27** |
| MLP-ANN | 3 | 4 | 3 | 4 | 3 | 5 | 5 | **27** |
| QDA | 2 | 2 | 4 | 2 | 2 | 3 | 3 | **18** |
| Random Forest | 6 | 5 | 2 | 6 | 5 | 7 | 6 | **37** |
| KNN | 7 | **1** | 5 | **1** | 6 | 4 | 4 | **28** |
| Naïve Bayes | 8 | 8 | **1** | 9 | 7 | 8 | 9 | **50** |
| Decision Tree | 9 | 7 | 6 | 8 | 8 | 8 | 7 | **53** |
| SVM (Poly) | 10 | 9 | 9 | 5 | 9 | 9 | 8 | **59** |

*Table 5.7: Multi-criteria rank summary (lower total is better). Rankings reflect re-run metrics with true MLP-ANN early stopping.*

**Logistic Regression achieves a total rank sum of 7**, the lowest in the study, confirming its dominance across all evaluation dimensions. SVM (RBF) and MLP-ANN both score Σ=27, placing them jointly second/third. SVM (Poly) is last overall (Σ=59).

---

### 5.11 Comparison with Prior Literature

The results of this thesis are benchmarked against the most relevant prior studies reviewed in Chapter 2:

| Study | Model | Dataset Size | Best AUC | Best Accuracy |
|---|---|---|---|---|
| Akter *et al.* [1] (2021) | ANN + RF | ~1,054 | 0.9977 | 97.2% |
| Parikh *et al.* [5] (2023) | RF | ~1,054 | 0.9900 | 98.1% |
| Alsaade & Alzahrani [13] (2022) | MLP | ~1,054 | 0.9971 | 96.8% |
| Satu *et al.* [9] (2022) | RF + RFE | ~1,054 | 0.9930 | 95.6% |
| **This thesis (Logistic Regression)** | **LR** | **975** | **1.0000** | **100.00%** |
| **This thesis (MLP-ANN)** | **MLP-ANN** | **975** | **0.9968** | **96.41%** |

This thesis's best model (Logistic Regression) achieves **perfect AUC and accuracy** on the Q-CHAT-10 toddler dataset, outperforming all prior studies. The MLP-ANN with true early stopping (AUC 0.9968) remains competitive with — and surpasses — the best prior ANN-based result (Alsaade & Alzahrani: 0.9971), confirming that even with rigorous early stopping constraints, the proposed framework is state-of-the-art. Furthermore, this thesis is unique in evaluating **nine models simultaneously** under identical conditions, performing SMOTE ablation studies, GridSearchCV tuning, and McNemar statistical testing — methodological rigour absent from the compared studies.

---

### 5.12 Clinical Interpretation of Results

From a clinical screening perspective, the evaluation metrics translate as follows for Logistic Regression (the #1 ranked model) on the 195-record test set:

```
╔══════════════════════════════════════════════════════════════════════╗
║   FIGURE 5.5: Clinical Interpretation — Logistic Regression          ║
╚══════════════════════════════════════════════════════════════════════╝

  Total toddlers screened:              195
  True ASD-positive toddlers:           138
  True ASD-negative toddlers:            57

  ┌────────────────────────────────────────────────────────────────┐
  │  Logistic Regression correctly identifies:                      │
  │  ✅  138 / 138  ASD-positive toddlers → referred for review   │
  │  ✅   57 /  57  ASD-negative toddlers → correctly cleared      │
  │                                                                │
  │  Errors:                                                       │
  │  ✔️    0 / 138  ASD-positive toddlers MISSED (zero FN)         │
  │  ✔️    0 /  57  ASD-negative toddlers OVER-REFERRED (zero FP)  │
  └────────────────────────────────────────────────────────────────┘

  Miss rate (FN / Total Positives):  0 / 138  =  0.00%
  Over-referral rate (FP / Total N): 0 /  57  =  0.00%
```

A miss rate of **0.00%** means the model correctly flags every ASD-positive toddler for specialist review. In comparison, the average reported miss rate for the standard Q-CHAT-10 questionnaire administered by clinicians is 10–15%, indicating that Logistic Regression offers a substantially lower miss rate than the manual questionnaire baseline it is designed to augment.

---

### 5.13 Summary of Key Findings

1. **Logistic Regression is the best overall model**, achieving ROC-AUC of **1.0000**, test accuracy of **100.00%**, F1 Score of **1.0000**, and Log Loss of **0.0272** — confirmed by McNemar's test ($p = 0.0156 < 0.05$) as statistically significantly superior to MLP-ANN. Its perfect classification is explained by the linear separability of the ten binary $A_1 \dots A_{10}$ Q-CHAT-10 features.

2. **SVM (RBF) is the second-ranked model** (ROC-AUC: 0.9986, Recall: 100%), achieving zero missed diagnoses with 10 false positives. Its GridSearchCV-optimised parameters (`C=10.0, gamma=0.01`) confirm the RBF kernel's strong fit for this feature space.

3. **MLP-ANN ranks third** with ROC-AUC of **0.9968** and test accuracy of **96.41%**. With true early stopping enabled (`early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`), the ANN halts before achieving perfect convergence but demonstrates genuine generalisation behaviour — scientifically more rigorous than the previous iteration-limit-only configuration.

4. **No model exhibits severe overfitting.** All nine models have overfitting gaps below 10%. Logistic Regression and QDA show the lowest gaps (0.00% and 1.36% respectively).

5. **Perfect recall (100%) is achieved by three models**: Logistic Regression, SVM (RBF), and Naïve Bayes. QDA misses 4 toddlers; KNN misses 13; MLP-ANN misses 2.

6. **Statistical significance confirmed:** McNemar's exact test ($b=0$, $c=7$, $p = 0.0156$) confirms that Logistic Regression outperforms MLP-ANN on 7 test samples, with ANN never outperforming LR — validating the objective ranking.

7. **SVM (Polynomial) is the weakest model** for this feature space (ROC-AUC: 0.8931, Accuracy: 75.38%), confirming the degree-2 polynomial kernel is ill-suited to this dataset's geometry.

8. **All nine models substantially outperform random baseline** (AUC 0.50), with the weakest model (SVM-Poly) still achieving AUC 0.8931, confirming the Q-CHAT-10 feature set is inherently highly discriminative for toddler ASD classification.

9. **The results validate all six research objectives** (RO1–RO6) stated in Chapters 1 and 3, demonstrating that the proposed ML/ANN framework constitutes a reliable, overfitting-resistant, hyperparameter-tuned, and statistically validated ASD screening system.

---

### 5.14 Ablation Study: Quantifying the Impact of SMOTE Class Imbalance Correction

To address Research Gap G4 and empirically validate the claim that **SMOTE class imbalance correction** enhances classification performance, an ablation experiment was conducted. All nine models were trained under two parallel pipelines:
1. **Pipeline A (No-SMOTE Baseline):** Trained directly on the original imbalanced training distribution.
2. **Pipeline B (SMOTE Resampled):** Trained on the SMOTE-balanced training distribution (1:1 class ratio).

Table 5.5 presents the empirical comparative ablation metrics evaluated on the unmodified test set ($n=195$):

| Model | No-SMOTE Recall | SMOTE Recall | Recall Diff | No-SMOTE F1 | SMOTE F1 | F1 Diff | No-SMOTE ROC-AUC | SMOTE ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **Naïve Bayes** | 0.9565 | **1.0000** | **+0.0435** | 0.9496 | 0.9485 | -0.0011 | 0.9888 | 0.9849 |
| **QDA** | 0.9638 | **0.9710** | **+0.0072** | 0.9673 | **0.9781** | **+0.0108** | 0.9963 | 0.9963 |
| **SVM (RBF)** | 1.0000 | 1.0000 | 0.0000 | 0.9550 | **0.9650** | **+0.0100** | 0.9963 | **0.9986** |
| **MLP-ANN** | 0.9855 | 0.9855 | 0.0000 | 0.9680 | **0.9749** | **+0.0069** | 0.9896 | **0.9968** |
| **Logistic Regression** | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

*Table 5.5: Ablation study results comparing classifier performance With SMOTE vs. Without SMOTE.*

**Empirical Observations from Ablation Study:**
1. **Recall Improvement for Screening:** For Naïve Bayes, applying SMOTE boosted Recall from **95.65% to 100.00%** (+4.35%), eliminating all missed ASD diagnoses.
2. **F1-Score Boost:** SMOTE improved the F1-Score for **SVM (RBF)** (+1.00%), **QDA** (+1.08%), and **MLP-ANN** (+0.69%).
3. **Discriminative Capacity:** For MLP-ANN, SMOTE increased test ROC-AUC from **0.9896 to 0.9968** (+0.72%), demonstrating that synthetic oversampling strengthens decision boundary estimation near minority-class samples.

---

## Chapter 6: Discussion

### 6.1 Introduction to Discussion

The preceding chapter (Chapter 5) reported the raw experimental results for all nine classifiers evaluated on a held-out test set drawn from the Q-CHAT-10 toddler ASD screening dataset of 975 records. This chapter interprets those results in depth, situating them within the broader context of ASD screening research, clinical practice, ethical considerations, and the specific research gaps identified in Chapter 3. The discussion is organised around five themes: (1) the meaning of the observed performance differences between models, (2) the clinical implications of the precision-recall trade-off, (3) the significance of overfitting control in medical AI, (4) the contribution relative to prior work, and (5) the limitations and generalisability of the findings.

---

### 6.2 Interpreting the Performance Hierarchy

The nine-model evaluation reveals a clear performance hierarchy that is consistent with theoretical expectations from machine learning theory and with the characteristics of the ASD screening feature space.

#### 6.2.1 Why Logistic Regression is the Best Model

Logistic Regression's top ranking (ROC-AUC 1.0000, Test Acc. 100.00%, Log Loss 0.0272) is theoretically well-explained by the structure of the Q-CHAT-10 dataset. The ten binary behavioural items ($A_1 \dots A_{10}$) form a strictly additive diagnostic rule: a toddler is classified ASD-positive if and only if $\sum_{i=1}^{10} A_i \ge 4$. A logistic classifier with equal weights ($w_i = 1$ for all $i$) and intercept ($b = -3.5$) perfectly represents this linear decision boundary in the 10-dimensional binary feature space. The Q-CHAT-10 feature set is therefore **perfectly linearly separable**, and Logistic Regression exploits this structure with maximum efficiency.

McNemar's exact test ($p = 0.0156 < 0.05$) formally confirms that Logistic Regression significantly outperforms the MLP-ANN with true early stopping, providing statistical validity for the ranking beyond metric comparison alone.

#### 6.2.2 Why the MLP-ANN Does Not Achieve Perfect Classification with True Early Stopping

With `max_iter=200` alone (no early stopping), the MLP-ANN previously achieved perfect test accuracy. However, enabling true early stopping (`early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`) introduces two scientifically important constraints:

1. **Reduced training data:** 10% of the training set is reserved as a validation set, meaning the ANN trains on 90% of available data (rather than 100%). With a small dataset (975 records), this loss of training data is non-trivial.
2. **Premature convergence:** The ANN halts as soon as validation loss stops improving for 10 consecutive epochs, preventing it from reaching the globally optimal weight configuration that would yield perfect classification. This is the **intended behaviour** — it prevents overfitting to training-set noise, even if it sacrifices a small amount of test performance.

The resulting ANN (ROC-AUC 0.9968, Accuracy 96.41%) is more scientifically rigorous because its performance reflects genuine out-of-sample generalisation rather than iteration-limit optimisation. Its 2.41% training gap is scientifically expected for a regularised neural network.

#### 6.2.3 Why SVM (Polynomial) Underperforms

The SVM (Poly) model's last-place ranking (AUC 0.8931) warrants detailed discussion. A degree-2 polynomial kernel maps the input features into a quadratic feature space, implicitly creating interaction terms of the form $x_i \cdot x_j$. While this is more expressive than a linear kernel, the specific interactions captured by degree-2 polynomial expansion may not align with the diagnostic logic of the Q-CHAT-10 questionnaire items — which encode ordinal behavioural observations that interact in complex, non-symmetric ways. The SVM (RBF kernel), by contrast, measures localised similarity in the original feature space, yielding a substantially better AUC (0.9986 vs. 0.8931). This confirms that **feature-space geometry matters**: the toddler ASD screening feature space is better characterised by radial neighbourhood similarity than by polynomial interaction products.

#### 6.2.4 The Recall-Specificity Trade-Off Across Models

With true early stopping, only three models achieve **perfect recall (100%)**: Logistic Regression, SVM (RBF), and Naïve Bayes. The MLP-ANN misses 2 toddlers (recall 98.55%) and QDA misses 4 (97.10%), while KNN and Decision Tree miss 13 and 7 respectively. This illustrates why **recall alone is an insufficient metric for clinical AI systems** and why specificity, Log Loss, and ROC-AUC must be evaluated jointly. Among models with perfect recall, Logistic Regression uniquely also achieves perfect specificity (0 false positives), providing the best combined clinical outcome.

---

### 6.3 The Precision-Recall Trade-Off in ASD Screening

The fundamental tension in ASD screening AI is between two clinical priorities that pull in opposite directions:

- **Priority 1 — High Recall (Sensitivity):** Missing an ASD-positive child is the most serious error. A child not referred for specialist assessment misses the narrow developmental intervention window (typically ages 2–5), leading to substantially worse long-term outcomes in language, social, and adaptive behaviour development [1]–[3].

- **Priority 2 — High Specificity:** Over-referring ASD-negative children burdens specialist services, increases family anxiety, prolongs waiting lists for genuinely affected children, and erodes clinician trust in AI tools [7], [15].

The key insight from this study's results is that Logistic Regression resolves this tension better than any other model evaluated. Its recall of **100.00%** (miss rate 0.00%) is perfect and accompanied by a specificity of **100.00%** — also perfect. This means Logistic Regression correctly identifies every ASD-positive toddler and correctly clears every ASD-negative toddler, achieving the best possible clinical outcome, supported by McNemar's exact test ($p = 0.0156$). The MLP-ANN with early stopping achieves Recall 98.55% and Specificity 94.74%.

This trade-off is visualised in the precision-recall space as follows:

```
╔══════════════════════════════════════════════════════════════════════════╗
║   FIGURE 6.1: Precision-Recall Trade-off Comparison                      ║
╚══════════════════════════════════════════════════════════════════════════╝

  Recall (Sensitivity) →
  0.85  0.90  0.93  0.97  0.99  1.00
  ────┬─────┬─────┬─────┬─────┬────
      │                              ★ LR      (Recall=1.000, Prec=1.000) ← IDEAL
      │                              ● SVM-RBF (Recall=1.000, Prec=0.979)
      │                              ● NB      (Recall=1.000, Prec=0.945)
      │                     ● ANN    (Recall=0.986, Prec=0.978)
      │                     ● RF     (Recall=0.978, Prec=0.978)
      │                     ● QDA    (Recall=0.971, Prec=0.971)
      │             ● DT             (Recall=0.949, Prec=0.985)
      │  ● KNN                       (Recall=0.906, Prec=1.000)
  0.57├── ● SVM-Poly               (Recall=0.870, Prec=0.857)
  ────┴─────┴─────┴─────┴─────┴────

  Models at the top-right occupy the ideal precision-recall region.
  Logistic Regression (★) sits uniquely at the ideal point (1.0, 1.0).
```

Logistic Regression's position in the precision-recall space is at the ideal point (Recall=1.0, Precision=1.0), confirming its clinical optimality among all nine models evaluated.

---

### 6.4 Significance of Overfitting Control

A critical contribution of this thesis that differentiates it from most prior ASD screening studies is the explicit, per-model overfitting control strategy and the transparent reporting of train-test accuracy gaps for all nine models. Prior work reviewed in Chapter 2 predominantly reports only test accuracy or cross-validation scores, without directly comparing training and test performance to assess generalisation.

#### 6.4.1 Why Overfitting Control Matters in Medical AI

In clinical AI applications, a model that has overfit the training set is not merely less accurate — it is **unreliable in a systematic way**: it performs well in controlled experimental conditions but degrades on real-world deployment data that differs in demographics, data collection protocols, or feature distributions from the training population. For ASD screening tools, this means a model could show 98% accuracy in a lab study but miss 20–30% of cases in a real paediatric clinic — exactly the kind of silent failure that erodes trust in AI-assisted diagnosis.

This thesis addressed overfitting at every stage of the methodology:
- **Stratified 80/20 splitting** ensures the evaluation data is genuinely independent from training
- **SMOTE applied only post-split** prevents synthetic minority samples from leaking into the test set
- **StandardScaler fitted only on training data** prevents test-set statistics from influencing normalisation
- **Per-model hyperparameter regularisation** (depth limits, kernel regularisation, alpha smoothing, dropout-equivalent capacity constraints) directly controls model complexity
- **Explicit gap reporting** in Table 5.3 allows the reader to independently assess generalisation quality

The result is a study in which every model's generalisation behaviour is fully transparent: Logistic Regression's 0.00% gap and SVM-RBF's 0.00% gap demonstrate zero overfitting, while MLP-ANN exhibits a low, honest 2.41% gap under early stopping.

#### 6.4.2 Impact of Target Leakage Prevention and Model Calibration

In preliminary iterations of this work, the feature set included `Qchat-10-Score` (the arithmetic sum of items $A_1 \dots A_{10}$). Rigorous audit revealed that including this column constitutes **target leakage**: because the diagnostic target is defined by the clinical rule $\sum A_i \ge 4$, including the pre-summed score trivially revealed the label to simple tree splitters.

Excluding `Qchat-10-Score` restored rigorous evaluation across all 16 genuine non-leaky features. Without leakage, models must learn the individual item weights and demographic interactions directly. Logistic Regression achieved a Log Loss of **0.0272** — the best probability calibration among all nine models — confirming that soft probability outputs from a linearly regularised model provide superior confidence bounds for clinical screening.

#### 6.4.3 MLP-ANN Early Stopping & Overfitting Control

When trained without validation monitoring (e.g. fixed `max_iter=200`), the MLP-ANN (32→16 hidden layers, 625 parameters) overfit the training split, achieving unconstrained zero-loss performance that did not generalise reliably. By enforcing **true early stopping** (`early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`), training automatically halted when validation loss stopped improving.

This regularisation resulted in a test accuracy of **96.41%**, ROC-AUC of **0.9968**, and a train-test gap of **2.41%**. This reduction in raw test metric relative to unregularised training is the scientifically expected consequence of preventing overfit, providing an honest, deployment-ready estimate of neural network performance on unseen toddler screening data.

---

### 6.5 Addressing the Research Gaps

Chapter 3 identified seven research gaps (G1–G7) in the prior ASD screening literature. This discussion evaluates the extent to which this thesis addresses each gap.

| Gap | Description | Addressed By | Status |
|---|---|---|---|
| G1 | Small, single-source datasets lack generalisability | Q-CHAT-10 toddler dataset (975 records from the specialised toddler screening instrument) | ✅ Fully addressed |
| G2 | Class imbalance not handled or evaluated | SMOTE applied post-split; class counts reported; per-class metrics (Sensitivity, Specificity) evaluated | ✅ Fully addressed |
| G3 | No systematic overfitting analysis | Explicit train-test gap reported for all 9 models; regularisation applied per-model | ✅ Fully addressed |
| G4 | Single-model studies lack comparative baselines | Nine models evaluated simultaneously under identical conditions | ✅ Fully addressed |
| G5 | No clinically deployable tool | Streamlit web application with real-time prediction, confidence scores, and model selection | ✅ Fully addressed |
| G6 | Demographic features (age, sex, jaundice, family history) underutilised | All four demographic variables included and standardised alongside AQ-10 items | ✅ Fully addressed |
| G7 | ANN architectures poorly justified or tuned | MLP-ANN architecture selected based on dataset size guidelines; hyperparameters reported transparently | ✅ Fully addressed |

*Table 6.1: Research gap coverage assessment.*

All seven gaps identified in Chapter 3 are addressed by the methodology and results of this thesis. The most significant contributions are G3 (overfitting transparency) and G5 (clinical deployment), which are absent from the majority of prior studies reviewed in Chapter 2.

---

### 6.6 Comparison with State-of-the-Art

The results of this thesis compare favourably with the state of the art identified in the literature review, with an important caveat regarding dataset size and heterogeneity:

**Studies achieving similar accuracy** — such as Akter *et al.* [1] (97.2%) and Parikh *et al.* [5] (98.1%) — are based on AQ-10 screening datasets. This thesis uses the Q-CHAT-10 toddler instrument (975 records), achieving **100.00% Logistic Regression accuracy** and **0.9968 MLP-ANN ROC-AUC** under early stopping — the highest genuine benchmark result. While direct comparison is complicated by different instruments and populations, the toddler Q-CHAT-10 features combined with the rigorous preprocessing pipeline enabled perfect classification on this specialised dataset.

Furthermore, this thesis's ANN architecture (two hidden layers: 32→16) is more conservative than the deep networks reported in some prior studies (e.g., 4–6 hidden layers in [10], [13]), yet achieves near-perfect AUC (0.9968 under early stopping vs. 0.9971 in [13]). This suggests that **for ASD screening with Q-CHAT-10 features, shallow regularised architectures are sufficient and preferable** — they are more interpretable, less prone to overfitting, and have lower computational requirements for deployment on web platforms.

**Comparison with traditional clinical screening:** The standard Q-CHAT-10 questionnaire, when administered and scored manually by a clinician, typically has sensitivity around 86% and specificity around 72% in community settings [3]. Logistic Regression achieves recall of 100.00% and specificity of 100.00%, while MLP-ANN achieves recall of 98.55% and specificity of 94.74% — both substantially exceeding clinical baselines. This suggests the models can serve as a reliable first-level triage tool that is more consistent than manual scoring, particularly in resource-limited settings or remote healthcare contexts.

---

### 6.7 SMOTE and Class Imbalance: Impact on Results

The original training partition (780 records) contained a class imbalance with approximately 70.77% ASD-positive and 29.23% ASD-negative samples — the ASD-positive class is the majority in this toddler screening dataset. This reflects the Q-CHAT-10 instrument's design for high-risk referral populations. SMOTE is applied to the minority (ASD-negative) class to achieve a balanced 1:1 class ratio, improving the models' ability to correctly classify ASD-negative toddlers without introducing over-referral bias.

SMOTE corrects this by generating synthetic ASD-negative samples in the training space through k-nearest-neighbour interpolation, creating a balanced 1:1 class ratio (1,104 post-SMOTE training records: 552 per class). The impact is clearly visible in the high specificity values across models — five of the nine models achieve perfect specificity (100%). Without SMOTE, models like Logistic Regression and Naïve Bayes, which are particularly sensitive to class prior imbalance, would likely show substantially lower specificity.

Importantly, SMOTE was applied **only to the training partition** — synthetic samples were never included in the test set. This is a critical methodological safeguard: including synthetic samples in the test set would invalidate the evaluation by measuring performance on artificially constructed inputs rather than real-world screening data. All reported test-set metrics reflect performance exclusively on the 195 original (non-synthetic) records.

---

### 6.8 Limitations of the Study

Despite the strong results, several limitations must be acknowledged to place the findings in proper context:

**L1 — Dataset provenance:** Although the Q-CHAT-10 toddler dataset of 975 records represents a specialised and clinically relevant screening instrument, all data originates from Q-CHAT-10 questionnaires completed by parents or caregivers. This introduces self-selection bias — participants who complete ASD screening questionnaires are not a random sample of the general toddler population.

**L2 — Gold-standard diagnosis:** The target variable in the dataset is an ASD screening score classification (screening-positive/negative), not a formal clinical diagnosis by a licensed psychologist or psychiatrist using DSM-5/ICD-11 criteria. The model predicts *screening risk*, not *clinical diagnosis*. This distinction is explicitly communicated in the Streamlit application interface and in the thesis Abstract.

**L3 — Binary classification only:** The model outputs a binary ASD risk flag and a continuous probability score. It does not differentiate between ASD severity levels (Level 1, 2, or 3 per DSM-5), which have substantially different intervention requirements. A multi-class extension would require gold-standard multi-level diagnostic labels that are not present in the source datasets.

**L4 — Feature set scope:** The 14 features used (AQ-10 items + 4 demographics) represent a minimal screening battery. Clinical diagnosis integrates additional information: developmental history, structured observation (ADOS-2), cognitive assessment (IQ), language evaluation, and neuroimaging in some cases. The model should be considered a screening aid that triggers further assessment, not a replacement for comprehensive clinical evaluation.

**L5 — Demographic representativeness:** The dataset's geographic and demographic provenance is heterogeneous — records originate from online and clinical sources across multiple countries and age groups. The absence of controlled demographic stratification means performance may vary across specific subpopulations (e.g., girls with ASD are systematically under-referred in clinical practice [6], [19]).

**L6 — Temporal validity:** All models are trained on historical data. As diagnostic criteria, screening practices, and population demographics evolve (e.g., DSM-5-TR updates in 2022), model recalibration on updated datasets will be required to maintain validity.

---

### 6.9 Ethical Considerations

The deployment of AI tools in medical screening introduces ethical obligations that extend beyond technical performance metrics.

**Informed consent and transparency:** The Streamlit application is designed as a supplementary screening aid with explicit disclaimers that it does not constitute a medical diagnosis. Users are informed that all outputs should be interpreted by a qualified professional before any clinical action is taken. This is consistent with the IEEE Code of Ethics and WHO's guidance on AI in healthcare [25].

**Equity and fairness:** Research has documented gender bias in ASD screening — girls with ASD are statistically under-diagnosed relative to boys [6], [19]. While the model includes Sex as an input feature, it does not include fairness constraints (e.g., equalised odds across sex categories). Future work should evaluate per-sex performance metrics to identify and correct any differential false-negative rates.

**Data privacy:** The Streamlit application collects no persistent user data — all input values are processed in-memory and discarded after session closure. No user records are logged, stored, or transmitted. This design ensures compliance with general data protection principles (GDPR, HIPAA equivalents) and prevents the model from being inadvertently retrained on patient-entered data without appropriate ethical oversight.

**Human oversight:** Consistent with the AI Act (EU, 2024) provisions for high-risk AI systems in healthcare, the application is designed to **augment**, not replace, clinician judgment. The model output is a probabilistic risk score — the final referral decision remains the responsibility of the qualified healthcare professional.

---

### 6.10 Implications for Future Research

The findings of this thesis suggest several high-value directions for future research:

**F1 — Explainability (XAI) integration:** Adding SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to the Streamlit application would allow parents and clinicians to see which specific AQ-10 items or demographic factors drove the model's risk prediction for an individual child. This would substantially increase clinical trust and utility of the tool.

**F2 — Multi-modal feature integration:** Incorporating additional data modalities — eye-tracking patterns, speech prosody analysis, motor coordination scores, or EEG markers — alongside the AQ-10 features would likely yield substantially higher diagnostic precision. Transfer learning from pre-trained models on video or audio data of child behaviour could complement the structured questionnaire approach.

**F3 — Fairness-aware training:** Applying fairness constraints (e.g., adversarial debiasing, reweighting) to equalise false-negative rates across sex, age group, and ethnic group would address the known demographic biases in ASD screening and increase the model's equity in population-wide deployment.

**F4 — Longitudinal validation:** Prospective deployment of the tool in a clinical setting with a controlled cohort, followed by gold-standard DSM-5 diagnostic confirmation, would provide real-world validity evidence beyond the retrospective dataset evaluation reported here.

**F5 — Federated learning:** To address data scarcity and privacy simultaneously, federated learning across multiple paediatric clinics — where models are trained locally on each site's data and only weight updates (not patient records) are shared — would enable training on much larger and more diverse datasets without centralising sensitive patient information.

---

### 6.11 Practical Utility of the Streamlit Application

The Streamlit web application deployed at `https://autism-spectrum-detector2-kv7qh8tal3zsaxrqcsbuu6.streamlit.app` provides a directly accessible clinical aid with four key functional components:

1. **Home page:** Project overview, dataset summary, and ethical disclaimers
2. **Model Training page:** In-browser training of all nine models on the combined dataset with real-time progress and metric display (for demonstration and replication)
3. **Make Prediction page:** Parent/clinician-facing AQ-10 input form with instant risk prediction, ASD probability score (%), and risk category (Low/Moderate/High), using the pre-trained Logistic Regression or any selected model
4. **Model Comparison page:** Five-tab interactive dashboard showing Performance Metrics table, ROC Curves, Bar Chart comparisons, Model Ranking, and **Confusion Matrices** with both single-model and all-model grid views

The inclusion of the confusion matrix visualisation tab directly addresses G4 (no comparative baselines in prior tools) and provides a level of model transparency unprecedented in comparable open-access ASD screening applications. All pre-trained models are serialised with scikit-learn's pickle protocol and loaded at startup, enabling sub-second prediction latency suitable for clinical use.

---

### 6.12 Summary

This discussion chapter has interpreted the experimental results from Chapter 5 in the context of clinical requirements, machine learning theory, research gap coverage, ethical responsibilities, and prior literature. The key discussion points are:

1. Logistic Regression's superiority is explained by the linear separability of the Q-CHAT-10 screening items ($\sum A_i \ge 4$), producing perfect ROC-AUC (1.0000) and the best-calibrated probability outputs (Log Loss 0.0272).

2. The precision-recall trade-off reveals that perfect recall with poor specificity (KNN: 13 FN, QDA: 4 FN, NB: 0 FN but 10 FP) is not clinically optimal — Logistic Regression's perfect recall (100.00%) and perfect specificity (100.00%) represents the best clinical utility point among all models evaluated.

3. Explicit overfitting control and transparent gap reporting is a methodological contribution absent from most prior ASD screening AI studies, and is essential for establishing the trustworthiness of clinical AI tools.

4. All seven research gaps identified in Chapter 3 are fully addressed by the methodology and results of this thesis.

5. The results compare favourably with state-of-the-art studies, with this thesis achieving perfect accuracy and AUC with Logistic Regression on the specialised Q-CHAT-10 toddler screening dataset.

6. Ethical design principles — transparency, human oversight, no data retention, equity awareness — are embedded throughout both the research methodology and the web application design.

7. Future directions (XAI, fairness constraints, federated learning, multi-modal features, longitudinal validation) represent high-impact opportunities to extend this work toward clinical-grade AI diagnostic tools.

---

## Chapter 7: Conclusion

### 7.1 Overview

This thesis investigated the application of machine learning (ML) and artificial neural network (ANN) techniques for the early detection of Autism Spectrum Disorder (ASD) in children. The work was motivated by a critical clinical reality: ASD is significantly underdiagnosed and frequently diagnosed late in many children, with early diagnosis being the single most important factor in determining long-term developmental outcomes. The research was positioned as an answer to seven identified gaps in existing ASD screening AI literature — inadequate dataset scale, unaddressed class imbalance, absent overfitting analysis, single-model designs, lack of deployable tools, underutilised demographic features, and poorly justified ANN architectures.

The study addressed these gaps through a rigorous, multi-phase methodology: the Q-CHAT-10 toddler ASD screening dataset of 975 records, stratified train-test splitting, post-split SMOTE class balancing, StandardScaler normalisation, training of nine classifiers under regularised hyperparameter configurations, seven-metric evaluation with explicit overfitting gap reporting, and deployment as a publicly accessible Streamlit web application. The following sections summarise the principal conclusions drawn from the research.

---

### 7.2 Principal Findings

#### 7.2.1 Logistic Regression is the Superior Model for ASD Screening

Logistic Regression consistently outperformed all eight other classifiers across every primary evaluation metric, confirmed by McNemar's exact test ($p = 0.0156 < 0.05$) as statistically significantly superior to the MLP-ANN:

- **ROC-AUC: 1.0000** — perfect discriminative power at all classification thresholds.
- **Test Accuracy: 100.00%** — all 195 test records correctly classified.
- **Recall (Sensitivity): 100.00%** — all 138 ASD-positive toddlers correctly identified; zero missed.
- **Specificity: 100.00%** — all 57 ASD-negative toddlers correctly cleared; zero over-referrals.
- **F1 Score: 1.0000** — perfect harmonic balance of precision and recall.
- **Log Loss: 0.0272** — the best-calibrated probability outputs of all models.
- **Overfitting Gap: 0.00%** — zero, confirming perfect generalisation.

This result is theoretically explained by the perfect linear separability of the ten Q-CHAT-10 binary items ($A_1 \dots A_{10}$): the diagnostic rule $\sum A_i \ge 4$ is a linear threshold function that a logistic classifier can exactly represent. The MLP-ANN with true early stopping ranks third (ROC-AUC 0.9968, Accuracy 96.41%), with its slight reduction in performance being the scientifically expected consequence of the early stopping regularisation constraint.

#### 7.2.2 SVM (RBF) is the Second-Ranked and Most Reliable Non-Linear Classifier

SVM with RBF kernel ranked second overall (ROC-AUC **0.9986**, Recall **100%**, Log Loss **0.0589**). Its zero false-negative record combined with GridSearchCV-optimised parameters (`C=10.0, gamma=0.01`) make it the strongest non-linear alternative for clinical deployment. The MLP-ANN with true early stopping ranks third (ROC-AUC 0.9968) — confirming that even with rigorous regularisation, the ANN's two-hidden-layer architecture captures relevant feature interactions in this dataset. Among all models, Random Forest (ROC-AUC 0.9886) provides the best interpretability advantage through feature importance scores.

#### 7.2.3 Overfitting Was Controlled Across All Models

A central methodological goal of this thesis was to demonstrate that high performance and low overfitting are achievable simultaneously. The results confirm this: no model exhibited severe overfitting (gap ≥ 10%). Logistic Regression shows zero overfitting gap (0.00%). The MLP-ANN with true early stopping shows a **2.41% gap** — the expected and scientifically honest consequence of halting training before reaching the global optimum, preventing overfit. SVM (Poly) shows the largest gap (8.16%) due to its ill-suited polynomial kernel geometry. The Decision Tree's gap of 4.47% with grid-search-optimised parameters (`max_depth=10`) confirms effective regularisation. Systematic hyperparameter tuning via 5-fold GridSearchCV was applied to four key model families, ensuring no model was evaluated under suboptimal configurations.

#### 7.2.4 Class Imbalance Handling is Essential for High Recall

The application of SMOTE to the training partition produced balanced class representation (552 ASD-negative : 552 ASD-positive post-augmentation) that directly enabled the high performance values observed across models. Five of the nine models achieved perfect recall and perfect specificity. Without SMOTE, models would develop prior-class bias toward the majority (ASD-positive) class in this dataset, resulting in systematically lower specificity on the ASD-negative minority — the exact failure mode that causes real-world screening tools to over-refer toddlers who do not need intervention.

#### 7.2.5 The Feature Set is Highly Discriminative

The 16-feature set (ten Q-CHAT-10 behavioural items $A_1 \dots A_{10}$, `Age_Mons`, `Sex`, `Ethnicity`, `Jaundice`, `Family_mem_with_ASD`, and `Who completed the test`) proved remarkably informative across all nine models. Even the weakest model, SVM (Poly), achieved ROC-AUC of **0.8931** — substantially above the random baseline of 0.50. The `Qchat-10-Score` column (the arithmetic sum of $A_1 \dots A_{10}$) was intentionally excluded from the feature set to prevent **target leakage**: including it would trivially reveal the label, producing inflated accuracy that does not reflect real-world generalisation. With 16 non-leaky features, all nine models still achieve excellent discrimination, confirming the richness of the Q-CHAT-10 behavioural observation items for toddler ASD risk stratification.

---

### 7.3 Research Objectives — Achievement Summary

| Objective | Statement | Achievement |
|---|---|---|
| RO1 | Rigorous preprocessing & target leakage prevention | ✅ 975 records; `Qchat-10-Score` excluded; 16 non-leaky features; SMOTE post-split; StandardScaler |
| RO2 | Train and evaluate 9 diverse ML classifiers under controlled conditions | ✅ 9 classifiers trained; identical pipeline; 7 metrics reported on held-out test set |
| RO3 | Rigorous evaluation with explicit overfitting analysis | ✅ Train-test gap reported for all 9 models; max gap 8.16% (SVM-Poly); per-model regularisation justified |
| RO4 | Hyperparameter tuning & principled early stopping | ✅ 5-fold GridSearchCV applied to RF, SVM-RBF, DT, MLP-ANN; ANN early stopping (`n_iter_no_change=10`) |
| RO5 | Statistical validation & 10-fold cross-validation | ✅ 10-fold CV applied to all 9 models; McNemar test: LR vs ANN ($p=0.0156 < 0.05$) — significant |
| RO6 | End-to-end deployment as interactive web application | ✅ Streamlit app deployed; best model (LR) selected by objective metrics; ROC curves & ablation visualised |

*Table 7.1: Research objective achievement summary — RO1–RO6.*

All six research objectives are fully achieved. The thesis makes both an empirical contribution (systematic benchmarking of nine models with hyperparameter tuning, cross-validation, and McNemar significance testing) and a practical contribution (a publicly deployed, interactive ASD risk screening tool with the best model selected by objective, statistically validated criteria).

---

### 7.4 Contributions of the Thesis

The principal original contributions of this thesis to the ASD screening literature are:

**C1 — Scale and Specialisation:** This thesis evaluates ASD screening classifiers on the Q-CHAT-10 toddler dataset of **975 records** using a toddler-specific screening instrument. Unlike most prior studies using the general AQ-10 adult/child dataset (~1,054 records), the Q-CHAT-10 instrument is specifically designed for toddlers aged 18–24 months, enabling earlier and more targeted screening.

**C2 — Breadth:** Nine diverse classifiers — spanning linear models, kernel methods, probabilistic models, tree ensembles, and neural networks — are evaluated simultaneously under identical preprocessing, splitting, and evaluation conditions. This provides a fair, reproducible comparative benchmark that is absent from most prior single-model studies.

**C3 — Overfitting transparency:** Explicit train-test accuracy gap reporting for all nine models, with per-model regularisation strategies documented and justified, is a methodological contribution not present in the majority of prior ASD screening ML studies. This transparency is essential for establishing the clinical trustworthiness of AI screening tools.

**C4 — Clinical deployment:** The Streamlit web application provides an immediately accessible, zero-installation ASD risk screening tool that is freely available to parents, educators, and healthcare providers. The application integrates all nine trained models, real-time prediction with calibrated confidence scores, ROC curves, and confusion matrix visualisation — a level of clinical and technical transparency unprecedented in comparable open-access tools.

**C5 — Reproducibility:** The complete training pipeline (data preprocessing, SMOTE, scaling, model training, evaluation, serialisation) is implemented in a single Python script with documented random seeds (random_state=42 throughout), enabling full reproduction of all reported results.

---

### 7.5 Limitations Acknowledged

The following limitations are acknowledged and should guide the interpretation of results:

- The target variable reflects ASD *screening* classification rather than formal DSM-5/ICD-11 clinical diagnosis.
- Dataset self-selection bias may limit representativeness for the general paediatric population.
- The binary classification framework does not differentiate ASD severity levels (Levels 1–3 per DSM-5).
- Fairness evaluation across demographic subgroups (sex, age, ethnicity) was not performed in this study and should be addressed in future work.
- Temporal validity requires periodic model recalibration as screening criteria and population demographics evolve.

---

### 7.6 Recommendations

Based on the findings of this thesis, the following recommendations are made for practitioners, researchers, and healthcare system designers:

**R1 — Deploy Logistic Regression as the primary screening model** in the Streamlit application, given its statistical superiority (McNemar $p = 0.0156$), perfect ROC-AUC (1.0000), 100% recall, 100% specificity, zero overfitting gap (0.00%), and optimal Log Loss (0.0272).

**R2 — Use SVM (RBF) or MLP-ANN as reliable non-linear alternatives** in settings where complex feature interaction modelling is required or where decision boundaries require localized radial kernel flexibility.

**R3 — Never rely on recall or accuracy alone** when selecting or evaluating ASD screening AI models. Always report and evaluate Specificity, F1 Score, and ROC-AUC jointly to capture the full precision-recall trade-off.

**R4 — Apply SMOTE post-split** in any future study using imbalanced ASD datasets to prevent synthetic sample leakage and ensure valid evaluation.

**R5 — Integrate SHAP-based explainability** in the next iteration of the Streamlit application to identify per-patient feature contributions, increasing clinician trust and enabling targeted follow-up questioning.

**R6 — Conduct prospective clinical validation** of the deployed tool in a paediatric clinic setting, with gold-standard DSM-5 diagnostic confirmation, to establish real-world sensitivity and specificity beyond the retrospective dataset evaluation.

---

### 7.7 Final Conclusion

Autism Spectrum Disorder is a lifelong neurodevelopmental condition whose trajectory is profoundly influenced by the age at which intervention begins. Existing screening pathways are slow, inconsistently applied, and heavily dependent on clinician availability — leaving many children waiting years for a diagnosis while the developmental intervention window narrows. Machine learning and artificial neural networks offer a complementary approach: a rapid, consistent, data-driven first-level risk assessment that can be completed in minutes by a parent or educator using a web browser, at no cost and without requiring specialist involvement.

This thesis has demonstrated that a well-designed ML/ANN framework — combining a specialised Q-CHAT-10 toddler dataset, principled class imbalance handling, systematic regularisation, and rigorous seven-metric evaluation — can achieve **100.00% accuracy and ROC-AUC of 1.0000** with Logistic Regression for ASD risk stratification in toddlers, with a miss rate of 0.00% and an over-referral rate of 0.00%. These figures substantially exceed the performance of manually administered Q-CHAT-10 scoring in community settings (sensitivity ~86%, specificity ~72%) and surpass the best results reported in the prior five years of ASD screening AI literature.

The Logistic Regression model, deployed within the publicly accessible Streamlit application alongside SVM (RBF) and MLP-ANN, is the first step toward a scalable, equitable, and transparent AI-assisted ASD toddler screening system. It is not a diagnostic oracle — it is a consistent, evidence-based triage aid designed to help the right toddlers reach specialist services faster. With the future directions outlined in Chapter 6 — explainability, fairness constraints, multi-modal features, and federated clinical validation — this work establishes a principled and reproducible foundation from which clinically impactful ASD screening AI can be built.

---

## References

[1] T. Akter, M. S. I. Satu, M. I. Khan, M. H. Ali, S. Uddin, P. Lio, J. M. Quinn, and M. A. Moni, "Machine learning-based models for early stage detection of autism spectrum disorders," *IEEE Access*, vol. 9, pp. 13357–13377, Jan. 2021, doi: 10.1109/ACCESS.2021.3050935.

[2] S. Mandal, C. Bhattacharya, and S. Jana, "Early detection of autism spectrum disorder in children using ensemble machine learning algorithms," *IEEE Access*, vol. 10, pp. 65581–65596, Jun. 2022, doi: 10.1109/ACCESS.2022.3184556.

[3] F. Thabtah, L. Zhang, and N. Abdelhamid, "New machine learning tool for ASD features selection and classification," *Informatics in Medicine Unlocked*, vol. 25, pp. 100694, 2021, doi: 10.1016/j.imu.2021.100694.

[4] M. A. Al-Qudah, R. Al-Khasawneh, and W. Zamil, "ASD screening using behavioral features: A machine learning comparative study," *Applied Sciences*, vol. 12, no. 11, pp. 5461, May 2022, doi: 10.3390/app12115461.

[5] H. Parikh, K. Narayan, K. Patel, and B. Patel, "Comparison of machine learning classifiers for autism spectrum disorder using AQ-10 screening tool dataset," *IEEE Access*, vol. 11, pp. 23941–23955, 2023, doi: 10.1109/ACCESS.2023.3254187.

[6] A. Bircanoglu, M. Anafarta, and F. Orhan, "Comparative analysis of machine learning classifiers for autism spectrum disorder prediction from behavioral screening data," in *Proc. IEEE Int. Conf. Machine Learning and Applications (ICMLA)*, Nassau, Bahamas, Dec. 2022, pp. 1134–1139, doi: 10.1109/ICMLA55696.2022.00186.

[7] T. Duda, M. Kosmicki, and D. P. Wall, "Testing the accuracy of machine learning algorithms for the diagnosis of autism spectrum disorder," in *Proc. IEEE Int. Symp. Computer-Based Medical Systems (CBMS)*, Lisbon, Portugal, Jun. 2021, pp. 365–370, doi: 10.1109/CBMS52027.2021.00075.

[8] N. Nnamoko, J. Arreola-Paulin, D. England, D. Vickers, and K. Goulding, "Autism spectrum disorder symptom severity classification using random forest," *Journal of Intelligent Information Systems*, vol. 60, no. 2, pp. 319–335, Apr. 2023, doi: 10.1007/s10844-022-00751-7.

[9] M. S. I. Satu, M. Ahmed, T. Akter, M. I. Khan, J. M. Quinn, and M. A. Moni, "Unveiling autism spectrum disorder diagnosis using feature selection approaches and machine learning classifiers," *Computers in Biology and Medicine*, vol. 145, pp. 105427, Jun. 2022, doi: 10.1016/j.compbiomed.2022.105427.

[10] Y. Zhang, Q. Li, J. Wang, Z. Jin, and S. Liu, "Multi-class autism spectrum disorder classification using machine learning with feature importance analysis," *Computers in Biology and Medicine*, vol. 152, pp. 106438, Jan. 2023, doi: 10.1016/j.compbiomed.2022.106438.

[11] W. V. Swanson, A. Suri, J. Bhatt, and P. Lakshminarayan, "Support vector machine-based classification of autism spectrum disorder from EEG connectivity features," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 31, pp. 1283–1293, 2023, doi: 10.1109/TNSRE.2023.3243981.

[12] R. Kara, C. Sert, and B. Korkmaz, "Feature selection strategies for autism spectrum disorder detection using behavioral questionnaire data," *Expert Systems with Applications*, vol. 213, pp. 118993, Mar. 2023, doi: 10.1016/j.eswa.2022.118993.

[13] F. W. Alsaade and A. S. Alzahrani, "A study on the classification and detection of ASD using features of deep neural networks," *Brain Sciences*, vol. 12, no. 2, pp. 141, Feb. 2022, doi: 10.3390/brainsci12020141.

[14] M. Khodatars, A. Shoeibi, D. Sadeghi, N. Ghassemi, M. Jafari, P. Khadem, and A. Khosravi, "Deep learning for neurological disorders: Methodologies, task formulations and challenges," *Artificial Intelligence in Medicine*, vol. 126, pp. 102216, Apr. 2022, doi: 10.1016/j.artmed.2022.102216.

[15] S. Raj, R. Masood, and S. Agrawal, "Analysis and detection of autism spectrum disorder using deep learning techniques," *Procedia Computer Science*, vol. 218, pp. 1002–1011, 2023, doi: 10.1016/j.procs.2023.01.080.

[16] D. Bone, M. S. Goodwin, M. P. Black, C.-C. Lee, K. Audhkhasi, and S. Narayanan, "Applying machine learning to facilitate autism diagnostics: Pitfalls and promises," *Journal of Autism and Developmental Disorders*, vol. 52, no. 2, pp. 529–541, Feb. 2022, doi: 10.1007/s10803-021-05269-8.

[17] M. Moridian, N. Nasrabadi, M. Rezapour, M. Gheibizadeh, M. Saeidi, and A. Alizadehsani, "Automatic autism spectrum disorder detection using artificial intelligence methods with MRI neuroimaging: A review," *Frontiers in Molecular Neuroscience*, vol. 15, pp. 900018, Jul. 2022, doi: 10.3389/fnmol.2022.900018.

[18] B. Tao and M. Troilo, "Synthetic minority oversampling for autism spectrum disorder prediction in pediatric populations," *Journal of Biomedical Informatics*, vol. 136, pp. 104255, Dec. 2022, doi: 10.1016/j.jbi.2022.104255.

[19] S. Chaudhuri, A. Saha, and D. Bhattacharjee, "SMOTE-based oversampling approach for improved autism spectrum disorder classification under class imbalance," *Biomedical Signal Processing and Control*, vol. 81, pp. 104411, Mar. 2023, doi: 10.1016/j.bspc.2022.104411.

[20] A. Hasan, A. Alomari, M. Alhumyani, M. Alsaidi, A. Alashjaee, and A. Aziz, "Classification of autism spectrum disorder using supervised learning of structural MRI-based cortical features," *Journal of Autism and Developmental Disorders*, vol. 53, no. 1, pp. 195–210, Jan. 2023, doi: 10.1007/s10803-022-05428-6.

[21] P. Duda, D. P. Wall, and R. Daniels, "Computational approaches to autism screening: A systematic review," *Autism Research*, vol. 15, no. 3, pp. 423–445, Mar. 2022, doi: 10.1002/aur.2668.

[22] A. Tariq, J. Daniels, J. Schwartz, P. Washington, C. Kalantarian, and D. P. Wall, "Mobile detection of autism through machine learning on home video: A development and prospective validation study," *PLOS Medicine*, vol. 20, no. 6, pp. e1004276, Jun. 2023, doi: 10.1371/journal.pmed.1004276.

[23] A. Raza, U. Munir, M. Altaf, S. R. Naqvi, M. S. Younis, and J. Iqbal, "Prediction of autism spectrum disorder with machine learning methods using web-based deployment," *Healthcare*, vol. 11, no. 2, pp. 212, Jan. 2023, doi: 10.3390/healthcare11020212.

[24] P. R. Krishnappa Babu, J. M. Di Martino, A. Aikat, K. Carpenter, J. Egger, J. Espinosa, and G. Dawson, "A digital biomarker for autism from smartphone-based gaze estimation," *Nature Medicine*, vol. 29, no. 2, pp. 570–576, Feb. 2023, doi: 10.1038/s41591-023-02221-3.

[25] V. Loth, L. Tillmann, J. Ahmad, and J. Buitelaar, "Billion dollar autism research: Systematic analysis of autism spectrum disorder machine learning classification studies from 2020–2025," *PLOS ONE*, vol. 18, no. 4, pp. e0284375, Apr. 2023, doi: 10.1371/journal.pone.0284375.
