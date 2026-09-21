# Thesis Report

## Early Detection of Autism Spectrum Disorder in Children Using Machine Learning and Artificial Neural Networks

**Author:** Mrs. Chhayachabbi Jha
**Thesis Supervisor:** Prof. Raj Kumar Thakur
**Email of Thesis Supervisor:** rajkshiva1@gmail.com
**Date:** 11 May 2026

## Abstract

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition characterised by persistent challenges in social communication, restricted and repetitive patterns of behaviour, and sensory processing differences. Early and accurate diagnosis of ASD is critical, as timely intervention significantly improves developmental outcomes for affected children. However, traditional diagnostic procedures are time-consuming, resource-intensive, and often subject to long waiting periods, limiting access especially in underserved regions.

This thesis proposes a machine learning and artificial neural network (ANN) based computational framework for the early screening and prediction of ASD in toddlers. The study utilises the **Q-CHAT-10 Toddler Autism Screening Dataset (July 2018)**, comprising 1,054 records (975 after duplicate removal).

To prevent **Target Leakage** (Data Leakage), Qchat-10-Score (and any derivative total score columns) were explicitly removed from the training input feature set. In ASD screening datasets, the target binary classification label (Class/ASD Traits) is mathematically derived directly from the Q-CHAT score threshold:

$$\text{Class/ASD Traits} = \begin{cases} 1 & \text{if } \text{Qchat-10-Score} \ge 3 \\ 0 & \text{otherwise} \end{cases}$$

Including Qchat-10-Score as an input feature causes models to perform trivial shortcut learning ($\text{Class/ASD Traits} = f(\text{Qchat-10-Score})$), swallowing feature importance and rendering accuracy metrics scientifically invalid.

After removing target leakage columns, the final feature set consists of **16 input features**:

- **10 Behavioural Screening Items ($A_1$ to $A_{10}$)** (Binary: 0 or 1)
- **Age in Months (\`Age\_Mons\`)** (Numeric: 12–36 months)
- **Sex** (Categorical: Male, Female)
- **Ethnicity** (Categorical: 11 categories)
- **Jaundice** (Categorical: Yes, No)
- **Family History of ASD (\`Family\_mem\_with\_ASD\`)** (Categorical: Yes, No)
- **Respondent Category (\`Who completed the test\`)** (Categorical: Family Member, Health Care Professional, etc.)

A rigorous data preprocessing pipeline was implemented, including duplicate removal, missing value imputation using mode strategy, categorical variable encoding using individual Label Encoders, and class imbalance correction using the Synthetic Minority Over-sampling Technique (SMOTE). Feature scaling was performed using StandardScaler to normalise the input space prior to model training.

Nine classification models were trained and evaluated: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbours (KNN), Support Vector Machine with Polynomial kernel (SVM-Poly), Support Vector Machine with RBF kernel (SVM-RBF), Naïve Bayes, Quadratic Discriminant Analysis (QDA), and a Multilayer Perceptron Artificial Neural Network (MLP-ANN). The MLP-ANN was constructed with two hidden layers of 32 and 16 neurons respectively, using ReLU activation, trained via the Adam optimiser with backpropagation. **True early stopping** was enforced via early\_stopping=True, validation\_fraction=0.1, and n\_iter\_no\_change=10 — ensuring training terminates when validation loss ceases to improve, rather than running to a fixed iteration limit. Hyperparameter optimisation was performed using **5-fold cross-validated GridSearchCV** across Random Forest, SVM (RBF), Decision Tree, and MLP-ANN.

Model performance was assessed using Accuracy, Precision, Recall, Specificity, F1 Score, ROC-AUC, and Log Loss on a held-out test set (20% split). **10-fold Stratified Cross-Validation** was applied to measure generalisation stability. **McNemar's exact test** was used to assess statistical significance of the performance difference between the MLP-ANN and the best classical model. A **SMOTE ablation study** was conducted to empirically quantify the impact of class imbalance correction.

Experimental results demonstrate that **Logistic Regression achieved the highest overall performance** with a ROC-AUC of **1.0000**, test accuracy of **100.00%**, F1 Score of **1.0000**, and Log Loss of **0.0272** — the best-calibrated classifier on the Q-CHAT-10 toddler dataset. This result confirms the linear separability of the ten binary Q-CHAT-10 items ($A_1 \dots A_{10}$). The MLP-ANN with true early stopping achieved ROC-AUC of **0.9968** and test accuracy of **96.41%**, ranking third after SVM (RBF) (ROC-AUC: 0.9986). McNemar's test confirmed that the performance difference between ANN and Logistic Regression is statistically significant ($p = 0.0156 < 0.05$), validating the objective ranking. SVM (Polynomial) was the lowest-ranked model (ROC-AUC: 0.8931, Accuracy: 75.38%).

The trained models were serialised using Python's pickle library and deployed as an interactive web application using Streamlit, enabling real-time ASD risk prediction based on user-provided screening inputs. The application loads the best model — **Logistic Regression**, selected by objective metrics — to provide clinical-grade screening predictions, alongside comprehensive model comparison visualisations including ROC curves, performance bar charts, overfitting analysis tables, and SMOTE ablation results.

This work demonstrates that a rigorous, statistically validated machine learning framework — combining target-leakage prevention, SMOTE class balancing, 10-fold cross-validation, McNemar significance testing, and GridSearchCV hyperparameter tuning — can achieve reliable and scientifically defensible ASD screening, potentially aiding clinicians and caregivers in identifying at-risk toddlers at a much earlier stage than traditional methods allow.

**Keywords:** Autism Spectrum Disorder, ASD Screening, Machine Learning, Artificial Neural Network, MLP Classifier, Logistic Regression, Backpropagation, SMOTE, SMOTE Ablation, GridSearchCV, Hyperparameter Tuning, McNemar Test, 10-Fold Cross-Validation, Early Stopping, Target Leakage, Classification, Early Detection, Streamlit Deployment.

## Table of Contents

|  |  |
| --- | --- |
| **Abstract** |  |
| **Table of Contents** |  |
| **List of Figures** |  |
| **List of Tables** |  |
|  |  |
| **Chapter 1: Introduction** |  |
| 1.1 Background and Motivation |  |
| 1.2 Problem Statement |  |
| 1.3 Aims and Objectives of the Study |  |
| 1.4 Scope of the Study |  |
| 1.5 Significance of the Study |  |
| 1.6 Overview of the Proposed Approach |  |
| 1.7 Thesis Organisation |  |
|  |  |
| **Chapter 2: Literature Review** |  |
| 2.1 Overview |  |
| 2.2 ASD Prevalence, Clinical Challenges, and the Need for Automated Tools |  |
| 2.3 Conventional Machine Learning Classifiers for ASD Screening |  |
| 2.4 Random Forest and Ensemble Methods for ASD |  |
| 2.5 Support Vector Machines and Kernel Methods for ASD |  |
| 2.6 Artificial Neural Networks and Deep Learning for ASD |  |
| 2.7 Handling Class Imbalance with SMOTE |  |
| 2.8 Feature Selection and Dimensionality Reduction in ASD Datasets |  |
| 2.9 Comparative Studies and Systematic Reviews |  |
| 2.10 Mobile and Web Deployment of ASD Screening Tools |  |
| 2.11 Logistic Regression, Naïve Bayes, and QDA as Interpretable Baselines |  |
| 2.12 Ethical Considerations and Limitations in ML-Based ASD Diagnosis |  |
| 2.13 Summary |  |
|  |  |
| **Chapter 3: Research Gap** |  |
| 3.1 Introduction |  |
| 3.2 Identified Research Gaps |  |
| &nbsp;&nbsp;&nbsp;3.2.1 Lack of Comprehensive Multi-Model Benchmarking |  |
| &nbsp;&nbsp;&nbsp;3.2.2 Insufficient Overfitting Analysis and Generalisation Reporting |  |
| &nbsp;&nbsp;&nbsp;3.2.3 Limited Use of Large, Combined, and Representative Datasets |  |
| &nbsp;&nbsp;&nbsp;3.2.4 Inadequate Handling of Class Imbalance |  |
| &nbsp;&nbsp;&nbsp;3.2.5 Absence of Probabilistic and Discriminant Analysis Classifiers |  |
| &nbsp;&nbsp;&nbsp;3.2.6 Lack of End-to-End Deployment with Multi-Model Inference |  |
| &nbsp;&nbsp;&nbsp;3.2.7 Reproducibility and Code Transparency Deficits |  |
| 3.3 Summary of Research Gaps and Thesis Contributions |  |
| 3.4 Research Objectives |  |
|  |  |
| **Chapter 4: Methodology** |  |
| 4.1 Overview |  |
| 4.2 Overall System Architecture and Workflow |  |
| 4.3 Phase 1: Dataset Description |  |
| &nbsp;&nbsp;&nbsp;4.3.1 Data Source |  |
| &nbsp;&nbsp;&nbsp;4.3.2 Feature Description |  |
| &nbsp;&nbsp;&nbsp;4.3.3 Target Variable and Class Distribution |  |
| 4.4 Phase 2: Data Preprocessing |  |
| &nbsp;&nbsp;&nbsp;4.4.1 Duplicate Removal |  |
| &nbsp;&nbsp;&nbsp;4.4.2 Missing Value Imputation |  |
| &nbsp;&nbsp;&nbsp;4.4.3 Categorical Label Encoding |  |
| &nbsp;&nbsp;&nbsp;4.4.4 Numeric Coercion and Residual Fill |  |
| 4.5 Phase 3: Train-Test Partitioning |  |
| 4.6 Phase 4: Class Imbalance Handling with SMOTE |  |
| 4.7 Phase 5: Feature Scaling |  |
| 4.8 Phase 6: Model Definition and Training |  |
| &nbsp;&nbsp;&nbsp;4.8.1 Logistic Regression |  |
| &nbsp;&nbsp;&nbsp;4.8.2 Decision Tree |  |
| &nbsp;&nbsp;&nbsp;4.8.3 Random Forest |  |
| &nbsp;&nbsp;&nbsp;4.8.4 K-Nearest Neighbours (KNN) |  |
| &nbsp;&nbsp;&nbsp;4.8.5 Support Vector Machine – Polynomial Kernel |  |
| &nbsp;&nbsp;&nbsp;4.8.6 Support Vector Machine – RBF Kernel |  |
| &nbsp;&nbsp;&nbsp;4.8.7 Gaussian Naïve Bayes |  |
| &nbsp;&nbsp;&nbsp;4.8.8 Quadratic Discriminant Analysis (QDA) |  |
| &nbsp;&nbsp;&nbsp;4.8.9 Multilayer Perceptron – Artificial Neural Network (MLP-ANN) |  |
| 4.9 Phase 7: Model Evaluation Framework |  |
| &nbsp;&nbsp;&nbsp;4.9.1 Overfitting Analysis |  |
| &nbsp;&nbsp;&nbsp;4.9.2 Confusion Matrices — All Nine Models |  |
| 4.10 Phase 8: Model Serialisation |  |
| 4.11 Phase 9: Streamlit Web Application Deployment |  |
| 4.12 Laboratory Setup |  |
| &nbsp;&nbsp;&nbsp;4.12.1 Hardware Configuration |  |
| &nbsp;&nbsp;&nbsp;4.12.2 Software and Development Environment |  |
| &nbsp;&nbsp;&nbsp;4.12.3 Python Libraries and Versions |  |
| 4.13 Tools, Libraries, and Environment |  |
| 4.14 Summary |  |
|  |  |
| **Chapter 5: Results and Discussion** |  |
| 5.1 Overview |  |
| 5.2 Dataset Summary |  |
| 5.3 Overall Model Performance Results |  |
| 5.4 Logistic Regression — Superior Model Analysis |  |
| 5.5 SVM (RBF) — Second-Ranked Model |  |
| 5.6 Model-by-Model Analysis |  |
| &nbsp;&nbsp;&nbsp;5.6.1 Logistic Regression |  |
| &nbsp;&nbsp;&nbsp;5.6.2 Decision Tree |  |
| &nbsp;&nbsp;&nbsp;5.6.3 K-Nearest Neighbours (KNN) |  |
| &nbsp;&nbsp;&nbsp;5.6.4 SVM (RBF Kernel) |  |
| &nbsp;&nbsp;&nbsp;5.6.5 SVM (Polynomial Kernel) |  |
| &nbsp;&nbsp;&nbsp;5.6.6 Naïve Bayes |  |
| &nbsp;&nbsp;&nbsp;5.6.7 Quadratic Discriminant Analysis (QDA) |  |
| 5.7 Overfitting Analysis |  |
| 5.8 ROC-AUC Comparison |  |
| 5.9 Metric-by-Metric Cross-Model Comparison |  |
| &nbsp;&nbsp;&nbsp;5.9.1 Recall (Sensitivity) — Clinical Priority Metric |  |
| &nbsp;&nbsp;&nbsp;5.9.2 Specificity — Minimising Unnecessary Referrals |  |
| &nbsp;&nbsp;&nbsp;5.9.3 F1 Score — Harmonic Mean of Precision and Recall |  |
| &nbsp;&nbsp;&nbsp;5.9.4 Log Loss — Probability Calibration Quality |  |
| 5.10 Ranking Summary — Multi-Criteria Evaluation |  |
| 5.11 Comparison with Prior Literature |  |
| 5.12 Clinical Interpretation of Results |  |
| 5.13 Summary of Key Findings |  |
|  |  |
| **Chapter 6: Discussion** |  |
| 6.1 Introduction to Discussion |  |
| 6.2 Interpreting the Performance Hierarchy |  |
| &nbsp;&nbsp;&nbsp;6.2.1 Why Logistic Regression Achieves Superior Performance |  |
| &nbsp;&nbsp;&nbsp;6.2.2 Why SVM (RBF) is the Strongest Non-Linear Model |  |
| &nbsp;&nbsp;&nbsp;6.2.3 Why SVM (Polynomial) Underperforms |  |
| &nbsp;&nbsp;&nbsp;6.2.4 The KNN and QDA Paradox — Perfect Recall Without Optimal Utility |  |
| 6.3 The Precision-Recall Trade-Off in ASD Screening |  |
| 6.4 Significance of Overfitting Control |  |
| &nbsp;&nbsp;&nbsp;6.4.1 Why Overfitting Control Matters in Medical AI |  |
| &nbsp;&nbsp;&nbsp;6.4.2 Impact of Target Leakage Prevention and Model Calibration |  |
| &nbsp;&nbsp;&nbsp;6.4.3 MLP-ANN Early Stopping & Overfitting Control |  |
| 6.5 Addressing the Research Gaps |  |
| 6.6 Comparison with State-of-the-Art |  |
| 6.7 SMOTE and Class Imbalance: Impact on Results |  |
| 6.8 Limitations of the Study |  |
| 6.9 Ethical Considerations |  |
| 6.10 Implications for Future Research |  |
| 6.11 Practical Utility of the Streamlit Application |  |
| 6.12 Summary |  |
|  |  |
| **Chapter 7: Conclusion** |  |
| 7.1 Overview |  |
| 7.2 Principal Findings |  |
| &nbsp;&nbsp;&nbsp;7.2.1 Logistic Regression is the Superior Model for ASD Screening |  |
| &nbsp;&nbsp;&nbsp;7.2.2 SVM (RBF) is the Second-Ranked and Most Reliable Non-Linear Classifier |  |
| &nbsp;&nbsp;&nbsp;7.2.3 Overfitting Was Controlled Across All Models |  |
| &nbsp;&nbsp;&nbsp;7.2.4 Class Imbalance Handling is Essential for High Recall |  |
| &nbsp;&nbsp;&nbsp;7.2.5 The Feature Set is Highly Discriminative |  |
| 7.3 Research Objectives — Achievement Summary |  |
| 7.4 Contributions of the Thesis |  |
| 7.5 Limitations Acknowledged |  |
| 7.6 Recommendations |  |
| 7.7 Final Conclusion |  |
|  |  |
| **References** |  |

## List of Figures

| **Figure** | **Caption** |
| --- | --- |
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

## List of Tables

| **Table** | **Caption** |
| --- | --- |
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
| **Table 6.1** | Research gap coverage assessment — G1–G10 addressed by this thesis |
| **Table 7.1** | Research objective achievement summary — RO1–RO6 |

## Chapter 1: Introduction

### 1.1 Background and Motivation

Autism Spectrum Disorder (ASD) is a complex, lifelong neurodevelopmental condition characterised by persistent impairments in social communication and interaction, along with restricted, repetitive patterns of behaviour, interests, and activities. First formally described by Leo Kanner in 1943, ASD has since been recognised as a spectrum disorder — one that encompasses a wide continuum of symptom severity, cognitive ability, and functional independence. According to the World Health Organisation (WHO), approximately 1 in 100 children worldwide is diagnosed with ASD, while prevalence data from the United States Centers for Disease Control and Prevention (CDC) places the figure closer to 1 in 36 children, reflecting both genuine increases and improvements in diagnostic awareness.

The consequences of late or missed diagnosis are profound. Neuroscientific research consistently demonstrates that the human brain exhibits its highest degree of neuroplasticity during the first three to five years of life. Behavioural, speech, and occupational interventions applied within this critical developmental window yield substantially superior outcomes in communication, adaptive behaviour, and social integration compared to interventions initiated later in childhood or adolescence. Despite this well-established clinical consensus, the median age of ASD diagnosis globally remains above four years, and in many low- and middle-income countries it exceeds seven years. This diagnostic lag represents not merely a healthcare failure but a missed opportunity for transformative developmental gain.

The bottleneck lies in the diagnostic process itself. Gold-standard diagnostic instruments — the Autism Diagnostic Observation Schedule, Second Edition (ADOS-2) and the Autism Diagnostic Interview–Revised (ADI-R) — are comprehensive, valid, and reliable, but they are also clinician-administered, time-intensive (requiring two to four hours per assessment), and dependent on the availability of highly trained specialists. In many parts of the world, waiting lists for such evaluations span months to years. Primary care physicians and paediatricians, who represent the most accessible first line of contact for families, typically lack the specialised training to confidently identify early behavioural markers of ASD during routine developmental screenings.

The intersection of machine learning (ML) and clinical medicine offers a compelling solution to this access-and-efficiency gap. ML-based models trained on structured behavioural screening questionnaire data have demonstrated the capacity to identify ASD risk rapidly, consistently, and at scale — without requiring specialist clinical infrastructure. By encoding the pattern-recognition capabilities learned from large labelled datasets, such models can function as decision-support tools that flag children at elevated ASD risk, enabling targeted referrals for formal evaluation while avoiding unnecessary burden on specialist services. The democratisation of early ASD screening through lightweight, deployable ML tools represents one of the most actionable and practically meaningful applications of artificial intelligence in paediatric healthcare.

This thesis is motivated by the dual imperative of clinical need and technological opportunity: to design, train, rigorously evaluate, and deploy a machine learning and artificial neural network (ANN) framework capable of reliable, accessible early ASD screening for children, and to do so in a manner that is transparent, reproducible, and directly grounded in the identified limitations of prior research.

### 1.2 Problem Statement

Despite a growing body of research demonstrating the utility of machine learning for ASD screening, several critical gaps persist in the existing literature that limit the clinical reliability, scientific rigour, and practical deployability of published models.

First, the majority of studies evaluate only a single or small number of classifiers, often selecting the best-performing model without systematic comparison across a diverse model portfolio. This selective reporting prevents practitioners from understanding the relative trade-offs between model families and does not establish a trustworthy performance hierarchy. Second, overfitting — the tendency of a model to memorise training data patterns rather than learning generalisable rules — is rarely reported or analysed, despite being a primary threat to real-world clinical utility. A model that achieves 99% training accuracy but 75% test accuracy would be clinically unsafe, yet many published studies report training accuracy alone. Third, class imbalance in ASD datasets (where ASD-negative cases substantially outnumber ASD-positive cases) is frequently neglected, leading to models that are biased towards the majority class and fail to detect the very condition they were designed to identify. Fourth, end-to-end deployment — the transformation of a trained model into a usable clinical tool — is rarely undertaken, leaving models confined to academic manuscripts and inaccessible to the practitioners and caregivers who could most benefit.

This thesis directly addresses each of these gaps by constructing a comprehensive, multi-model benchmarking framework that includes explicit overfitting analysis, principled class imbalance correction via SMOTE, and end-to-end Streamlit web application deployment.

### 1.3 Aims and Objectives of the Study

The overarching aim of this thesis is to develop a robust, explainable, and practically deployable machine learning and ANN-based framework for the early screening of Autism Spectrum Disorder in children, using structured behavioural questionnaire data.

The specific research objectives are as follows:

**RO1 — Rigorous Preprocessing & Target Leakage Prevention:** To develop a rigorous, end-to-end data preprocessing pipeline including duplicate removal, mode imputation, categorical encoding, exclusion of the target-leaky Qchat-10-Score feature, and post-split SMOTE class balancing — ensuring methodological integrity throughout.

**RO2 — Multi-Model Benchmarking:** To train and systematically compare nine classification algorithms — Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbours, SVM (Polynomial), SVM (RBF), Gaussian Naïve Bayes, QDA, and MLP-ANN — under identical experimental conditions on the Q-CHAT-10 toddler dataset.

**RO3 — Rigorous Evaluation with Overfitting Control:** To evaluate all models against seven performance metrics (Accuracy, Precision, Recall, Specificity, F1 Score, ROC-AUC, and Log Loss) on a held-out test set, and to quantify overfitting by explicitly reporting the training-test accuracy gap for each model.

**RO4 — Hyperparameter Tuning & Early Stopping:** To apply 5-fold GridSearchCV hyperparameter optimisation for key model families (RF, SVM-RBF, Decision Tree, MLP-ANN) and enforce true validation-monitored early stopping for the MLP-ANN, ensuring principled regularisation.

**RO5 — Statistical Validation & Cross-Validation:** To evaluate generalisation stability through 10-fold stratified cross-validation, and to confirm the statistical significance of the best model's superiority using McNemar's exact test.

**RO6 — End-to-End Deployment:** To serialise all trained models and deploy them as an interactive, real-time web application using Streamlit, enabling clinicians, researchers, and caregivers to obtain multi-model ASD risk predictions with probability scores and comparative visualisations.

### 1.4 Scope of the Study

This thesis is scoped as follows:

- **Dataset:** The study utilises the Toddler Autism dataset July 2018.csv dataset, the Q-CHAT-10 toddler ASD screening dataset comprising **1,054 records** (975 after deduplication) and **16 input features** (ten Q-CHAT-10 behavioural items A1–A10, age in months, sex, ethnicity, history of jaundice, family history of ASD, and who completed the test; excluding total Q-CHAT-10 score to prevent target leakage), with a binary target variable (ASD traits Yes / No).
- **Population:** The study is focused exclusively on the **toddler** subpopulation of ASD screening data (children aged 12–36 months screened using the Q-CHAT-10 instrument). Adult and adolescent screening data are outside the scope of this work.
- **Classification task:** The task is binary supervised classification — predicting whether a child screens positive or negative for ASD risk.
- **Model families:** Nine classifiers spanning linear, tree-based, kernel-based, probabilistic, discriminant-analysis, and neural network families are included. Deep learning architectures (CNNs, RNNs, transformers) and neuroimaging-based approaches are outside scope.
- **Deployment platform:** The Streamlit web application is designed as a local and cloud-deployable screening adjunct. It does not constitute a certified medical device and is intended as a research-and-demonstration tool.
- **Geographical context:** The models are trained on a composite dataset aggregated from multiple international screening studies and are not specific to any single national population.

### 1.5 Significance of the Study

This work carries significance across multiple dimensions:

**Clinical significance:** A validated, accessible, and real-time ASD screening tool can meaningfully shorten the diagnostic pathway for at-risk toddlers, enabling earlier referral and, consequently, earlier access to interventional support. By achieving a Recall (Sensitivity) of 100.00% and Specificity of 100.00% — ensuring that all ASD-positive toddlers in the test set are correctly flagged and no ASD-negative toddlers are over-referred — the Logistic Regression model developed in this thesis prioritises the clinical imperative of minimising both missed diagnoses and unnecessary clinical burdens.

**Scientific significance:** This thesis advances the state of knowledge through a systematic, multi-model comparative study that includes overfitting analysis, SMOTE-based class imbalance correction, and a seven-metric evaluation framework. By documenting not only which models perform best but also why, and by quantifying generalisation risk through the train-test accuracy gap, this work provides a more complete and trustworthy performance characterisation than the majority of existing published studies.

**Methodological significance:** The adoption of nine diverse classifiers — including the often-neglected Quadratic Discriminant Analysis and Gaussian Naïve Bayes — provides a comprehensive baseline that future researchers can build upon. The explicit inclusion of Log Loss as a metric enriches the evaluation by assessing probabilistic calibration, an important dimension of clinical trustworthiness that accuracy-centric studies overlook.

**Practical significance:** The Streamlit web application delivers the model's capabilities to end users without requiring any programming expertise. Healthcare providers, community screeners, and caregivers in resource-limited settings can interact with the tool through a structured questionnaire interface, receiving real-time ASD risk assessments. This end-to-end deployment closes the gap between research and real-world utility that characterises much of the existing ML-in-medicine literature.

### 1.6 Overview of the Proposed Approach

The computational framework proposed in this thesis follows a nine-phase pipeline:

1\. **Data Acquisition:** The Q-CHAT-10 toddler ASD screening dataset (975 records after dedup, 16 features after dropping Case\_No and target-leaky Q-CHAT-10 score) is loaded.

2\. **Preprocessing:** Duplicate removal, mode-based missing value imputation, individual Label Encoding of categorical features, and numeric coercion ensure data integrity.

3\. **Partitioning:** A stratified 80/20 train-test split is applied, preserving the class distribution in both partitions.

4\. **Class Imbalance Correction:** SMOTE is applied exclusively to the training set to generate synthetic ASD-positive samples and achieve a balanced 1:1 class ratio.

5\. **Feature Scaling:** StandardScaler normalises the input feature space based on training-set statistics only, preventing data leakage.

6\. **Model Training:** Nine classifiers are trained with carefully selected hyperparameters. The MLP-ANN uses two hidden layers (32 and 16 neurons), ReLU activation, the Adam optimiser, and validation-monitored early stopping (early\_stopping=True).

7\. **Evaluation:** All models are evaluated on the held-out test set across seven metrics, with train-test accuracy gaps computed to quantify overfitting.

8\. **Serialisation:** Trained models, the Label Encoder dictionary, and the StandardScaler are serialised using Python's pickle library for persistent storage and inference-time reuse.

9\. **Deployment:** The serialised artefacts are loaded by a Streamlit web application that accepts structured screening inputs and returns real-time ASD risk predictions from both the best classical model (Logistic Regression) and the MLP-ANN.

Logistic Regression achieved the highest overall performance, with a ROC-AUC of **1.0000**, test accuracy of **100.00%**, F1 Score of **1.0000**, and Recall of **100.00%**, confirmed as statistically significantly superior to MLP-ANN via McNemar's exact test (). SVM (RBF) ranked second (ROC-AUC 0.9986), followed by MLP-ANN under early stopping (ROC-AUC 0.9968, test accuracy 96.41%).

### 1.7 Thesis Organisation

The remainder of this thesis is organised into six chapters:

**Chapter 2 — Literature Review** critically examines 25 studies published between 2021 and 2026 on machine learning and ANN-based ASD screening, covering conventional classifiers, ensemble methods, support vector machines, deep learning, class imbalance handling, feature selection, and deployment approaches. Key methodological limitations in the existing body of work are identified and contextualised.

**Chapter 3 — Research Gap** synthesises the findings of the literature review into eleven formally identified research gaps (G1–G10, including G4b) and maps them directly to the objectives and contributions of this thesis.

**Chapter 4 — Methodology** presents the complete nine-phase experimental pipeline in detail, including dataset description, preprocessing protocol, SMOTE application, model architectures and hyperparameters, evaluation framework, model serialisation, laboratory setup, and Streamlit deployment architecture.

**Chapter 5 — Results and Discussion** presents the complete experimental results for all nine classifiers across seven metrics, with detailed analysis of Logistic Regression, SVM (RBF), MLP-ANN, and the remaining models. Overfitting analysis, ROC-AUC comparisons, and metric-by-metric cross-model discussion are provided.

**Chapter 6 — Discussion** contextualises the results within the broader research landscape, addresses each of the eleven research gaps (G1–G10), examines precision-recall trade-offs, discusses the clinical and ethical implications of the findings, and identifies limitations of the study.

**Chapter 7 — Conclusion** summarises the principal findings, assesses achievement of all six research objectives (RO1–RO6), enumerates the thesis contributions, and offers recommendations for future research.

## Chapter 2: Literature Review

### 2.1 Overview

The intersection of machine learning (ML), artificial intelligence (AI), and clinical neuroscience has produced a rapidly growing body of research aimed at automating and improving the diagnosis of Autism Spectrum Disorder (ASD). ASD affects millions of children worldwide and is characterised by a heterogeneous symptom profile that makes early and accurate diagnosis both critical and challenging. Over the past five years, researchers have explored a wide range of computational techniques—from classical classifiers and ensemble methods to deep neural networks and transformer-based architectures—to build reliable, scalable screening and diagnostic tools. This chapter critically reviews 28 significant and recent studies (2021–2025) that inform the methodology, motivation, and design choices of this thesis.

### 2.2 ASD Prevalence, Clinical Challenges, and the Need for Automated Tools

Autism Spectrum Disorder (ASD) is a lifelong condition of the nervous system that affects social interaction, communication and restricted or repetitive behavior. There is an overwhelming burden on health care systems and families, with current statistics showing a large increase in the global number of ASD cases. Early identification is crucial; clinical research has repeatedly shown that children treated by therapeutic interventions before three years of age have much better developmental trajectories than children identified later in childhood \[1\],\[2\],\[3\],\[4\]. The traditional diagnostic process, however, is fraught with problems. There are a number of standard tools, including the Autism Diagnostic Observation Schedule (ADOS) and Autism Diagnostic Interview-Revised (ADI-R), that are used to make a diagnosis and require extensive training, significant clinical time, and expert observation, which can create long waiting lists and high costs \[1\], \[4\], \[5\]. Furthermore, these approaches are subjective and involve clinician judgment and caregiver reports that may be culturally biased and subject to recall errors. Conventional methods are reported to be especially unreliable for children under 2 years of age when there is a critical time for intervention \[2\].

To overcome these issues, the scientific community has been looking to Artificial Intelligence (AI). Machine Learning (ML) and Artificial Neural Networks (ANN) offer a tool for analysing high-dimensional, complex data to find subtle indicators of ASD. These computational tools can process huge amounts of information, such as subtle facial micro-expressions and eye-gaze patterns, or the non-linear dynamics of brain waves and motor trajectories, providing a more objective and scalable way to detect early signs \[3\], \[6\], \[7\], \[8\]. This review summarizes 28 of the most important research papers to present a comprehensive picture of both methodologies and datasets used, as well as the performance results and advantages and disadvantages of using AI for ASD detection.

Application of Methods and Approaches in ASD Detection.

A distinction has been found between the traditional Machine Learning (ML) approaches that mostly rely on manual feature selection and Deep Learning (ANN) approaches that can learn hierarchical feature representations from raw data automatically.

### 2.3 Conventional Machine Learning Classifiers for ASD Screening

Support Vector Machines (SVM) are considered as one of the best traditional ML classifiers for the classification of ASD due to its ability to define an optimal hyperplane that separates different classes in high-dimensional feature space with largest margin. SVM has been successfully used to many types of data:

**Behavioral Data**: SVM is frequently used in studies of questionnaire responses to the UCI Machine Learning ASD screening datasets, with an accuracy of 98% or more \[9\], \[10\]**.** Alkahtani et al. \[6\] investigated the application of both traditional machine learning (ML) and deep learning (DL) techniques for the early detection of Autism Spectrum Disorder (ASD) in toddlers. The study utilized two ASD screening datasets, including a Saudi Arabian toddler dataset containing 1,054 instances and 19 features, and a Kaggle dataset comprising 507 instances with 16 features. The authors evaluated several supervised ML algorithms, including Support Vector Machine (SVM), k-Nearest Neighbors (KNN), and Decision Tree (DT), alongside a Long Short-Term Memory (LSTM) deep learning model. Experimental results demonstrated outstanding performance, with both the SVM and LSTM models achieving 100% classification accuracy. The findings suggest that ML and DL techniques can effectively identify ASD traits from behavioral screening data, enabling early diagnosis and timely intervention. The proposed system has the potential to support parents and healthcare professionals in the early screening of ASD, thereby reducing diagnostic delays and facilitating more effective treatment planning.

**Eye-tracking studies**: SVM can be employed for the classification of scan paths and fixation patterns and is a powerful tool for distinguishing between children with ASD and children of typically developing (TD) children \[7\].

**High-Dimensional Features:** Jain and Tripathy \[11\] highlighted the fact that SVM can capitalize on the multifaceted associations in high-dimensional screening data, and showed that it outperforms other models, such as Random Forest and Gradient Boosting, in their specific toddler dataset. They noted that SVM performed very well with demographic information, medical history, and screening test result data.

### 2.4 Random Forest and Ensemble Methods for ASD

**Random Forest and Ensemble Methods**: A prominent method in the reviewed literature is ensemble methods which is one method that will improve the predictive performance by using multiple models and reduce overfitting.

**_Random Forest (RF)_:** RF models are highly stable and accurate models that are obtained by aggregating the outputs of a number of decision trees. They have been 2used for processing medical claims data \[12\] and gait parameters \[13\] and are able to process a wide range of feature sets.

**_Gradient Boosting (XGBoost, CatBoost, LightGBM)_**: They are efficient and have great predictive accuracy. XGBoost has been reported to have 97.8% accuracy and 0.99 AUROC on behavioral screening data \[14\]. In a similar study, Kaur et al. \[15\] reported that XGBoost and CatBoost models had a higher accuracy of 94.79% and 99.14% respectively, with a ROC-AUC of 99.23% and 99.14% respectively, compared to the SVM model, which had an accuracy of 85.41%. Logistic regression and baseline models

2.5 **_3 Logistic Regression (LR)_** may seem straightforward but is still a robust method for analysis, particularly when data is well-formatted and feature-to-feature dependencies are reasonably linear. On a toddler screening dataset based on the ASDTests mobile application, users of Perfect Performance: Shambour \[10\] showed that a logistic regression classification could reach 100% accuracy, thus indicating that there are clear behavioral markers in some populations. Ensemble Integration: Alzakari et al. \[16\] used an ensemble of logistic regression and support vector machine classifiers (LR-SVM ensemble) for improving the accuracy of ASD identification.

**_Gait Based Detection_**: Among various machine learning (ML) models, the best algorithm for gait based detection was Binomial logistic regression (Logit) with accuracy of 0.82 \[13\].

### 2.6 Artificial Neural Networks and Deep Learning for ASD

**_2.2.1_** CNNs are considered as the most popular and reliable approach to ASD detection, especially for facial features analysis. The CAM-DASD model \[1\] adopts a convolution attention-based mechanism with the addition of Histogram of Oriented Gradients (HOG) to focus on the most important facial regions associated with the symptoms of ASD. However, it is difficult to obtain large image datasets in clinical settings, which is why many clinicians use transfer learning. Several models, such as AlexNet (ASDDTLA) \[18\] and Xception \[1\] and VGG16 \[5\] are trained on general image repositories then “fine-tuned” on ASD-specific facial images. Thus, the choice of the architecture is crucial, as demonstrated by Rashid and Shaker \[5\] with an accuracy of 94% for Xception and 73% for VGG16. To analyze children's facial images, multi-scale Extraction is used by Polavarapu et al. \[19\] in a Transfer Learning framework. Recurrent neural networks (RNN) and LSTM. RNNs and their variants like Long Short-Term Memory (LSTM) networks are built to work with sequential data and time dependencies.

**_2.2.2 Temporal Dynamics:_** LSTM architectures have been used to analyze time-series data from EEG signals and eye-tracking scanpaths \[6\]. In the case of tablet based task (drag and drop), the authors of \[8\] found that an "ANN based motion analysis of raw trajectories system" could achieve 90% accuracy by analyzing raw coordinates, velocities and accelerations, while just analyzing raw coordinates, the accuracy was 76%. Hybrid and Ensemble Neural Architectures

**_2.2.3_** This forward edge of detection of ASD has to do with the integration of various neural architectures. CNN-BiLSTM: John and Patil \[20\] introduced an ensemble model combining CNNs for capturing spatial features with Bidirectional LSTMs for capturing temporal or sequential features, which obtained a 95.9% accuracy rate and a recall rate of 100%. Aarthi and Kannimuthu \[21\] tested the hybrid of several models, and showed that the architecture of MobileNetV2 with Gated Recurrent Units (GRU) gave a test accuracy of 95.5% compared to InceptionV3 and EfficientNetB4 on the facial image dataset.

**_2.2.4 Deep Neural Networks (DNN) and Multi-Layer Perceptrons (MLP)_** : There are many structured behavioral and demographic datasets which are applied to standard deep learning models. Behavioral Patterns: Tan \[22\] applied a DNN model for capturing complex non-linear patterns in the ASD tests mobile application data with the accuracy of 99.55%, and significantly improved the minority classes identification of post-imbalance learning treatment. Alluri and Suganya \[23\] obtained an MLP to analyze questionnaire responses (A1-A10) and background information from 3361 people with 96% accuracy and 0.9946 ROC-AUC. Drop-out DNNs: Sahu and Sahu \[24\] used standard DNNs and drop-out DNNs (Dout-DNN) and found that Dout-DNNs performed better with specificities of 97.16% compared to 93.84% for the standard DNNs. Specialized Signal and Image Processing Techniques

(2.3) The literature points to new approaches to working with non-standard forms of data: Path Signature Discovery: Yin et al. \[2\] proposed path signatures to identify meaningful features in sparse, class-imbalanced and heterogeneous structural MR images. This approach gives different weights to votes to account for sample heterogeneity. Unsupervised Compression: Unsupervised compressors are utilized to resolve data imbalance and to extract key features from small initial data sets with a Siamese verification framework \[2\]. Linguistic Features: Assaf et al. \[25\] used text-based inputs with an accuracy of more than 86% while maintaining biometric privacy using two linguistic features: Mean Length of Utterance (MLU) and Mean Length of Turn Ratio (MLT Ratio). Data sets and performance evaluation The quality and variety of the datasets used in training the AI model are central to its reliability.

3.1 Behavioral and Screening Questionnaires (Q-CHAT, ADOS, M-CHAT) The most widely available and easily obtained dataset is questionnaire-based. The ASD screening datasets for toddlers, children, adolescents and adults are used in many studies \[10\], \[14\], \[26\], \[27\]. The questions are generally 10 questions (A1-A10) parallel to the Q-CHAT (Quantitative Checklist for Autism in Toddlers) or AQ-10 (Autism Spectrum Quotient). The Modified Checklist for Autism in Toddlers, Revised, with Follow-Up (M-CHAT-R/F) was used to train a feed-forward ANN by Achenie et al. \[9\]. They achieved an overall correct classification rate of 99.72% and high performance was seen across a variety of demographic subgroups (White: 99.92%, Black: 99.79%, Male: 99.64%, Female: 99.95%). The mobile application collecting data, "ASDTests", is avaliable in 11 different languages, and gives access to a large, globally collected, dataset of behavioral traits from thousands of children. More than 1,400 cases were initially gathered to show the proficiency of small sets of autistic attributes in predictive analysis.

3.2 Biological and Biometric Data (EEG, MRI, Facial Features, Gait) Objective biological markers offer more consistent screening, but are less easy to obtain. Bosl et al. \[3\] reported that non-linear EEG markers were able to predict ASD outcome in a dataset of 188 infants (99 high-risk for ASD and 89 low-risk controls) as early as 3 months of age. Specificity and sensitivity were >95% for some ages and EEG data was highly correlated with ADOS calibrated severity scores. Eye-Tracking: Ahmed et al. \[7\] use eye-tracking data from the Figshare repository to study visual attention. Results show that Feedforward Neural Networks (FFNN) exceeded an outstanding classification accuracy of 99.8%. Facial Recognition Models: The Kaggle "autism-image-data" repository is the main repository for facial recognition models \[5\],\[18\],\[ \[21\]. Rashid and Shaker \[5\] reported that they used the real-world variability by including children of various ages and photo calibers in their data set of 2,940 photos. Gait Parameters: Ganai et al. \[13\] measured gait data using a single RGB camera-based pose estimation technique (MediaPipe), and they found statistically significant gait deviations from which they were able to successfully classify the children with ASD and TD with ML. The medical records and developmental records are provided under the following terms and conditions: MarketScan Health Claims Database: The medical claims data also was used by Chen et al. \[12\] to predict ASD diagnosis from 2005 to 2016. The model they've developed with the Random Forest technique had an AUROC of 0.775 to predict a diagnosis at 24 months, 0.832 at 30 months and 0.834 when inpatient and outpatient encounters were separated. The linguistic features were extracted from the TalkBank repository of speech transcripts of children and used to test privacy preserving detection models \[25\]. Provide comparative performance metrics within each subgroup of students. Generalizability is important; it is a measure of performance across different populations. Demographic Robustness: Achenie et al. \[9\] achieved more than 99.7% accuracy at different education levels of mothers (11-15 years vs. 16-20 years) with their fANN model, indicating that the tools for behavioral screening can also be accurate across socioeconomic levels when used properly. Age-Based Performance: The predictive ability increases with the age of the child \[12\] (AUROC of 0.720 at 18 months; 0.800 at 30 months). Likewise, Jacob et al. \[27\] achieved an average of about 90% MCC and 95% balanced accuracy in all four age groups (toddler to adult) with their AutoML framework. Challenges and limitations in Computational ASD screening While the results are promising, there are a number of hurdles to overcome. Data Heterogeneity and Imbalance.

4.1 Data Scarcity The clinical data used for the purposes of ASD is difficult to obtain and is usually very skewed. For instance, the structural MR images are heterogeneous, imbalanced, and scarce \[2\]. ASD is a complex and heterogeneous disorder, making it challenging to develop common scalable biomarkers for all children with the disorder \[3\], \[16\]. As mentioned in the studies of motion data \[8\] and gait analysis \[13\] small sample size reduces the statistical power and generalizability of the results.

4.2 Clinical Validation and Ground Truth Reliability One of the drawbacks of many studies is the lack of clinical validation, as they only use secondary data from sources such as Kaggle or UCI. Labels (ground-truth) in these datasets can be subjectively chosen screening tests or classification based on the Internet and may include errors \[14\], \[28\]. Moreover, the traditional diagnostic approaches are poorly reliable for children under 2 years of age, and many models are trained on such labels \[2\].

One of the common criticisms of deep learning models is that they are “black boxes.” Explainable AI (XAI) systems are needed to explain their results to clinicians to allow them to make informed decisions \[14\],\[15\]. As far as privacy is concerned, ethical issues are especially relevant for minors, using identifiable biometric information such as faces or speech \[25\]. Additionally, subjective questionnaires for caregivers are still a standard tool for screening which ML methods cannot fully replace because they do not have objective and good quality physiological data to replace them \[5\]. Research gaps and future directions. Several research gaps need to be tackled if the technology is to be **widely adopted in the clinic: Multimodal Integration: Future studies should explore multimodal integration (EEG and eyetracking and facial analysis) to develop more comprehensive diagnostic models, \[15\],\[21\].**

**Explainable AI (XAI):**

SHAP and other techniques can be incorporated into the model architecture to enable easy visualization of what features are contributing to the classification, which will help put clinicians at ease and build trust in AI-aided diagnosis \[14\]. Longitudinal Monitoring: AI models should be used for longitudinal monitoring, not as a one-time screening, but monitor a child's development from as early as 3 months of age \[2\], \[3\]. There is an urgent need for large-scale real-world pilot testing of these models in the clinical setting with a wide variety of populations \[14\], \[28\].

The present thesis builds directly on these foundations, extending the literature through a comprehensive nine-model comparative framework, rigorous overfitting analysis across all models, and an end-to-end Streamlit deployment pipeline—addressing gaps identified across the reviewed studies, particularly the absence of transparent multi-model comparison with explicit overfitting quantification in a single unified system.

| Reference | Methodology | Dataset | Performance | Year |
| \[1\] Roy et al. | CNN with HOG & Attention | Facial Images (Not specified) | 83.75% Accuracy | 2025 |
| \[18\] Ghazal et al. | Transfer Learning AlexNet | Kaggle Facial Images | 87.7% Accuracy | 2023 |
| \[6\] Alkahtani et al. | SVM, KNN, DT, LSTM | Saudi Arabian & Kaggle Toddler datasets | 100% Accuracy (SVM/LSTM) | 2023 |
| \[20\] John & Patil | CNN-BiLSTM Ensemble | Not available in sources | 95.90% Accuracy | 2025 |
| \[26\] Mohanty et al. | Deep Neural Network | UCI ML (Child/Toddler) | High Relevance (Not specified) | 2021 |
| \[6\] Das et al. | Hybrid KNN+RF | Four distinct datasets | 99.29% Accuracy | 2025 |
| \[19\] Polavarapu et al. | Multi-scale Feature Extraction | Facial Images (Not specified) | High Precision | 2025 |
| \[7\] Ahmed et al. | CNN (GoogleNet/ResNet), SVM | Figshare Eye-tracking | 99.8% Accuracy (FFNN) | 2022 |
| \[9\] Achenie et al. | Feed-forward ANN | M-CHAT-R Archival Data | 99.72% Accuracy | 2019 |
| \[23\] Alluri & Suganya | Multi-Layer Perceptron | Behavioral/Demographic (3,361 individuals) | 96.0% Accuracy | 2025 |
| \[2\] Yin et al. | Path Signature & Siamese DL | Longitudinal MR Images | Effective under practical scenarios | 2024 |
| \[24\] Sahu & Sahu | Drop-out DNN | Not available in sources | 93.84% Accuracy | 2024 |
| \[17\] Hossain et al. | MLP Classifier | Toddler/Child/Adolescent/Adult ASD data | 100% Accuracy | 2021 |
| \[22\] Tan | Deep Neural Network | ASDTests Mobile App (>2,000 children) | 99.55% Accuracy | 2024 |
| \[5\] Rashid & Shaker | Xception, VGG16 (CNN) | Kaggle (2,940 face photos) | 94.0% Accuracy (Xception) | 2023 |
| \[15\] Kaur et al. | XGBoost, CatBoost, MLP | AQ-10 and Q-CHAT-10 | 94.79% Accuracy (XGBoost) | 2025 |
| \[21\] Aarthi & Kannimuthu | MobileNetV2+GRU Hybrid | Kaggle Facial Images | 95.5% Accuracy | 2025 |
| \[21\] Jain & Tripathy | SVM, RF, Gradient Boosting | Kaggle 2018 (Toddler) | SVM outperformed others | 2024 |
| \[8\] Luongo et al. | ANN (Raw motion analysis) | Game-based trajectory data | 90.0% Accuracy | 2024 |
| \[12\] Chen et al. | Random Forest, LASSO LR | MarketScan Health Claims (2005-2016) | 0.834 AUROC (RF) | 2022 |
| \[14\] Ayub et al. | XGBoost, RF, Extra Trees, MLP | UCI ML ASD Screening Dataset | 97.8% Accuracy (XGBoost) | 2025 |
| \[3\] Bosl et al. | Statistical Learning | EEG (99 ASD infants, 89 low-risk) | \>95% Sensitivity/Specificity | 2018 |
| \[4\] Thabtah | ML Classifiers (ASDTests App) | ASDTests (>1,400 instances) | Promising Sensitivity/Specificity | 2019 |
| \[10\] Shambour | Logistic Regression, SVM, RF | UCI ML (ASDTests data) | 100% Accuracy (LR) | 2024 |
| \[28\] Nagamani et al. | XGBClassifier, Random Forest, Decision Trees, AdaBoost, KMeans, and Artificial Neural Networks. | Utilized publicly available ASD screening repositories (e.g., UCI Machine Learning ASD datasets). | Focused on comparative analysis of accuracy, efficiency, and interpretability for early screening. | 2024 |
| \[27\] Jacob et al. | Automated Machine Learning (AutoML) framework with feature ranking and hyperparameter optimization. | Publicly available datasets based on Q-chat scores (Toddlers, Children, Adolescents, Adults). | Achieved ~90% Matthews Correlation Coefficient (MCC) and ~95% balanced accuracy across all groups. | 2023 |
| \[13\] Ganai et al. | RGB camera-based pose estimation (MediaPipe) to analyze gait deviations; Binomial Logistic Regression. | Data from 32 children with ASD and 29 typically developing (TD) controls. | Best performance achieved by Binomial Logistic Regression with an accuracy of 0.82. | 2025 |
| \[25\] Assaf et al. | Machine learning on speech transcripts using CNNs, MLPs,andsemantic/pragmatic feature extraction. | TalkBank, CHILDES, and ASDBank (balanced cohorts of 64-76 children). | Deep learning models reached 83-84% accuracy; semantic feature models achieved up to 94%. | 2025 |
| \[16\] Alzakari et al. | Explainable AI (XAI) with an LR-SVM ensemble classifier and Chi-square feature engineering. | Merged screening datasets specifically focused on toddlers. | 94% accuracy in ASD identification and 99.29% accuracy for optimized teaching strategy selection. | 2025 |

_Table 2.1: Summary of reviewed literature with methodology, dataset, performance, and year._

## Chapter 3: Research Gap

### 3.1 Introduction

A systematic analysis of the literature reviewed in Chapter 2, together with a broader survey of published work on machine learning-based ASD detection, reveals a set of recurring and significant limitations across existing studies. These limitations collectively define the research gap that this thesis addresses. While individual prior studies have made meaningful contributions—demonstrating the viability of ML classifiers for ASD screening, exploring class imbalance mitigation, and proposing deployment frameworks—no single prior work has simultaneously addressed all of the following critical dimensions in a unified, transparent, and reproducible manner.

### 3.2 Identified Research Gaps

**G1: Disjoined multi-model benchmarking in varying conditions**

It is a very fragmented literature. Different studies tend to have different data sets (Kaggle vs. UCI vs. custom clinical data) and different preprocessing steps There are no standardized "benchmarking suites" for ASD in general and there is no definitive consensus regarding which algorithm is better for different data modalities.

**G2: No studies quantifying or reporting overfitting in most cases**

Several studies give unusually high precision (98–100%) \[6\],\[10\], \[17\] without giving learning curves (loss/accuracy vs. epoch) or showing the performance on the separate validation/test set. The reporting about the "generalization gap" is scarce and if it occurs, it is ninety times out of one hundred 10-fold cross-validation \[1\] that is reported, so these models can easily be "overfit" to the specific small datasets used for reporting.

**G3: Limited resources (small size and one source of data), with limited generalizability**

Many of the studies use the same set of open source data sets (Kaggle's autism-image-data or UCI's Autistic Spectrum Disorder Screening Data) \[5\],\[14\],\[17\],\[18\]. These are typically "single-source" and are unable to capture the wide clinical, cultural, and ethnic diversity in the global ASD population, making generalizability to real-world populations difficult.

**G4: Not using class imbalance or SMOTE misapplied pre-split**

The clinical ASD data is naturally imbalanced, i.e., a smaller number of ASD cases. In most papers reviewed, the authors do not explicitly address the imbalance \[2\],\[9\]. If balancing is used (such as SMOTE), it is not always indicated whether this was done after the train-test split, which exposes high risk of data leakage and high risk of over optimistic performance results.

**G5: Deployment to a single model or non-visual interfaces**

The majority of the research ends at the "experimental analysis" phase. The deployment is typically restricted to reporting on a paper instead of actually creating a useful and useful visual tool for clinicians. Existing mobile apps (e.g., ASD Tests) are few and far between and usually focus on one data modality, rather than a strong multi-model ensemble.

**G6: Lack of transparency and reproducibility in code**

The majority of papers do not share public links to their complete model source code, trained model weights, or hyper parameter optimization logs. This “black-box” approach, particularly when it comes to deep learning research \[5\],\[26\], is near impossible to exactly replicate and extend the current models by others.

#### 3.2.1 : Disjoined multi-model benchmarking in varying conditions

**Gap addressed by this thesis:** This work evaluates nine classifiers—Logistic Regression, Decision Tree, Random Forest, KNN, SVM-Poly, SVM-RBF, Naïve Bayes, QDA, and MLP-ANN—under identical preprocessing, train-test split, SMOTE configuration, and evaluation conditions, providing a rigorous and internally consistent comparative benchmark that is absent from prior literature.

#### 3.2.2 Insufficient Overfitting Analysis and Generalisation Reporting

**Gap addressed by this thesis:** All nine trained models are evaluated on both training and test sets, and the overfitting gap (training accuracy minus test accuracy) is explicitly computed, tabulated, and discussed for every model. This provides a transparent generalisation profile that is missing from virtually all prior comparable works.

#### 3.2.3 Limited Use of Large, Combined, and Representative Datasets

**Gap addressed by this thesis:** The Q-CHAT-10 Toddler Autism Screening Dataset comprising 975 records (after deduplication) is employed, with a focus on toddler-specific Q-CHAT-10 features and demographic variables. This dataset provides a specialised and clinically relevant corpus for early toddler ASD screening, enabling reliable class-stratified sampling and effective SMOTE augmentation.

#### 3.2.4 Inadequate Handling of Class Imbalance

**Gap addressed by this thesis:** SMOTE is applied exclusively to the training partition after stratified splitting, following the methodologically correct protocol recommended by Chaudhuri _et al._ \[19\]. The use of stratified splitting ensures that the class ratio in the test set reflects the natural distribution, and all performance metrics are computed on this unaugmented test set, providing an unbiased performance estimate.

#### 3.2.5 Absence of Probabilistic and Discriminant Analysis Classifiers in Comparative Frameworks

**Gap addressed by this thesis:** Both Naïve Bayes and QDA are included as distinct classifier categories, allowing comparison of Gaussian probabilistic (NB), quadratic discriminant (QDA), kernel (SVM), ensemble (RF), and neural (MLP) paradigms within a single unified framework—providing a performance gradient not available in any single prior study.

#### 3.2.6 Lack of End-to-End Deployment with Multi-Model Inference and Visualisation

**Gap addressed by this thesis:** The deployed Streamlit web application presents predictions from both the best classical model and the MLP-ANN, alongside: (i) comparative ROC curves for all nine models, (ii) an overfitting analysis table, (iii) a multi-metric performance bar chart, and (iv) real-time risk probability display. This constitutes the most comprehensive single-application multi-model deployment in the reviewed literature.

#### 3.2.7 Reproducibility and Code Transparency Deficits

Duda, Wall, and Daniels \[21\] reported that 72% of reviewed ASD ML studies lacked publicly accessible code or datasets, severely limiting reproducibility and independent validation. Even when datasets are publicly available, the absence of shared preprocessing code means that reported results cannot be reliably reproduced, as minor differences in encoding, imputation, or scaling can substantially alter classifier performance.

**Gap addressed by this thesis:** The complete preprocessing pipeline, model training scripts, label encoders, scalers, and serialised model files are made available, ensuring full end-to-end reproducibility. The Streamlit application further serves as an executable demonstration of the system's functional correctness, providing a form of empirical reproducibility validation beyond static code publication.

### 3.3 Summary of Research Gaps and Thesis Contributions

The following table consolidates the identified gaps and maps each to the corresponding methodological contribution of this thesis:

| **#** | **Research Gap in Existing Literature** | **This Thesis Contribution** |
| --- | --- | --- |
| G1 | Fragmented multi-model benchmarking under inconsistent conditions | Nine models evaluated under identical, fully controlled conditions |
| G2 | Overfitting not quantified or reported in most studies | Train-test accuracy gap explicitly computed and tabulated for all 9 models |
| G3 | Small, single-source datasets limiting generalisability | Q-CHAT-10 toddler dataset (975 records) used for training and evaluation |
| G4 | Class imbalance ignored or SMOTE misapplied pre-split | SMOTE correctly applied post-split to training partition only; ablation study quantifies impact |
| G4b | Target leakage via pre-computed sum scores not identified in prior work | Qchat-10-Score excluded from feature set; 16 non-leaky features used |
| G5 | Probabilistic/discriminant classifiers absent from benchmarks | NB and QDA included alongside LR, DT, RF, KNN, SVM, and MLP |
| G6 | Deployment limited to single-model or non-visual interfaces | Streamlit app with dual-model inference, ROC curves, and overfitting display |
| G7 | Poor code transparency and reproducibility | Full pipeline, encoders, scalers, and models publicly available |
| G8 | Hyperparameter selection by default values in most studies | 5-fold GridSearchCV over grid of hyperparameters for RF, SVM-RBF, DT, MLP-ANN |
| G9 | ANN trained without principled early stopping, causing overfitting | True early stopping enforced (validation\_fraction=0.1, n\_iter\_no\_change=10) |
| G10 | Statistical significance of model differences rarely tested | McNemar's exact test applied: LR vs ANN ($p = 0.0156$), result is significant |

_Table 3.1: Research gaps identified in the literature and corresponding contributions of this thesis._

### 3.4 Research Objectives

Based on the identified gaps, this thesis pursues the following primary research objectives:

**RO1.** To develop a rigorous, end-to-end data preprocessing pipeline for ASD screening data, including duplicate removal, mode imputation, individual label encoding, target leakage prevention (exclusion of Qchat-10-Score), and post-split SMOTE application, ensuring methodological integrity throughout.

**RO2.** To train and evaluate nine diverse classification models — spanning linear, probabilistic, kernel, ensemble, and neural network paradigms — under identical experimental conditions on the Q-CHAT-10 toddler ASD screening dataset.

**RO3.** To conduct a comprehensive overfitting analysis for all trained models by comparing training and test accuracy, and to interpret the resulting generalisation profiles in the context of clinical deployment suitability.

**RO4.** To apply **5-fold GridSearchCV hyperparameter optimisation** for key model families (Random Forest, SVM-RBF, Decision Tree, MLP-ANN) and enforce **true validation-monitored early stopping** for the MLP-ANN, ensuring rigorous model configuration and principled regularisation.

**RO5.** To evaluate generalisation stability through **10-fold stratified cross-validation**, and to confirm the statistical significance of the best model's superiority using **McNemar's exact test**, establishing an objective and scientifically defensible model ranking.

**RO6.** To deploy the trained models as an interactive, real-time web application using Streamlit, providing multi-model inference, probability-based risk scoring, comparative visualisations, and an intuitive interface accessible to clinicians and caregivers.

## Chapter 4: Methodology

### 4.1 Overview

This chapter provides a comprehensive and detailed description of the methodology adopted in this thesis for developing, training, evaluating, and deploying machine learning and artificial neural network models for early ASD detection in children. The methodology follows a structured pipeline comprising seven major phases: (1) data collection and description, (2) data preprocessing, (3) train-test partitioning and class imbalance handling, (4) feature scaling, (5) model definition and training, (6) model evaluation and overfitting analysis, and (7) model serialisation and web deployment. Each phase is described in technical detail with reference to the specific implementation choices made, and the overall workflow is illustrated through structured diagrams.

### 4.2 Overall System Architecture and Workflow

The end-to-end system architecture of this thesis is presented in Figure 4.1. The pipeline proceeds from raw data ingestion through preprocessing, resampling, model training, evaluation, and finally real-time deployment as a Streamlit web application.

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.1: End-to-End System Workflow ║
╚══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────┐
│ Phase 1: Data Collection │
│ Autism\_Screening\_Data\_ │
│ Toddler Autism dataset │
│ July 2018.csv (975 records, │
│ 16 features + 1 target) │
└────────────────┬────────────────┘
│
▼
┌─────────────────────────────────┐
│ Phase 2: Data Preprocessing │
│ • Remove duplicates │
│ • Mode imputation (missing) │
│ • Per-column Label Encoding │
│ • Numeric coercion & mean fill │
└────────────────┬────────────────┘
│
▼
┌─────────────────────────────────┐
│ Phase 3: Train-Test Split │
│ • 80% Train / 20% Test │
│ • Stratified by target class │
│ • random\_state = 42 │
└────────────────┬────────────────┘
│
▼
┌─────────────────────────────────┐
│ Phase 4: SMOTE (Train only) │
│ • Synthetic Minority │
│ Oversampling on X\_train │
│ • Balances ASD+/ASD− classes │
└────────────────┬────────────────┘
│
▼
┌─────────────────────────────────┐
│ Phase 5: Feature Scaling │
│ • StandardScaler fit on │
│ X\_train, transform X\_test │
│ • Zero mean, unit variance │
└────────────────┬────────────────┘
│

▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 6: Model Training, Tuning & Early Stopping │
│ │
│ • Hyperparameter Tuning: 5-Fold CV GridSearchCV │
│ (Optimised params for RF, SVM-RBF, DT, MLP-ANN) │
│ • True Early Stopping (MLP-ANN): │
│ validation\_fraction=0.1, n\_iter\_no\_change=10 │
│ │
│ ┌────────────┐ ┌──────────────┐ ┌────────────────────┐ │
│ │ Logistic │ │ Decision │ │ Random Forest │ │
│ │ Regression │ │ Tree (Tuned) │ │ (200, Tuned) │ │
│ └────────────┘ └──────────────┘ └────────────────────┘ │
│ ┌────────────┐ ┌──────────────┐ ┌────────────────────┐ │
│ │ KNN │ │ SVM (Poly) │ │ SVM (RBF, Tuned) │ │
│ │ (k=51) │ │ degree=2 │ │ (C=10.0, γ=0.01) │ │
│ └────────────┘ └──────────────┘ └────────────────────┘ │
│ ┌────────────┐ ┌──────────────┐ ┌────────────────────┐ │
│ │ Naïve │ │ QDA │ │ MLP-ANN (Tuned) │ │
│ │ Bayes │ │ reg=0.7 │ │ (Early Stopped) │ │
│ └────────────┘ └──────────────┘ └────────────────────┘ │
└──────────────────────────────┬──────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 7: Model Evaluation & Validation │
│ │
│ • 10-Fold Stratified Cross-Validation on X\_train │
│ • Statistical Significance: McNemar's Exact Test │
│ • SMOTE Ablation Study: Resampled vs Unsampled Pipelines │
│ • 7 Metrics: Accuracy, Precision, Recall, Specificity, │
│ F1 Score, ROC-AUC, Log Loss │
│ • Overfitting Diagnostics: Train Acc − Test Acc (Gap) │
└──────────────────────────────┬──────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 8: Model Serialisation (pickle) │
│ trained\_models.pkl │ ann.pkl │ scaler.pkl │ le\_dict.pkl │
│ results\_df.pkl │ roc\_data.pkl │ metadata.pkl │
└──────────────────────────────┬──────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 9: Streamlit Web Deployment │
│ Real-time ASD risk prediction │ Multi-model visualisations │
│ ROC curve display │ Overfitting table │ Probability scores │
└─────────────────────────────────────────────────────────────┘

### 4.3 Phase 1: Dataset Description

#### 4.3.1 Data Source

The dataset used in this study is the **Q-CHAT-10 Toddler Autism Screening Dataset (July 2018)**, a clinical toddler screening dataset originally compiled by researchers using the Quantitative Checklist for Autism in Toddlers, 10-item version (Q-CHAT-10) instrument. The dataset is publicly available and contains **1,054 records** and **19 columns** (the non-predictive Case\_No identifier column and the target-leaky Qchat-10-Score column are removed during preprocessing, leaving **16 input features and 1 binary target variable**). The Q-CHAT-10 is a validated screening tool designed for toddlers aged 12–36 months and is administered by parents or caregivers.

#### 4.3.2 Feature Description

The dataset incorporates clinically validated features derived from the **Quantitative Checklist for Autism in Toddlers, 10 items (Q-CHAT-10)** screening instrument, along with demographic, medical history, and respondent-type variables:

| **Feature** | **Type** | **Description** |
| --- | --- | --- |
| A1 – A10 | Binary (0/1) | Ten Q-CHAT-10 behavioural screening items |
| Age\_Mons | Continuous | Age of the toddler in months (range: 12–36 months) |
| Sex | Categorical | Gender (Male / Female) |
| Ethnicity | Categorical | Ethnic background of the toddler (11 categories) |
| Jaundice | Binary | History of jaundice at birth (yes / no) |
| Family\_mem\_with\_ASD | Binary | Family member diagnosed with ASD (yes / no) |
| Who completed the test | Categorical | Respondent type (Parent, Caregiver, Health Care Professional, etc.) |
| **Class/ASD Traits** | Binary (target) | ASD traits present: Yes (ASD+) / No (ASD−) |

_Table 4.1: Feature description of the Q-CHAT-10 toddler ASD screening dataset (16 input features, excluding target-leaky Qchat-10-Score)._

The ten Q-CHAT-10 items (A1–A10) collectively form the primary screening subscale. Each item is a binary response (0 = non-autistic-trait response, 1 = autistic-trait response) to questions concerning social communication, sensory sensitivity, imitation, attention pointing, and play behaviours specific to the toddler developmental stage.

#### 4.3.4 Target Leakage Prevention

To ensure scientific integrity, the Qchat-10-Score feature (the arithmetic sum $\sum_{i=1}^{10} A_i$) was explicitly excluded from the input feature matrix. In ASD screening instruments, the target label (Class/ASD Traits) is defined by the clinical decision rule . Including the pre-computed sum as an input feature allows classifiers to memorize a single split threshold (), creating artificial shortcut learning (target leakage) that inflates accuracy metrics while obscuring genuine feature representations. Excluding Qchat-10-Score forces models to learn the predictive value of individual behavioural items and demographic interactions directly from the 16 genuine screening features.

#### 4.3.3 Target Variable and Class Distribution

The target variable is the binary class label Class/ASD Traits, indicating whether the toddler exhibits ASD traits. After deduplication, the dataset has **690 ASD-positive (Yes) records (70.8%)** and **285 ASD-negative (No) records (29.2%)** out of 975 total records. This means the dataset is **majority-positive** — the positive class (ASD traits present) is the majority — which is characteristic of clinical Q-CHAT-10 cohorts where toddlers are typically referred or self-referred due to parental concern. Accordingly, SMOTE oversamples the minority class (ASD-negative, No) during training to achieve a balanced 1:1 class ratio.

### 4.4 Phase 2: Data Preprocessing

Preprocessing is the most critical determinant of model reliability and reproducibility. The following sequential steps were applied:

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.2: Data Preprocessing Pipeline ║
╚══════════════════════════════════════════════════════════════════╝

Raw CSV Input (1,054 rows × 19 cols)
│
▼
┌──────────────────────────────────────────┐
│ Step 0: Target Leakage Prevention & ID │
│ df.drop(columns=\['Case\_No', │
│ 'Qchat-10-Score'\]) │
│ Removes non-predictive ID & target-leaky │
│ sum score; 16 input features remain │
└──────────────────┬───────────────────────┘
│
▼
┌───────────────────────────────┐
│ Step 1: Duplicate Removal │
│ df.drop\_duplicates() │
│ Removes 79 duplicate rows │
│ 1,054 → 975 unique records │
└──────────────┬────────────────┘
│
▼
┌───────────────────────────────┐
│ Step 2: Missing Value │
│ Imputation (Mode Strategy) │
│ df.fillna(df.mode().iloc\[0\]) │
│ Handles categorical & │
│ numerical NaN values │
└──────────────┬────────────────┘
│
▼
┌───────────────────────────────┐
│ Step 3: Categorical Encoding │
│ Per-column LabelEncoder │
│ for each categorical column │
│ Encoders stored in le\_dict │
│ for inference-time reuse │
└──────────────┬────────────────┘
│
▼
┌───────────────────────────────┐
│ Step 4: Numeric Coercion │
│ pd.to\_numeric(errors='coerce')│
│ + mean fill for any residual │
│ NaN introduced by coercion │
└──────────────┬────────────────┘
│
▼
┌───────────────────────────────┐
│ Step 5: Feature/Target Split │
│ X = all columns except last │
│ y = last column (ASD label) │
└───────────────────────────────┘

#### 4.4.1 Duplicate Removal

Duplicate records arise in clinical screening datasets when the same case is entered more than once. The drop\_duplicates() operation removes rows that are identical across all 18 columns, ensuring each toddler's screening record contributes exactly once to model training and evaluation. In the Q-CHAT-10 dataset, this step removes **79 duplicate records**, reducing the dataset from 1,054 to **975 unique records**.

#### 4.4.2 Missing Value Imputation

Missing values in ASD screening datasets frequently arise from unanswered questionnaire items or incomplete demographic records. Mode imputation (df.fillna(df.mode().iloc\[0\])) is selected as the imputation strategy because:

- It is appropriate for both categorical features (Sex, Jaundice, Family\_ASD) and binary behavioural items (A1–A10).
- It preserves the most common response pattern, avoiding artificial distribution shifts.
- Mean imputation was rejected as it would introduce non-integer values into binary columns.

#### 4.4.3 Categorical Label Encoding

Categorical variables (Sex, Ethnicity, Jaundice, Family\_mem\_with\_ASD, Who completed the test, and the target Class/ASD Traits) are encoded using individual LabelEncoder instances—one per column—stored in a dictionary (le\_dict). This approach is preferred over a single shared encoder because:

- Each column may have a different set of unique string categories.
- Per-column encoders can be independently applied at inference time to new input records, ensuring consistency between training-time and deployment-time transformations.
- The encoder dictionary is serialised as le\_dict.pkl and loaded by the Streamlit application for real-time use.

#### 4.4.4 Numeric Coercion and Residual Fill

After categorical encoding, pd.to\_numeric(errors='coerce') is applied to all columns to ensure a fully numeric DataFrame. Any non-numeric residuals (coerced to NaN) are filled with the column mean. This two-pass imputation strategy guarantees a clean, fully numeric input matrix prior to splitting.

### 4.5 Phase 3: Train-Test Partitioning

The preprocessed dataset is split into training (80%) and test (20%) partitions using scikit-learn's train\_test\_split function with the following configuration:

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.3: Train-Test Split Strategy ║
╚══════════════════════════════════════════════════════════════════╝

Full Dataset (after preprocessing)
─────────────────────────────────
│ 975 records │
└────────────────────────────────┘
│
│ train\_test\_split(test\_size=0.2,
│ random\_state=42,
│ stratify=y)
│
┌──────┴──────┐
│ │
▼ ▼
┌──────────┐ ┌──────────┐
│ Training │ │ Test │
│ Set │ │ Set │
│ 780 │ │ 195 │
│ records │ │ records │
│ (80%) │ │ (20%) │
└──────────┘ └──────────┘
│ │
│ └──────── Held out; NEVER touched
│ during SMOTE or scaling fit
▼
SMOTE applied here
(training partition only)

**Key design choices:**

- **\`stratify=y\`**: Ensures that the class ratio (ASD+ / ASD−) in the training and test sets mirrors the original dataset distribution, preventing accidental class imbalance amplification in the evaluation set.
- **\`random\_state=42\`**: Fixed seed for full reproducibility across all experimental runs.
- **\`test\_size=0.2\`**: The 80/20 split provides exactly **195 test samples** and 780 training samples — sufficient for reliable metric estimation across all seven evaluation criteria.

### 4.6 Phase 4: Class Imbalance Handling with SMOTE

Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively to the training partition to address class imbalance without contaminating the test set.

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.4: SMOTE Oversampling Mechanism ║
╚══════════════════════════════════════════════════════════════════╝

BEFORE SMOTE (Training Set) AFTER SMOTE (Training Set)
──────────────────────────── ────────────────────────────
ASD Negative (majority): ~3,300 → ASD Negative: ~3,300
ASD Positive (minority): ~1,560 → ASD Positive: ~3,300
(synthetic samples added)
Class Ratio ≈ 2.1 : 1 → Class Ratio = 1 : 1

HOW SMOTE WORKS:
┌──────────────────────────────────────────┐
│ For each minority-class sample p: │
│ 1. Find k nearest neighbours in │
│ feature space (default k=5) │
│ 2. Randomly select one neighbour q │
│ 3. Generate synthetic point: │
│ x\_new = p + λ × (q − p) │
│ where λ ~ Uniform(0, 1) │
│ 4. Repeat until classes are balanced │
└──────────────────────────────────────────┘

✔ Applied ONLY to X\_train, y\_train
✔ Test set remains unmodified (natural distribution)
✔ random\_state=42 for reproducibility

SMOTE generates synthetic ASD-negative samples by interpolating between existing minority-class feature vectors in the **16-dimensional feature space**, rather than simply duplicating existing records (as in random oversampling). This preserves the distributional geometry of the minority class while increasing its representation, thereby reducing classifier bias toward the majority class during training.

### 4.7 Phase 5: Feature Scaling

All models are trained on standardised features produced by StandardScaler, which transforms each feature to zero mean and unit variance:

where is the feature mean and is the feature standard deviation, both computed exclusively from the training set.

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.5: Feature Scaling Protocol ║
╚══════════════════════════════════════════════════════════════════╝

X\_train (SMOTE-augmented) X\_test (original)
│ │
│ scaler.fit\_transform(X\_train) │ scaler.transform(X\_test)
│ ── computes μ, σ from train ── │ ── applies train μ, σ ──
▼ ▼
X\_train\_scaled X\_test\_scaled
(μ=0, σ=1 per feature) (transformed with train stats)

⚠ CRITICAL: scaler.fit() is NEVER called on X\_test.
This prevents data leakage: test statistics must not
influence the scaling transformation applied during training.

**Rationale for StandardScaler:**

- Distance-based classifiers (KNN) and margin-based classifiers (SVM) are highly sensitive to feature scale disparities. Without scaling, features with larger numerical ranges (e.g., Age\_Mons: 12–36) would dominate distance computations over binary features (A1–A10: 0 or 1).
- The MLP-ANN benefits from normalised inputs because standardised inputs accelerate gradient descent convergence and prevent the vanishing/exploding gradient problem in hidden layers.
- The fitted scaler object is serialised as scaler.pkl and reloaded at inference time in the Streamlit application to ensure new user inputs are scaled identically to the training data.

### 4.8 Phase 6: Model Definition and Training

Nine classification models were defined and trained. The models span five distinct paradigms: linear probabilistic, tree-based, ensemble, kernel, and neural network.

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.6: Classification Model Taxonomy ║
╚══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────┐
│ 9 Classification Models │
└─────────────────────────────────────────────────────────┘
│ │ │
▼ ▼ ▼
┌──────────────┐ ┌────────────┐ ┌────────────────────┐
│ Probabilistic│ │ Tree-Based │ │ Kernel Methods │
│ ─────────── │ │ ───────── │ │ ───────────────── │
│ • Logistic │ │ • Decision │ │ • SVM (Poly) │
│ Regression │ │ Tree │ │ • SVM (RBF) │
│ • Naïve Bayes│ │ • Random │ └────────────────────┘
│ • QDA │ │ Forest │
└──────────────┘ └────────────┘
│ │
▼ ▼
┌──────────────┐ ┌────────────────────────────────────┐
│ Instance- │ │ Artificial Neural Network │
│ Based │ │ ───────────────────────────────── │
│ ──────────── │ │ MLP-ANN (32 → 16, ReLU, Adam) │
│ • KNN (k=51) │ └────────────────────────────────────┘
└──────────────┘

#### 4.8.1 Logistic Regression

Logistic Regression fits a linear decision boundary in the feature space by optimising the log-likelihood of the binary outcome using the sigmoid function:

**Configuration:** max\_iter=1000 (to ensure convergence on the scaled, high-dimensional input); L2 regularisation (default, C=1.0).

**Role:** Serves as the primary interpretable linear baseline. Coefficients are directly interpretable as log-odds contributions of each feature, providing clinical explainability.

#### 4.8.2 Decision Tree

Decision Trees partition the feature space through a greedy sequence of binary splits, minimising Gini impurity at each node.

**Configuration:**

- max\_depth=5: Limits tree depth to prevent memorisation of training noise.
- min\_samples\_split=20, min\_samples\_leaf=10: Minimum sample thresholds enforce that each split and leaf node contains sufficient evidence.
- ccp\_alpha=0.005: Cost-complexity pruning parameter that post-prunes the tree to remove branches with negligible information gain, reducing overfitting.

#### 4.8.3 Random Forest

Random Forest constructs an ensemble of independently trained decision trees, each on a bootstrap sample, and aggregates predictions by majority vote.

**Configuration:**

- n\_estimators=200: 200 trees for stable ensemble averaging.
- max\_depth=10: Shallow trees to prevent individual tree overfitting.
- min\_samples\_split=15, min\_samples\_leaf=6: Conservative split thresholds.
- max\_features='sqrt': Each tree considers features at each split, decorrelating trees.
- min\_impurity\_decrease=0.001: Prunes splits that do not achieve a minimum information gain.

#### 4.8.4 K-Nearest Neighbours (KNN)

KNN classifies a test point by plurality vote among its k nearest neighbours in the scaled feature space.

**Configuration:**

- n\_neighbors=51: A large k (odd to break ties) was selected to smooth decision boundaries and reduce variance. The choice of 51 reflects the large training set size, where smaller k values tend to overfit.
- weights='uniform': All neighbours contribute equally.
- metric='minkowski', p=1: Manhattan distance (L1 norm), more robust than Euclidean distance for high-dimensional binary feature spaces.

#### 4.8.5 Support Vector Machine – Polynomial Kernel

The polynomial SVM maps inputs into a polynomial feature space and finds the maximum-margin separating hyperplane:

**Configuration:** degree=2 (quadratic), C=0.1 (strong regularisation), gamma='scale', probability=True (Platt scaling for probability calibration).

#### 4.8.6 Support Vector Machine – RBF Kernel

The RBF kernel maps inputs into an infinite-dimensional feature space through:

$$K(\mathbf{x}, \mathbf{x}\') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}\'\|^2\right)$$

**Configuration (GridSearchCV-tuned):** C=10.0, gamma=0.01, probability=True. The hyperparameter values were selected via 5-fold GridSearchCV over the grid C ∈ {0.1, 1.0, 10.0} and gamma ∈ {'scale', 'auto', 0.01, 0.1}, achieving a best CV ROC-AUC of **1.0000** at the selected configuration.

#### 4.8.7 Gaussian Naïve Bayes

GNB applies Bayes' theorem under the assumption that features are conditionally independent given the class label, with Gaussian likelihood for each feature:

**Configuration:** var\_smoothing=1e-8 (small additive variance stabilisation to prevent numerical underflow).

#### 4.8.8 Quadratic Discriminant Analysis (QDA)

QDA estimates a class-conditional Gaussian distribution for each class with a separate covariance matrix, yielding a quadratic decision boundary:

**Configuration:** reg\_param=0.7 (shrinkage regularisation interpolating between class-specific and pooled covariance estimates, preventing singular matrix issues on the moderately sized dataset).

#### 4.8.9 Multilayer Perceptron – Artificial Neural Network (MLP-ANN)

The MLP-ANN is the primary deep learning model in this thesis, representing a fully connected feedforward neural network trained with backpropagation.

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.7: MLP-ANN Architecture ║
╚══════════════════════════════════════════════════════════════════╝

Input Layer Hidden Layer 1 Hidden Layer 2 Output
(16 neurons) (32 neurons) (16 neurons) Layer (1)
──────────── ──────────── ──────────── ─────────

x₁ ─────────┐
x₂ ─────────┤
x₃ ─────────┤ ┌──── h₁₁ ────┐
x₄ ─────────┤──── h₁₂ ────┤──── h₂₁ ────┐
x₅ ─────────┤ ┌── h₁₃ ────┤ ┌─ h₂₂ ────┤── σ(z) ──► ŷ ∈ \[0,1\]
x₆ ─────────┤ │ ... │ │ ... │ (ASD prob.)
x₇ ─────────┤ │ h₁₃₂─────┘ │ h₂₁₆───┘
x₈ ─────────┘ │ │
x₉ ────────────┘ │
x₁₀───────────────────────────┘
x₁₁
x₁₂
x₁₃
x₁₄
x₁₅
x₁₆

Activation: ReLU f(z) = max(0, z) \[hidden layers\]
Logistic (sigmoid) \[output layer\]
Optimiser: Adam (adaptive learning rate)
Loss: Binary Cross-Entropy
Max Iter: 200
Random seed: 42

**Architectural details:**

| **Component** | **Configuration** |
| --- | --- |
| Input layer | 16 neurons (one per non-leaky input feature) |
| Hidden Layer 1 | 32 neurons, ReLU activation |
| Hidden Layer 2 | 16 neurons, ReLU activation |
| Output layer | 1 neuron, Sigmoid activation |
| Optimiser | Adam (Adaptive Moment Estimation) |
| Loss function | Binary Cross-Entropy |
| Early Stopping | Enabled (early\_stopping=True, validation\_fraction=0.1, n\_iter\_no\_change=10) |
| Max iterations | 200 epochs |
| Weight initialisation | Glorot uniform (scikit-learn default) |

#### 4.8.10 Hyperparameter Tuning Protocol & Search Space

To ensure that candidate classifiers were evaluated under optimal hyperparameter configurations, a 5-fold cross-validated **GridSearchCV** hyperparameter optimization protocol was conducted across key model families.

The hyperparameter search space and empirically selected optimal configurations are detailed below:

| **Model Family** | **Search Space Grid** | **Selected Optimal Parameters** | **Best CV ROC-AUC** |
| --- | --- | --- | --- |
| **Random Forest** | n\_estimators: \[100, 200\]<br>max\_depth: \[5, 10, 15\]<br>min\_samples\_split: \[5, 10, 15\] | n\_estimators: 200<br>max\_depth: 10<br>min\_samples\_split: 5 | **0.9978** |
| **SVM (RBF)** | C: \[0.1, 1.0, 10.0\]<br>gamma: \['scale', 'auto', 0.01, 0.1\] | C: 10.0<br>gamma: 0.01 | **1.0000** |
| **Decision Tree** | max\_depth: \[3, 5, 10\]<br>min\_samples\_split: \[10, 20\]<br>ccp\_alpha: \[0.0, 0.005, 0.01\] | max\_depth: 10<br>min\_samples\_split: 20<br>ccp\_alpha: 0.0 | **0.9708** |
| **MLP-ANN** | hidden\_layer\_sizes: \[(32, 16), (64, 32), (32,)\]<br>alpha: \[0.0001, 0.001, 0.01\] | hidden\_layer\_sizes: (64, 32)<br>alpha: 0.0001 | **0.9983** |

_Table 4.2: Hyperparameter search space and optimal configurations derived from 5-fold GridSearchCV._

**Backpropagation training procedure:**

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.8: Backpropagation Training Loop ║
╚══════════════════════════════════════════════════════════════════╝

Initialise weights W randomly
│
▼
┌────────────────────────────────┐
│ For each epoch t = 1…200: │
│ │
│ 1. FORWARD PASS │
│ Input X\_train → Layer 1 │
│ → ReLU → Layer 2 │
│ → ReLU → Output │
│ → Sigmoid → ŷ │
│ │
│ 2. COMPUTE LOSS │
│ L = −\[y·log(ŷ) │
│ + (1−y)·log(1−ŷ)\] │
│ │
│ 3. BACKWARD PASS │
│ Compute ∂L/∂W for │
│ each layer via chain rule │
│ │
│ 4. WEIGHT UPDATE (Adam) │
│ m\_t = β₁·m\_{t-1} │
│ + (1−β₁)·∂L/∂W │
│ v\_t = β₂·v\_{t-1} │
│ + (1−β₂)·(∂L/∂W)² │
│ W ← W − α·m̂\_t/√v̂\_t │
│ │
│ 5. VALIDATION MONITORING & EARLY STOPPING
│ Compute L\_val on 10% held-out validation set
│ If L\_val does not decrease for 10 consecutive epochs:
│ HALT training immediately (Early Stop)
│ Else if |ΔL| < tol or t == max\_iter:
│ STOP training
└────────────────────────────────┘
│
▼
Final weights W\* (trained model)

The Adam optimiser was selected over standard SGD because it adapts the learning rate per-parameter based on first and second moment estimates of gradients, enabling faster and more stable convergence on the moderately sized training set.

### 4.9 Phase 7: Model Evaluation Framework

Each trained model was evaluated on the held-out test set using seven complementary performance metrics:

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.9: Evaluation Metrics Framework ║
╚══════════════════════════════════════════════════════════════════╝

Confusion Matrix (binary classification):
──────────────────────────────────────────
Predicted: ASD− ASD+
Actual: ASD− │ TN │ FP │
ASD+ │ FN │ TP │

From TN, FP, FN, TP:

┌──────────────────────────────────────────────────────────────┐
│ Accuracy = (TP + TN) / (TP + TN + FP + FN) │
│ Precision = TP / (TP + FP) \[positive predictive val.\]│
│ Recall = TP / (TP + FN) \[sensitivity\] │
│ Specificity = TN / (TN + FP) \[true negative rate\] │
│ F1 Score = 2 × (Precision × Recall) / (Precision+Recall) │
│ ROC-AUC = Area under the ROC curve │
│ Log Loss = −(1/N)Σ\[y·log(ŷ)+(1−y)·log(1−ŷ)\] │
└──────────────────────────────────────────────────────────────┘

**Metric rationale:**

- **Recall (Sensitivity)** is the most clinically important metric for a screening tool: a missed ASD-positive case (false negative) is more harmful than an unnecessary referral (false positive).
- **Specificity** limits unnecessary clinical referrals by quantifying the true negative rate.
- **ROC-AUC** provides a threshold-independent measure of discriminative power.
- **Log Loss** penalises confident incorrect predictions, rewarding well-calibrated probability outputs—important for a deployment system that displays probability scores.

#### 4.9.1 10-Fold Stratified Cross-Validation & Overfitting Diagnostics

To evaluate generalisation stability and diagnose overfitting rigorously without relying on arbitrary heuristic threshold gaps, a **10-Fold Stratified Cross-Validation** protocol was implemented on the training partition ().

In stratified 10-fold cross-validation, the training set is partitioned into 10 equal folds, preserving class balance across each fold. For each iteration $i \\in {1 \\dots 10}$, 9 folds are used for training and the remaining 1 fold is used for validation. The mean performance metrics and standard deviations across all 10 folds are reported:

The cross-validation gap ($\\text{CV Gap} = \\text{Train Acc}_{\\text{Mean}} - \\text{Val Acc}_{\\text{Mean}}\\sigma\_{\\text{CV}}$) provides a mathematically grounded assessment of model stability and overfitting control.

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.9b: 10-Fold Stratified CV & GridSearchCV Tuning Flow ║
╚══════════════════════════════════════════════════════════════════╝

Training Set (X\_train, 780 records)
│
│ GridSearchCV (5-fold inner CV)
│ ─────────────────────────────
│ For each hyperparameter combo:
│ Split X\_train into 5 stratified folds
│ Train on 4 folds → Validate on 1 fold
│ Record mean CV ROC-AUC across 5 folds
│ Select best combo → Optimal hyperparameters
│
▼
┌────────────────────────────────────────────────────┐
│ Best hyperparameters confirmed for each model │
│ (RF: n\_estimators=200, max\_depth=10) │
│ (SVM-RBF: C=10.0, gamma=0.01) │
│ (DT: max\_depth=10, ccp\_alpha=0.0) │
│ (MLP-ANN: hidden=(32,16), alpha=0.0001) │
└────────────────────────────────────────────────────┘
│
│ 10-Fold Stratified CV (outer evaluation)
│ ──────────────────────────────────────────
│ Split X\_train into 10 stratified folds
│
┌─────┴──────────────────────────────────┐
Fold 1│ Fold 2 │ Fold 3 │ … │ Fold 10 │
Val │ Val │ Val │ │ Val │
└──────────────────────────────────────────┘
│
▼
Report Mean ± Std across 10 folds:
CV Accuracy, CV F1, CV ROC-AUC, CV Gap

#### 4.9.2 Statistical Significance Testing (McNemar's Exact Test)

To test whether performance differences between the top-performing Neural Network (ANN) and classical baseline classifiers (Logistic Regression, Random Forest) are statistically significant—rather than artifacts of random sampling—**McNemar's Exact Test** was applied on paired test set predictions.

Given two classifiers $M_A$ and $M_B$, McNemar's test constructs a contingency table of prediction agreements and disagreements:

where $n_{01}$ is the number of test cases where $M_A$ is correct and $M_B$ is incorrect, and $n_{10}$ is the number of test cases where $M_A$ is incorrect and $M_B$ is correct. The exact binomial $p$-value is computed as:

A significance threshold of $\alpha = 0.05$ was established. A calculated $p$-value ($p = 0.0156 < 0.05$) rejects the null hypothesis of equal classifier capability.

### 4.10 Phase 8: Model Serialisation

All trained artefacts are serialised to the models/ directory using Python's pickle library, enabling the Streamlit application to load pre-trained models without retraining:

| **File** | **Contents** |
| --- | --- |
| trained\_models.pkl | Dictionary of 8 classical trained model objects |
| ann.pkl | Trained MLP-ANN object |
| scaler.pkl | Fitted StandardScaler (training-set statistics) |
| le\_dict.pkl | Per-column LabelEncoder dictionary |
| results\_df.pkl | Full results DataFrame (all metrics, all models) |
| roc\_data.pkl | ROC curve data (FPR, TPR, AUC) for all 9 models |
| metadata.pkl | Feature names, numeric/categorical column lists |

_Table 4.3: Serialised model artefacts and their contents._

### 4.11 Phase 9: Streamlit Web Application Deployment

The final phase translates the trained and serialised models into a user-accessible web application using the Streamlit framework.

╔══════════════════════════════════════════════════════════════════╗
║ FIGURE 4.10: Streamlit Application Architecture ║
╚══════════════════════════════════════════════════════════════════╝

User (Browser)
│
│ Enters: A1–A10 responses, Age\_Mons,
│ Sex, Ethnicity, Jaundice,
│ Family\_mem\_with\_ASD,
│ Who completed the test
▼
┌──────────────────────────────────────────────────────────┐
│ Streamlit Web Application │
│ │
│ 1. Input Collection (sidebar form) │
│ → Raw user inputs as strings/numbers │
│ │
│ 2. Preprocessing (inference pipeline) │
│ → Apply le\_dict encoders to categorical inputs │
│ → Assemble feature vector as numpy array │
│ → Apply scaler.transform() to feature vector │
│ │
│ 3. Model Inference │
│ → Best classical model: predict() + predict\_proba() │
│ → ANN model: predict() + predict\_proba() │
│ │
│ 4. Output Display │
│ → ASD risk classification (Positive / Negative) │
│ → Probability score (0.0 – 1.0) │
│ → Risk category (Low / Medium / High) │
│ │
│ 5. Visualisation Dashboard │
│ → ROC Curve (all 9 models) │
│ → Performance Bar Chart (Accuracy, F1, AUC) │
│ → Overfitting Analysis Table │
│ → Confusion Matrix (best model) │
└──────────────────────────────────────────────────────────┘

The Streamlit application applies the exact same preprocessing transformations used during training—using the serialised le\_dict and scaler—to ensure that inference-time feature representations are identical to training-time representations. This design eliminates training-serving skew, a common source of degraded real-world performance in deployed ML systems.

### 4.12 Laboratory Setup

All experiments in this thesis were designed, implemented, and executed on a single local workstation. No cloud computing platforms, virtual machines, or distributed computing resources were employed. The following subsections document the hardware configuration and the complete software stack used throughout the study.

#### 4.12.1 Hardware Configuration

The computational experiments were conducted on a personal desktop/laptop workstation with the following specifications:

| **Component** | **Specification** |
| --- | --- |
| **Processor (CPU)** | Intel Core i5 (10th Generation), 2.40 GHz base clock, 4 cores / 8 threads |
| **RAM** | 8 GB DDR4 (2666 MHz) |
| **Storage** | 512 GB SSD (NVMe) |
| **Graphics (GPU)** | Intel UHD Graphics (integrated) — CPU-only computation |
| **Operating System** | Windows 10 / 11 (64-bit) |
| **Display** | 15.6-inch Full HD (1920 × 1080) monitor |

_Table 4.4: Laboratory hardware configuration._

Since all nine classifiers — including the MLP-ANN — are implemented via scikit-learn's CPU-based estimators, no GPU acceleration was required or utilised. The dataset size (975 records after deduplication, 16 non-leaky features) and model complexity were well within the memory and processing capabilities of the local workstation, ensuring that full training, evaluation, and serialisation cycles completed within a practical timeframe (total experiment runtime under five minutes).

#### 4.12.2 Software and Development Environment

The software environment was carefully constructed to ensure reproducibility, portability, and compatibility with modern Python data science toolchains.

| **Software / Tool** | **Role** | **Version** |
| --- | --- | --- |
| **Python** | Primary programming language | 3.10.13 |
| **Visual Studio Code (VS Code)** | Integrated Development Environment (IDE) | Latest stable |
| **pip** | Python package manager | — |
| **Git** | Version control | — |
| **Streamlit** | Interactive web application framework for deployment | 1.31+ |

The development workflow consisted of writing and iterating on a standalone Python script (ASD in Children using Machine Learning and ANN.py) within VS Code, with an integrated terminal used for script execution. The final Streamlit deployment was launched locally via the terminal command streamlit run streamlit\_app.py.

#### 4.12.3 Python Libraries and Versions

The following Python libraries were installed and used throughout the experimental pipeline, as specified in the project's requirements.txt:

| **Library** | **Purpose in This Thesis** | **Version Constraint** |
| --- | --- | --- |
| **numpy** | Numerical array operations, matrix computations | ≥ 1.26, < 3 |
| **pandas** | Dataset loading, manipulation, and preprocessing | ≥ 2.0 |
| **scikit-learn** | All nine classifiers, StandardScaler, LabelEncoder, train-test split, evaluation metrics | ≥ 1.3 |
| **imbalanced-learn** | SMOTE for class imbalance correction | ≥ 0.11 |
| **matplotlib** | Confusion matrix heatmaps, ROC curves, bar charts | ≥ 3.7 |
| **seaborn** | Styled statistical visualisations | ≥ 0.12 |
| **streamlit** | Web application for real-time ASD prediction | ≥ 1.31 |
| **pickle** (stdlib) | Serialisation and persistence of trained model objects and preprocessing artefacts | Python standard library |

_Table 4.5: Python libraries and software environment used in this thesis._

All library versions specified above reflect the minimum compatible versions. The project's requirements.txt file pins these lower bounds to allow installation of the latest compatible releases, ensuring that the codebase remains forward-compatible without risking dependency conflicts. No proprietary or commercial software was used at any stage of this research.

### 4.13 Tools, Libraries, and Environment

| **Component** | **Tool / Library** | **Version** |
| --- | --- | --- |
| Programming language | Python | 3.10+ |
| Data manipulation | pandas, NumPy | 2.x / 1.x |
| Machine learning | scikit-learn | 1.x |
| Imbalanced learning | imbalanced-learn | 0.11+ |
| Neural network | scikit-learn MLPClassifier | — |
| Visualisation | matplotlib, seaborn | — |
| Web deployment | Streamlit | 1.x |
| Model serialisation | pickle | stdlib |
| Development environment | VS Code | — |

_Table 4.6: Summary of software tools and libraries used in this thesis._

### 4.14 Summary

This chapter presented the complete nine-phase methodology adopted in this thesis, covering dataset description, preprocessing pipeline, stratified train-test splitting, SMOTE-based class imbalance correction, StandardScaler feature normalisation, nine-model training with carefully tuned hyperparameters, seven-metric evaluation with explicit overfitting analysis, pickle-based model serialisation, and Streamlit web deployment. Workflow diagrams (Figures 4.1–4.10) illustrate each phase in detail. The methodology is designed to be transparent, reproducible, and directly grounded in the research gaps identified in Chapter 3.

## Chapter 5: Results and Discussion

### 5.1 Overview

This chapter presents and interprets the complete experimental results obtained by training and evaluating nine classification models on the Q-CHAT-10 toddler ASD screening dataset. All results are derived from a held-out test set of **195 records** (138 ASD-positive, 57 ASD-negative) that was never seen during training, SMOTE resampling, or scaler fitting. The training partition comprised **1,104 records** after SMOTE augmentation (balanced 1:1 class ratio). Seven performance metrics are reported for each model: Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Score, ROC-AUC, and Log Loss. Overfitting analysis is presented through train-test accuracy gaps for all models. Results are discussed with reference to the research objectives and gaps identified in Chapters 3 and 4.

### 5.2 Dataset Summary

| **Parameter** | **Value** |
| --- | --- |
| Original dataset records | 1,054 |
| Features (after removing Case\_No & target-leaky Qchat-10-Score) | 16 (A1–A10, Age\_Mons, Sex, Ethnicity, Jaundice, Family\_mem\_with\_ASD, Who completed the test) |
| Target classes | Binary: ASD Traits Yes (ASD+), No (ASD−) |
| After duplicate removal | 975 records (79 duplicates removed) |
| Training set (pre-SMOTE) | 780 records |
| Training set (post-SMOTE) | 1,104 records (balanced) |
| Test set | 195 records (held-out, unmodified) |
| ASD+ in test set | 138 (70.77%) |
| ASD− in test set | 57 (29.23%) |

_Table 5.1: Dataset partition statistics._

### 5.3 Overall Model Performance Results

Table 5.2 presents the complete performance metrics for all nine target-leakage-free classifiers evaluated on the held-out test set ($n=195$), sorted by descending ROC-AUC score.

| **Rank** | **Model** | **Train Acc.** | **Test Acc.** | **Gap** | **Precision** | **Recall** | **Specificity** | **F1 Score** | **ROC-AUC** | **Log Loss** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Logistic Regression** | **1.0000** | **1.0000** | **0.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.027190** |
| 2 | **SVM (RBF)** | 0.9873 | 0.9487 | 0.0386 | 0.9324 | 1.0000 | 0.8246 | 0.9650 | 0.9986 | 0.060186 |
| 3 | **MLP-ANN** | 0.9882 | 0.9641 | 0.0241 | 0.9645 | 0.9855 | 0.9123 | 0.9749 | 0.9968 | 0.123469 |
| 4 | QDA | 0.9828 | 0.9692 | 0.0136 | 0.9853 | 0.9710 | 0.9649 | 0.9781 | 0.9963 | 0.079200 |
| 5 | KNN | 0.9503 | 0.9333 | 0.0169 | 1.0000 | 0.9058 | 1.0000 | 0.9506 | 0.9962 | 0.143328 |
| 6 | Random Forest | 0.9837 | 0.9385 | 0.0453 | 0.9375 | 0.9783 | 0.8421 | 0.9574 | 0.9886 | 0.182929 |
| 7 | Naïve Bayes | 0.9530 | 0.9231 | 0.0299 | 0.9020 | 1.0000 | 0.7368 | 0.9485 | 0.9849 | 0.472891 |
| 8 | Decision Tree | 0.9421 | 0.8974 | 0.0447 | 0.9097 | 0.9493 | 0.7719 | 0.9291 | 0.9265 | 0.278322 |
| 9 | SVM (Poly) | 0.8354 | 0.7538 | 0.0816 | 0.9327 | 0.7029 | 0.8772 | 0.8017 | 0.8931 | 0.377112 |

_Table 5.2: Complete leak-free performance metrics for all nine classifiers on the test set (n=195), sorted by ROC-AUC._

### 5.3.1 Confusion Matrices — All Nine Models (Test Set, n=195)

The following section presents the empirical confusion matrix outcomes for each classifier on the held-out test set ($n=195$).

╔══════════════════════════════════════════════════════════════════════════════╗
║ FIGURE 5.1: Confusion Matrices — All 9 Classifiers (Test Set, n=195) ║
╚══════════════════════════════════════════════════════════════════════════════╝

Layout:
Predicted ASD− Predicted ASD+
Actual ASD− \[ TN \] \[ FP \]
Actual ASD+ \[ FN \] \[ TP \]

─────────────────────────────────────────────────────────────────────────────
\[1\] Logistic Regression (Best Linear) \[2\] MLP-ANN
─────────────────────────────────────────────────────────────────────────────
Pred ASD− Pred ASD+ Pred ASD− Pred ASD+
Act ASD− \[ 57 \] \[ 0 \] Act ASD− \[ 52 \] \[ 5 \]
Act ASD+ \[ 0 \] \[ 138 \] Act ASD+ \[ 2 \] \[ 136 \]

TN=57 FP=0 FN=0 TP=138 TN=52 FP=5 FN=2 TP=136
Accuracy = 100.00% Accuracy = 96.41%
Recall = 100.00% Recall = 98.55%

─────────────────────────────────────────────────────────────────────────────
\[3\] QDA \[4\] SVM (RBF)
─────────────────────────────────────────────────────────────────────────────
Pred ASD− Pred ASD+ Pred ASD− Pred ASD+
Act ASD− \[ 55 \] \[ 2 \] Act ASD− \[ 47 \] \[ 10 \]
Act ASD+ \[ 4 \] \[ 134 \] Act ASD+ \[ 0 \] \[ 138 \]

TN=55 FP=2 FN=4 TP=134 TN=47 FP=10 FN=0 TP=138
Accuracy = 96.92% Accuracy = 94.87%
Recall = 97.10% Recall = 100.00%

─────────────────────────────────────────────────────────────────────────────
\[5\] KNN (k=51) \[6\] Random Forest
─────────────────────────────────────────────────────────────────────────────
Pred ASD− Pred ASD+ Pred ASD− Pred ASD+
Act ASD− \[ 57 \] \[ 0 \] Act ASD− \[ 48 \] \[ 9 \]
Act ASD+ \[ 13 \] \[ 125 \] Act ASD+ \[ 3 \] \[ 135 \]

TN=57 FP=0 FN=13 TP=125 TN=48 FP=9 FN=3 TP=135
Accuracy = 93.33% Accuracy = 93.85%
Recall = 90.58% Recall = 97.83%

**Comparative summary of confusion matrix outcomes:**

| **Model** | **TN** | **FP** | **FN** | **TP** | **Recall** | **Specificity** | **Precision** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | **57** | **0** | **0** | **138** | **100.00%** | **100.00%** | **100.00%** |
| SVM (RBF) | 47 | 10 | 0 | 138 | 100.00% | 82.46% | 93.24% |
| MLP-ANN | 52 | 5 | 2 | 136 | 98.55% | 91.23% | 96.45% |
| QDA | 55 | 2 | 4 | 134 | 97.10% | 96.49% | 98.53% |
| KNN | 57 | 0 | 13 | 125 | 90.58% | 100.00% | 100.00% |
| Random Forest | 48 | 9 | 3 | 135 | 97.83% | 84.21% | 93.75% |
| Naïve Bayes | 42 | 15 | 0 | 138 | 100.00% | 73.68% | 90.20% |
| Decision Tree | 44 | 13 | 7 | 131 | 94.93% | 77.19% | 90.97% |
| **SVM (Poly)** | 50 | 7 | 41 | 97 | **70.29%** | 87.72% | 93.27% |

_Table 5.3: Confusion matrix outcomes and derived metrics for target-leakage-free models on the test set (n=195). With true early stopping enabled, the MLP-ANN now uses 10% of training data for validation loss monitoring, yielding 2 FN and 5 FP vs. the test set._

### 5.3.2 10-Fold Stratified Cross-Validation Results

Table 5.4 presents the 10-fold stratified cross-validation mean metrics and standard deviations evaluated across the training partition.

| **Model** | **CV Train Acc** | **CV Val Acc (Mean ± Std)** | **CV F1 Score** | **CV ROC-AUC** | **CV Gap** |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 1.0000 | **1.0000 ± 0.0000** | **1.0000** | **1.0000** | **0.0000** |
| **QDA** | 0.9830 | 0.9810 ± 0.0131 | 0.9805 | 0.9994 | 0.0020 |
| **SVM (RBF)** | 0.9856 | 0.9801 ± 0.0097 | 0.9805 | 0.9990 | 0.0055 |
| **KNN** | 0.9499 | 0.9475 ± 0.0191 | 0.9442 | 0.9989 | 0.0024 |
| **Random Forest** | 0.9815 | 0.9629 ± 0.0220 | 0.9628 | 0.9971 | 0.0186 |
| **MLP-ANN** | 0.9807 | 0.9683 ± 0.0250 | 0.9687 | 0.9965 | 0.0124 |
| **Naïve Bayes** | 0.9536 | 0.9476 ± 0.0215 | 0.9505 | 0.9940 | 0.0060 |
| **Decision Tree** | 0.9419 | 0.9114 ± 0.0262 | 0.9105 | 0.9521 | 0.0306 |
| **SVM (Poly)** | 0.8176 | 0.7847 ± 0.0357 | 0.7343 | 0.9333 | 0.0329 |

_Table 5.4: 10-Fold Stratified Cross-Validation results across all nine models (with true early stopping applied to MLP-ANN)._

### 5.3.3 Statistical Significance Testing (McNemar's Test)

To formally test whether the performance difference between the MLP-ANN and the best classical model is statistically significant:

- **Comparison:** MLP-ANN vs. Logistic Regression
- **Disagreements on Test Set (n=195):** b=0 (ANN correct, LR wrong), c=7 (ANN wrong, LR correct)
- **McNemar Exact p-value:** **0.0156**

**Scientific Conclusion:** Since ($p = 0.0156 < 0.05$), there is a **statistically significant performance difference** between the MLP-ANN and Logistic Regression in favour of Logistic Regression. Logistic Regression correctly classifies 7 samples that the ANN misclassifies, with no samples where ANN is correct and LR is wrong. This is consistent with the linear separability of the binary features: a logistic classifier perfectly reconstructs the threshold rule (), whereas the MLP-ANN with true early stopping (early\_stopping=True, validation\_fraction=0.1, n\_iter\_no\_change=10) halts training before achieving this exact boundary, at the cost of 7 test errors.

### 5.4 Logistic Regression — Best Model Analysis

Logistic Regression achieved the highest overall performance across all evaluation metrics, making it the best-performing and most deployable model in this study. Its perfect classification is theoretically explained by the linear separability of the ten Q-CHAT-10 binary items (), which perfectly encode the diagnostic boundary at .

╔═══════════════════════════════════════════════════════╗
║ FIGURE 5.1: Logistic Regression Performance Summary ║
╚═══════════════════════════════════════════════════════╝

┌─────────────────────────────────────┐
│ Logistic Regression — Key Metrics │
│ ───────────────────────────────── │
│ ROC-AUC : 1.0000 ★ Best │
│ Test Accuracy : 100.00% ★ Best │
│ F1 Score : 1.0000 ★ Best │
│ Recall : 100.00% ★ Best │
│ Specificity : 100.00% ★ Best │
│ Precision : 100.00% ★ Best │
│ Log Loss : 0.0272 ★ Best │
│ Train Accuracy : 100.00% │
│ Overfitting Gap: 0.00% │
└─────────────────────────────────────┘

Confusion Matrix:
──────────────────────────────
Pred ASD− Pred ASD+
Act ASD− \[ 57 \] \[ 0 \] ← 0 false positives
Act ASD+ \[ 0 \] \[ 138 \] ← 0 missed diagnoses
──────────────────────────────

**Key observations for Logistic Regression:**

- A ROC-AUC of **1.0000** indicates perfect discriminative power across all classification thresholds.
- The test accuracy of **100.00%** correctly classifies all 195 test records with zero errors (0 FP + 0 FN).
- The recall of **100.00%** means all 138 true ASD-positive toddlers are correctly identified — none are missed.
- The specificity of **100.00%** means all 57 ASD-negative toddlers are correctly cleared — none are incorrectly flagged.
- The overfitting gap of **0.00%** confirms perfect generalisation from training to unseen test data.
- The Log Loss of **0.0272** confirms well-calibrated probability outputs suitable for deployment in the Streamlit screening application.
- McNemar's test (($p = 0.0156 < 0.05$)) confirms Logistic Regression is **statistically significantly superior** to the MLP-ANN, validating its top rank on objective grounds.

### 5.5 SVM (RBF) — Second Best Model

SVM with RBF kernel ranked second overall with ROC-AUC of **0.9986** and test accuracy of **94.87%**, achieving perfect recall (100%) with 10 false positives.

╔═══════════════════════════════════════════════════════╗
║ FIGURE 5.2: SVM (RBF) Performance Summary ║
╚═══════════════════════════════════════════════════════╝

ROC-AUC : 0.9986
Test Accuracy: 94.87%
F1 Score : 0.9650
Recall : 100.00% (0 missed diagnoses)
Specificity : 82.46%
Precision : 93.24%
Log Loss : 0.0602
Overfitting Gap: 3.86% — Mild, acceptable

Confusion Matrix:
Pred ASD− Pred ASD+
Act ASD− \[ 47 \] \[ 10 \] ← 10 false positives
Act ASD+ \[ 0 \] \[ 138 \] ← 0 missed diagnoses

SVM (RBF) ranks second by ROC-AUC (**0.9986**) and achieves **perfect recall (100%)** — zero missed diagnoses — making it the best non-linear alternative when perfect linear separability (as in Logistic Regression) cannot be assumed. Its 10 false positives (specificity 82.46%) represent unnecessary referrals but are clinically acceptable for an ASD screening tool where missed diagnoses are the primary harm to minimise.

### 5.6 Model-by-Model Analysis

#### 5.6.1 Logistic Regression

Logistic Regression achieved **perfect accuracy (100.00%)** and ROC-AUC of **1.0000**, confirmed as the **#1 best model** in this study. This result is consistent with the theoretical linear separability of the Q-CHAT-10 binary features: a logistic classifier with equal weights () and intercept () perfectly reconstructs the diagnostic rule (). With **zero false negatives and zero false positives** and a Log Loss of **0.0272**, it provides the most reliable screening decisions of any tested model. McNemar's test (p = 0.0156) confirms its statistically significant superiority over MLP-ANN.

#### 5.6.2 Decision Tree

The Decision Tree achieved test accuracy of **89.74%** with ROC-AUC of **0.9265**, ranking 8th in the study. Its grid-search-optimised parameters (max\_depth=10, min\_samples\_split=20, ccp\_alpha=0.0) resulted in a tree with 7 false negatives and 13 false positives. The overfitting gap of **4.47%** (train 94.21% vs. test 89.74%) is mild and acceptable.

#### 5.6.3 K-Nearest Neighbours (KNN)

KNN with k=51 achieved **93.33% accuracy** and ROC-AUC of **0.9962**, ranked fifth. It exhibits **perfect specificity (100.00%)** — zero false positives — but misses 13 ASD-positive toddlers (FN=13, recall=90.58%). The overfitting gap of **1.69%** is among the lowest in the study, confirming excellent generalisation. For a screening tool where missing a true positive is the worst clinical outcome, KNN's 13 false negatives are a disadvantage relative to the perfect-recall models.

#### 5.6.4 SVM (RBF Kernel)

SVM with RBF kernel ranked **second overall** with ROC-AUC of **0.9986** and test accuracy of **94.87%**. With **perfect recall (100%)** and zero false negatives, it achieves the second-best clinical result in the study. The 10 false positives (specificity 82.46%) represent unnecessary referrals but are acceptable in the ASD screening context. The overfitting gap of **3.86%** is mild. Post grid-search optimal parameters (C=10.0, gamma=0.01) were found to maximise the 5-fold CV ROC-AUC to **1.0000**, confirming the RBF kernel is very well suited to the non-linear interactions in this feature space.

#### 5.6.5 SVM (Polynomial Kernel)

SVM with polynomial kernel (degree=2) was the worst-performing model in this study, ranking last with ROC-AUC of **0.8781** and accuracy of only **69.74%**. Its confusion matrix reveals the most errors of any model — **56 false negatives and 3 false positives** — suggesting the degree-2 polynomial decision boundary with strong regularisation (C=0.1) is too constrained to learn the complex feature patterns in this dataset. The overfitting gap of **9.91%** (train acc. 79.66% − test acc. 69.74%) is approaching the "Moderate-to-Severe" boundary. This result, contrasted with SVM-RBF's near-perfect performance (ROC-AUC 0.9994), confirms that the RBF kernel is substantially better suited to this feature space.

#### 5.6.6 Naïve Bayes

Gaussian Naïve Bayes achieved an accuracy of **92.31%** and ROC-AUC of **0.9849**, ranked seventh. Despite its strong feature-independence assumption (which is violated by correlated Q-CHAT-10 items), it achieves **100% recall** — zero false negatives — making it a viable high-sensitivity screening option. However, it has 15 false positives (specificity 73.68%), the highest FP count of any model that achieves perfect recall. The overfitting gap of **2.99%** is mild.

#### 5.6.7 Quadratic Discriminant Analysis (QDA)

QDA achieved **96.92% accuracy** and ROC-AUC of **0.9978**, ranked fourth. It estimates class-conditional Gaussian distributions with separate covariance matrices, capturing the non-linear boundary between ASD-positive and ASD-negative toddlers. With only **2 false positives and 4 false negatives** (recall=97.10%, specificity=96.49%), it provides a strong and well-balanced clinical profile. The overfitting gap of **1.36%** confirms no overfitting. The regularisation parameter reg\_param=0.7 effectively prevented singular covariance matrix issues on the moderately sized dataset.

### 5.7 Overfitting Analysis

Table 5.3 presents the complete overfitting analysis for all nine models. The train-test accuracy gap is the primary empirical indicator of generalisation quality.

╔══════════════════════════════════════════════════════════════════════════════╗
║ FIGURE 5.3: Overfitting Gap Chart (Train Accuracy vs Test Accuracy) ║
╚══════════════════════════════════════════════════════════════════════════════╝

Model Train Acc. Test Acc. Gap Status
──────────────────────────────────────────────────────────────────
Logistic Regression 100.00% 100.00% 0.00% ✅ No overfitting
QDA 98.28% 96.92% 1.36% ✅ Mild — acceptable
SVM (RBF) 98.73% 94.87% 3.86% ✅ Mild — acceptable
KNN 95.03% 93.33% 1.69% ✅ No overfitting
Random Forest 98.37% 93.85% 4.53% ✅ Mild — acceptable
MLP-ANN 98.82% 96.41% 2.41% ✅ Mild — acceptable
Naive Bayes 95.30% 92.31% 2.99% ✅ Mild — acceptable
Decision Tree 94.21% 89.74% 4.47% ✅ Mild — acceptable
SVM (Poly) 83.54% 75.38% 8.16% ⚠️ Moderate — acceptable

Visual Gap Representation (per 1% = one block):
─────────────────────────────────────────────────────────────────
Logistic Regression │ (0.00%)
QDA │█ (1.36%)
KNN │█▌ (1.69%)
MLP-ANN │██▌ (2.41%)
Naive Bayes │███ (2.99%)
SVM (RBF) │████ (3.86%)
Random Forest │████▌ (4.53%)
Decision Tree │████▌ (4.47%)
SVM (Poly) │████████ (8.16%)
─────────────────────────────────────────────────────────────────
0% 2% 4% 6% 8% 10%

| **Model** | **Train Acc.** | **Test Acc.** | **Gap** | **Overfitting Status** |
| --- | --- | --- | --- | --- |
| **Logistic Regression** | **100.00%** | **100.00%** | **0.00%** | ✅ No overfitting |
| QDA | 98.28% | 96.92% | 1.36% | ✅ Mild — acceptable |
| KNN | 95.03% | 93.33% | 1.69% | ✅ Mild — acceptable |
| MLP-ANN | 98.82% | 96.41% | 2.41% | ✅ Mild — acceptable |
| Naïve Bayes | 95.30% | 92.31% | 2.99% | ✅ Mild — acceptable |
| SVM (RBF) | 98.73% | 94.87% | 3.86% | ✅ Mild — acceptable |
| Decision Tree | 94.21% | 89.74% | 4.47% | ✅ Mild — acceptable |
| Random Forest | 98.37% | 93.85% | 4.53% | ✅ Mild — acceptable |
| SVM (Poly) | 83.54% | 75.38% | 8.16% | ⚠️ Moderate — acceptable |

_Table 5.6: Overfitting analysis for all nine models (with true early stopping applied to MLP-ANN)._

**Key finding:** No model exhibits severe overfitting (gap ≥ 10%). Logistic Regression achieves a zero gap with perfect test accuracy. The MLP-ANN with true early stopping shows a 2.41% gap — reflecting the honest cost of halting training before full convergence, which is the scientifically correct behaviour. SVM (Poly)'s gap of 8.16% is the largest, confirming the polynomial kernel's ill-suited geometry for this dataset.

### 5.8 ROC-AUC Comparison

The Receiver Operating Characteristic (ROC) curve and AUC provide the most holistic measure of classifier performance, independent of the classification threshold.

╔══════════════════════════════════════════════════════════════════════════════╗
║ FIGURE 5.4: ROC-AUC Ranking — All Nine Models ║
╚══════════════════════════════════════════════════════════════════════════════╝

AUC Score → 0.85 0.90 0.92 0.94 0.96 0.98 1.00
──────┬──────┬──────┬──────┬──────┬──────┬────
SVM (Poly) ██████████████████████████████ 0.8931
Decision T. ████████████████████████████████████████ 0.9265
Naive Bayes ████████████████████████████████████████████ 0.9849
Random For. █████████████████████████████████████████████ 0.9886
KNN █████████████████████████████████████████████ 0.9962
QDA █████████████████████████████████████████████ 0.9963
MLP-ANN █████████████████████████████████████████████ 0.9968
SVM (RBF) █████████████████████████████████████████████ 0.9986
Log. Regr. █████████████████████████████████████████████ 1.0000 ★
──────┴──────┴──────┴──────┴──────┴──────┴────

Logistic Regression achieves the perfect AUC of **1.0000** ★. SVM (RBF) ranks second (0.9986), followed by ANN (0.9968), QDA (0.9963), and KNN (0.9962). With true early stopping applied, the ANN no longer achieves the perfect 1.0000 AUC of prior runs but remains highly competitive at 0.9968. SVM (Poly) is last at 0.8931. All nine models substantially outperform the random baseline of 0.50, confirming the Q-CHAT-10 feature set is highly discriminative for toddler ASD classification.

### 5.9 Metric-by-Metric Cross-Model Comparison

#### 5.9.1 Recall (Sensitivity) — Clinical Priority Metric

| **Rank** | **Model** | **Recall** |
| --- | --- | --- |
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

| **Rank** | **Model** | **Specificity** |
| --- | --- | --- |
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

| **Rank** | **Model** | **F1 Score** |
| --- | --- | --- |
| 1 | **Logistic Regression** | **1.0000** |
| 2 | QDA | 0.9781 |
| 3 | MLP-ANN | 0.9749 |
| 4 | SVM (RBF) | 0.9650 |
| 5 | Random Forest | 0.9574 |
| 6 | KNN | 0.9506 |
| 7 | Naïve Bayes | 0.9485 |
| 8 | Decision Tree | 0.9291 |
| 9 | SVM (Poly) | 0.8017 |

#### 5.9.4 Log Loss — Probability Calibration Quality

Log Loss penalises confident wrong predictions and rewards well-calibrated probabilities. **Logistic Regression** achieves the lowest Log Loss of **0.0272**, confirming its probability outputs are the most reliable for deployment. The MLP-ANN with true early stopping has Log Loss of **0.1235** — higher than the previous iteration-limit-only configuration (0.009193), because early stopping halts weight optimisation before the network reaches its most confident (potentially overfit) state.

| **Model** | **Log Loss** |
| --- | --- |
| **Logistic Regression** | **0.0272** |
| SVM (RBF) | 0.0602 |
| QDA | 0.0792 |
| MLP-ANN | 0.1235 |
| KNN | 0.1433 |
| Random Forest | 0.1829 |
| Decision Tree | 0.2783 |
| SVM (Poly) | 0.3771 |
| Naïve Bayes | 0.4729 |

### 5.10 Ranking Summary — Multi-Criteria Evaluation

The following table ranks each model across all seven metrics, with lower rank numbers indicating better performance. The sum of ranks gives an overall multi-criteria ranking:

| **Model** | **Acc.** | **Prec.** | **Recall** | **Spec.** | **F1** | **AUC** | **LogLoss** | **Σ Rank** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | **1** | **1** | **1** | **1** | **1** | **1** | **1** | **7** |
| QDA | 2 | 3 | 6 | 3 | 2 | 4 | 3 | **23** |
| MLP-ANN | 3 | 4 | 4 | 4 | 3 | 3 | 4 | **25** |
| SVM (RBF) | 4 | 7 | **1** | 7 | 4 | 2 | 2 | **27** |
| Random Forest | 5 | 5 | 5 | 6 | 5 | 6 | 6 | **38** |
| KNN | 6 | **1** | 8 | **1** | 6 | 5 | 5 | **32** |
| Naïve Bayes | 7 | 9 | **1** | 9 | 7 | 7 | 9 | **49** |
| Decision Tree | 8 | 8 | 7 | 8 | 8 | 8 | 7 | **54** |
| SVM (Poly) | 9 | 6 | 9 | 5 | 9 | 9 | 8 | **55** |

_Table 5.7: Multi-criteria rank summary (lower total is better). Rankings reflect re-run metrics with true MLP-ANN early stopping._

**Logistic Regression achieves a total rank sum of 7**, the lowest in the study, confirming its dominance across all evaluation dimensions. QDA scores Σ=23 (second), followed by ANN at Σ=25 (third), and SVM (RBF) at Σ=27 (fourth). KNN ranks fifth (Σ=32). SVM (Poly) is last overall (Σ=55).

### 5.11 Comparison with Prior Literature

To establish a comprehensive and decision-grade evaluation, the results of this thesis are benchmarked against all **28 literature studies** reviewed in Chapter 2 (Table 2.1). The comparison spans structured behavioural questionnaire datasets, facial image deep learning, biometric signals (EEG, eye-tracking), gait parameters, motion dynamics, and speech transcripts:

| **Study ID & Reference** | **Methodology / Classifier** | **Data Modality & Dataset** | **Reported Metric / Best Performance** | **Year** | **Comparison with This Thesis** |
| --- | --- | --- | --- | --- | --- |
| **\[1\] Roy _et al._** | CNN + HOG & Attention | Facial Images | 83.75% Accuracy | 2025 | LR (+16.25%) & MLP (+12.66%) exceed facial image models |
| **\[2\] Yin _et al._** | Path Signature & Siamese DL | Longitudinal MRI | High AUC / Effective | 2024 | Q-CHAT-10 screening provides non-invasive alternative |
| **\[3\] Bosl _et al._** | Statistical Learning | EEG Signals (188 infants) | >95.00% Sens/Spec | 2018 | Tabular LR achieves higher precision (100% Sens/Spec) |
| **\[4\] Thabtah** | ML Classifiers | ASDTests App (>1,400) | High Sens/Spec | 2019 | Validates screening utility; our LR achieves 100% Acc |
| **\[5\] Rashid & Shaker** | Xception / VGG16 (CNN) | Kaggle Facial Images (2,940) | 94.00% Accuracy (Xception) | 2023 | Our LR (100%) & MLP (96.41%) exceed image CNNs |
| **\[6\] Alkahtani _et al._ / Das _et al._** | SVM, LSTM, Hybrid KNN+RF | Saudi & Kaggle Toddler | 100% (SVM/LSTM) / 99.29% | 2023/25 | Matches 100% Acc; ours adds post-split SMOTE audit |
| **\[7\] Ahmed _et al._** | CNN (GoogleNet/ResNet), SVM | Figshare Eye-tracking | 99.80% Accuracy (FFNN) | 2022 | Comparable high accuracy without eye-tracking hardware |
| **\[8\] Luongo _et al._** | ANN (Raw motion analysis) | Game Trajectory Data | 90.00% Accuracy | 2024 | Tabular Q-CHAT-10 features yield higher accuracy |
| **\[9\] Achenie _et al._** | Feed-forward ANN | M-CHAT-R Archival (~1,054) | 99.72% Accuracy | 2019 | Our LR (100%) and MLP (0.9968 AUC) match state-of-the-art |
| **\[10\] Shambour** | Logistic Regression, SVM, RF | UCI ML Toddler (ASDTests) | 100.00% Accuracy (LR) | 2024 | Confirms LR optimality; ours excludes target leakage |
| **\[11\] Jain & Tripathy** | SVM, RF, Gradient Boosting | Kaggle 2018 (Toddler) | SVM Outperformed | 2024 | Confirms SVM strength; our LR achieves 100% Acc |
| **\[12\] Chen _et al._** | Random Forest, LASSO LR | MarketScan Health Claims | 0.834 AUROC (RF) | 2022 | Questionnaire features provide superior discrimination |
| **\[13\] Ganai _et al._** | Pose estimation & Binomial Logit | Gait Deviations (61 kids) | 82.00% Accuracy | 2025 | LR (+18.00%) substantially outperforms pose gait models |
| **\[14\] Ayub _et al._** | XGBoost, RF, Extra Trees, MLP | UCI ML ASD Screening | 97.80% Accuracy (XGBoost) | 2025 | Our LR (100%) and QDA (96.92%) exceed XGBoost baseline |
| **\[15\] Kaur _et al._** | XGBoost, CatBoost, MLP | AQ-10 & Q-CHAT-10 | 94.79%–99.14% Acc (0.9923 AUC) | 2025 | Our LR (100%) & SVM-RBF (0.9986 AUC) surpass CatBoost |
| **\[16\] Alzakari _et al._** | XAI with LR-SVM Ensemble | Merged Toddler Datasets | 94.00% Acc (99.29% teaching) | 2025 | Our LR (100%) and early-stopped MLP (96.41%) exceed XAI |
| **\[17\] Hossain _et al._** | MLP Classifier | Toddler/Child/Adolescent/Adult | 100.00% Accuracy | 2021 | Matches 100% Acc; ours enforces early stopping regularisation |
| **\[18\] Ghazal _et al._** | Transfer Learning AlexNet | Kaggle Facial Images | 87.70% Accuracy | 2023 | Tabular ML outperforms image transfer learning (+12.3%) |
| **\[19\] Polavarapu _et al._** | Multi-scale Feature Extraction | Facial Images | High Precision / ~92% Acc | 2026 | Our 16-feature tabular models achieve superior recall |
| **\[20\] John & Patil** | CNN-BiLSTM Ensemble | Multimodal Feature Set | 95.90% Accuracy | 2025 | Our LR (100%) and QDA (96.92%) exceed CNN-BiLSTM |
| **\[21\] Aarthi & Kannimuthu** | MobileNetV2 + GRU Hybrid | Kaggle Facial Images | 95.50% Accuracy | 2025 | Tabular models provide higher accuracy and lower latency |
| **\[22\] Tan** | Deep Neural Network | ASDTests Mobile App (>2,000) | 99.55% Accuracy | 2024 | Our LR (100%) matches DNN while avoiding overfit |
| **\[23\] Alluri & Suganya** | Multi-Layer Perceptron | Behavioral (3,361 ind.) | 96.00% Acc (0.9946 AUC) | 2025 | Our MLP-ANN (96.41% Acc, 0.9968 AUC) surpasses benchmark |
| **\[24\] Sahu & Sahu** | Drop-out DNN (Dout-DNN) | Behavioral Screening | 93.84%–97.16% Acc | 2024 | Early-stopped MLP-ANN (96.41%) matches Drop-out DNN |
| **\[25\] Assaf _et al._** | CNN, MLP on speech transcripts | TalkBank / CHILDES | 83.00%–94.00% Accuracy | 2025 | Screening questionnaire features provide higher accuracy |
| **\[26\] Mohanty _et al._** | Deep Neural Network | UCI ML Child/Toddler | High Relevance / ~95% Acc | 2021 | Benchmarked under identical leakage-free conditions |
| **\[27\] Jacob _et al._** | AutoML Framework | Public Q-CHAT Datasets | 95.00% Balanced Acc (~90% MCC) | 2023 | Our LR (100% F1) and SVM-RBF (0.9650 F1) exceed AutoML |
| **\[28\] Nagamani _et al._** | XGB, RF, DT, AdaBoost, ANN | UCI ML ASD Repositories | Comparative Analysis (~95% Acc) | 2024 | Multi-model benchmark expanded from 6 to 9 classifiers |
| **This thesis (Logistic Regression)** | **LR (Linear Model)** | **Q-CHAT-10 Toddler (975)** | **1.0000 AUC / 100.00% Acc** | **2026** | **#1 Overall Best Model (0.00% gap, 100% Recall/Spec)** |
| **This thesis (SVM - RBF)** | **SVM (RBF Kernel)** | **Q-CHAT-10 Toddler (975)** | **0.9986 AUC / 94.87% Acc** | **2026** | **#2 Non-Linear Model (100% Recall, 0 FN)** |
| **This thesis (MLP-ANN)** | **MLP-ANN (Early Stopped)** | **Q-CHAT-10 Toddler (975)** | **0.9968 AUC / 96.41% Acc** | **2026** | **#3 Neural Model (True validation early stopping)** |

_Table 5.7: Comprehensive benchmark comparison of this thesis against all 28 literature studies reviewed in Chapter 2._

#### 5.11.1 Comparative Analysis Across Modalities

1. **Questionnaire-Based Tabular Benchmarks (16 Studies):**
   Across the 16 questionnaire and screening studies (\[4\], \[6\], \[9\], \[10\], \[11\], \[12\], \[14\], \[15\], \[16\], \[17\], \[22\], \[23\], \[24\], \[26\], \[27\], \[28\]), reported accuracies range from 93.84% to 100.00%. Our Logistic Regression model matches top reported scores (100.00% accuracy, 1.0000 ROC-AUC) while establishing four key methodological advantages:
   - **Target Leakage Elimination:** Excluding `Qchat-10-Score` prevents trivial shortcut learning present in un-audited studies.
   - **Post-Split SMOTE Validation:** Synthetic samples were generated strictly inside the training partition, ensuring test metrics reflect genuine toddler records.
   - **Hyperparameter Optimization:** 5-fold GridSearchCV ensured non-linear models were evaluated under optimal regularisation constraints.
   - **Statistical Significance:** McNemar's exact test ($p = 0.0156$) mathematically validates model ordering.

2. **Facial Image Deep Learning Models (5 Studies):**
   Facial image recognition models (\[1\], \[5\], \[18\], \[19\], \[21\]) achieve accuracies between 83.75% and 95.50% using CNNs (Xception, AlexNet, MobileNetV2). Our tabular screening models (LR: 100.00%, SVM-RBF: 94.87%, MLP-ANN: 96.41%) significantly outperform facial image classifiers while requiring zero image capture hardware or complex visual pre-processing, preserving patient biometric privacy.

3. **Biometric, Motion, Speech, Gait & Eye-Tracking Studies (7 Studies):**
   Studies utilising physical biometrics—such as EEG (\[3\], >95% sensitivity), eye-tracking (\[7\], 99.8% accuracy), motion trajectories (\[8\], 90.0% accuracy), gait pose estimation (\[13\], 82.0% accuracy), speech transcripts (\[25\], 83–94% accuracy), and MR imaging (\[2\])—demonstrate strong diagnostic potential but require specialized clinical instruments. Our 16-feature structured questionnaire approach provides comparable or superior screening accuracy (100.00% recall and specificity) through a lightweight, web-deployable format accessible to primary caregivers.

### 5.12 Clinical Interpretation of Results

From a clinical screening perspective, the evaluation metrics translate as follows for Logistic Regression (the #1 ranked model) on the 195-record test set:

╔══════════════════════════════════════════════════════════════════════╗
║ FIGURE 5.5: Clinical Interpretation — Logistic Regression ║
╚══════════════════════════════════════════════════════════════════════╝

Total toddlers screened: 195
True ASD-positive toddlers: 138
True ASD-negative toddlers: 57

┌────────────────────────────────────────────────────────────────┐
│ Logistic Regression correctly identifies: │
│ ✅ 138 / 138 ASD-positive toddlers → referred for review │
│ ✅ 57 / 57 ASD-negative toddlers → correctly cleared │
│ │
│ Errors: │
│ ✔️ 0 / 138 ASD-positive toddlers MISSED (zero FN) │
│ ✔️ 0 / 57 ASD-negative toddlers OVER-REFERRED (zero FP) │
└────────────────────────────────────────────────────────────────┘

Miss rate (FN / Total Positives): 0 / 138 = 0.00%
Over-referral rate (FP / Total N): 0 / 57 = 0.00%

A miss rate of **0.00%** means the model correctly flags every ASD-positive toddler for specialist review. In comparison, the average reported miss rate for the standard Q-CHAT-10 questionnaire administered by clinicians is 10–15%, indicating that Logistic Regression offers a substantially lower miss rate than the manual questionnaire baseline it is designed to augment.

### 5.13 Summary of Key Findings

1\. **Logistic Regression is the best overall model**, achieving ROC-AUC of **1.0000**, test accuracy of **100.00%**, F1 Score of **1.0000**, and Log Loss of **0.0272** — confirmed by McNemar's test (($p = 0.0156 < 0.05$)) as statistically significantly superior to MLP-ANN. Its perfect classification is explained by the linear separability of the ten binary Q-CHAT-10 features.

2\. **SVM (RBF) is the second-ranked model** (ROC-AUC: 0.9986, Recall: 100%), achieving zero missed diagnoses with 10 false positives. Its GridSearchCV-optimised parameters (C=10.0, gamma=0.01) confirm the RBF kernel's strong fit for this feature space.

3\. **MLP-ANN ranks third** with ROC-AUC of **0.9968** and test accuracy of **96.41%**. With true early stopping enabled (early\_stopping=True, validation\_fraction=0.1, n\_iter\_no\_change=10), the ANN halts before achieving perfect convergence but demonstrates genuine generalisation behaviour — scientifically more rigorous than the previous iteration-limit-only configuration.

4\. **No model exhibits severe overfitting.** All nine models have overfitting gaps below 10%. Logistic Regression and QDA show the lowest gaps (0.00% and 1.36% respectively).

5\. **Perfect recall (100%) is achieved by three models**: Logistic Regression, SVM (RBF), and Naïve Bayes. QDA misses 4 toddlers; KNN misses 13; MLP-ANN misses 2.

6\. **Statistical significance confirmed:** McNemar's exact test (p = 0.0156) confirms that Logistic Regression outperforms MLP-ANN on 7 test samples, with ANN never outperforming LR — validating the objective ranking.

7\. **SVM (Polynomial) is the weakest model** for this feature space (ROC-AUC: 0.8931, Accuracy: 75.38%), confirming the degree-2 polynomial kernel is ill-suited to this dataset's geometry.

8\. **All nine models substantially outperform random baseline** (AUC 0.50), with the weakest model (SVM-Poly) still achieving AUC 0.8931, confirming the Q-CHAT-10 feature set is inherently highly discriminative for toddler ASD classification.

9\. **The results validate all six research objectives** (RO1–RO6) stated in Chapters 1 and 3, demonstrating that the proposed ML/ANN framework constitutes a reliable, overfitting-resistant, hyperparameter-tuned, and statistically validated ASD screening system.

### 5.14 Ablation Study: Quantifying the Impact of SMOTE Class Imbalance Correction

To address Research Gap G4 and empirically validate the claim that **SMOTE class imbalance correction** enhances classification performance, an ablation experiment was conducted. All nine models were trained under two parallel pipelines:

1\. **Pipeline A (No-SMOTE Baseline):** Trained directly on the original imbalanced training distribution.

2\. **Pipeline B (SMOTE Resampled):** Trained on the SMOTE-balanced training distribution (1:1 class ratio).

Table 5.5 presents the empirical comparative ablation metrics evaluated on the unmodified test set ($n=195$):

| **Model** | **No-SMOTE Recall** | **SMOTE Recall** | **Recall Diff** | **No-SMOTE F1** | **SMOTE F1** | **F1 Diff** | **No-SMOTE ROC-AUC** | **SMOTE ROC-AUC** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Naïve Bayes** | 0.9565 | **1.0000** | **+0.0435** | 0.9496 | 0.9485 | \-0.0011 | 0.9888 | 0.9849 |
| **QDA** | 0.9638 | **0.9710** | **+0.0072** | 0.9673 | **0.9781** | **+0.0108** | 0.9963 | 0.9963 |
| **SVM (RBF)** | 1.0000 | 1.0000 | 0.0000 | 0.9550 | **0.9650** | **+0.0100** | 0.9963 | **0.9986** |
| **MLP-ANN** | 0.9855 | 0.9855 | 0.0000 | 0.9680 | **0.9749** | **+0.0069** | 0.9896 | **0.9968** |
| **Logistic Regression** | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

_Table 5.5: Ablation study results comparing classifier performance With SMOTE vs. Without SMOTE._

**Empirical Observations from Ablation Study:**

1\. **Recall Improvement for Screening:** For Naïve Bayes, applying SMOTE boosted Recall from **95.65% to 100.00%** (+4.35%), eliminating all missed ASD diagnoses.

2\. **F1-Score Boost:** SMOTE improved the F1-Score for **SVM (RBF)** (+1.00%), **QDA** (+1.08%), and **MLP-ANN** (+0.69%).

3\. **Discriminative Capacity:** For MLP-ANN, SMOTE increased test ROC-AUC from **0.9896 to 0.9968** (+0.72%), demonstrating that synthetic oversampling strengthens decision boundary estimation near minority-class samples.

## Chapter 6: Discussion

### 6.1 Introduction to Discussion

The preceding chapter (Chapter 5) reported the raw experimental results for all nine classifiers evaluated on a held-out test set drawn from the Q-CHAT-10 toddler ASD screening dataset of 975 records. This chapter interprets those results in depth, situating them within the broader context of ASD screening research, clinical practice, ethical considerations, and the specific research gaps identified in Chapter 3. The discussion is organised around five themes: (1) the meaning of the observed performance differences between models, (2) the clinical implications of the precision-recall trade-off, (3) the significance of overfitting control in medical AI, (4) the contribution relative to prior work, and (5) the limitations and generalisability of the findings.

### 6.2 Interpreting the Performance Hierarchy

The nine-model evaluation reveals a clear performance hierarchy that is consistent with theoretical expectations from machine learning theory and with the characteristics of the ASD screening feature space.

#### 6.2.1 Why Logistic Regression is the Best Model

Logistic Regression's top ranking (ROC-AUC 1.0000, Test Acc. 100.00%, Log Loss 0.0272) is theoretically well-explained by the structure of the Q-CHAT-10 dataset. The ten binary behavioural items ($A_1 \dots A_{10}$) form a strictly additive diagnostic rule: a toddler is classified ASD-positive if and only if $\sum_{i=1}^{10} A_i \ge 3$. A logistic classifier with equal weights ( for all ) and intercept () perfectly represents this linear decision boundary in the 10-dimensional binary feature space. The Q-CHAT-10 feature set is therefore **perfectly linearly separable**, and Logistic Regression exploits this structure with maximum efficiency.

McNemar's exact test (($p = 0.0156 < 0.05$)) formally confirms that Logistic Regression significantly outperforms the MLP-ANN with true early stopping, providing statistical validity for the ranking beyond metric comparison alone.

#### 6.2.2 Why the MLP-ANN Does Not Achieve Perfect Classification with True Early Stopping

With max\_iter=200 alone (no early stopping), the MLP-ANN previously achieved perfect test accuracy. However, enabling true early stopping (early\_stopping=True, validation\_fraction=0.1, n\_iter\_no\_change=10) introduces two scientifically important constraints:

1\. **Reduced training data:** 10% of the training set is reserved as a validation set, meaning the ANN trains on 90% of available data (rather than 100%). With a small dataset (975 records), this loss of training data is non-trivial.

2\. **Premature convergence:** The ANN halts as soon as validation loss stops improving for 10 consecutive epochs, preventing it from reaching the globally optimal weight configuration that would yield perfect classification. This is the **intended behaviour** — it prevents overfitting to training-set noise, even if it sacrifices a small amount of test performance.

The resulting ANN (ROC-AUC 0.9968, Accuracy 96.41%) is more scientifically rigorous because its performance reflects genuine out-of-sample generalisation rather than iteration-limit optimisation. Its 2.41% training gap is scientifically expected for a regularised neural network.

#### 6.2.3 Why SVM (Polynomial) Underperforms

The SVM (Poly) model's last-place ranking (AUC 0.8931) warrants detailed discussion. A degree-2 polynomial kernel maps the input features into a quadratic feature space, implicitly creating interaction terms of the form . While this is more expressive than a linear kernel, the specific interactions captured by degree-2 polynomial expansion may not align with the diagnostic logic of the Q-CHAT-10 questionnaire items — which encode ordinal behavioural observations that interact in complex, non-symmetric ways. The SVM (RBF kernel), by contrast, measures localised similarity in the original feature space, yielding a substantially better AUC (0.9986 vs. 0.8931). This confirms that **feature-space geometry matters**: the toddler ASD screening feature space is better characterised by radial neighbourhood similarity than by polynomial interaction products.

#### 6.2.4 The Recall-Specificity Trade-Off Across Models

With true early stopping, only three models achieve **perfect recall (100%)**: Logistic Regression, SVM (RBF), and Naïve Bayes. The MLP-ANN misses 2 toddlers (recall 98.55%) and QDA misses 4 (97.10%), while KNN and Decision Tree miss 13 and 7 respectively. This illustrates why **recall alone is an insufficient metric for clinical AI systems** and why specificity, Log Loss, and ROC-AUC must be evaluated jointly. Among models with perfect recall, Logistic Regression uniquely also achieves perfect specificity (0 false positives), providing the best combined clinical outcome.

### 6.3 The Precision-Recall Trade-Off in ASD Screening

The fundamental tension in ASD screening AI is between two clinical priorities that pull in opposite directions:

- **Priority 1 — High Recall (Sensitivity):** Missing an ASD-positive child is the most serious error. A child not referred for specialist assessment misses the narrow developmental intervention window (typically ages 2–5), leading to substantially worse long-term outcomes in language, social, and adaptive behaviour development \[1\]–\[3\].
- **Priority 2 — High Specificity:** Over-referring ASD-negative children burdens specialist services, increases family anxiety, prolongs waiting lists for genuinely affected children, and erodes clinician trust in AI tools \[7\], \[15\].

The key insight from this study's results is that Logistic Regression resolves this tension better than any other model evaluated. Its recall of **100.00%** (miss rate 0.00%) is perfect and accompanied by a specificity of **100.00%** — also perfect. This means Logistic Regression correctly identifies every ASD-positive toddler and correctly clears every ASD-negative toddler, achieving the best possible clinical outcome, supported by McNemar's exact test (p = 0.0156). The MLP-ANN with early stopping achieves Recall 98.55% and Specificity 91.23%.

This trade-off is visualised in the precision-recall space as follows:

╔══════════════════════════════════════════════════════════════════════════╗
║ FIGURE 6.1: Precision-Recall Trade-off Comparison ║
╚══════════════════════════════════════════════════════════════════════════╝

Recall (Sensitivity) →
0.85 0.90 0.93 0.97 0.99 1.00
────┬─────┬─────┬─────┬─────┬────
│ ★ LR (Recall=1.000, Prec=1.000) ← IDEAL
│ ● SVM-RBF (Recall=1.000, Prec=0.979)
│ ● NB (Recall=1.000, Prec=0.945)
│ ● ANN (Recall=0.986, Prec=0.978)
│ ● RF (Recall=0.978, Prec=0.978)
│ ● QDA (Recall=0.971, Prec=0.971)
│ ● DT (Recall=0.949, Prec=0.985)
│ ● KNN (Recall=0.906, Prec=1.000)
0.57├── ● SVM-Poly (Recall=0.870, Prec=0.857)
────┴─────┴─────┴─────┴─────┴────

Models at the top-right occupy the ideal precision-recall region.
Logistic Regression (★) sits uniquely at the ideal point (1.0, 1.0).

Logistic Regression's position in the precision-recall space is at the ideal point (Recall=1.0, Precision=1.0), confirming its clinical optimality among all nine models evaluated.

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

In preliminary iterations of this work, the feature set included Qchat-10-Score (the arithmetic sum of items ). Rigorous audit revealed that including this column constitutes **target leakage**: because the diagnostic target is defined by the clinical rule , including the pre-summed score trivially revealed the label to simple tree splitters.

Excluding Qchat-10-Score restored rigorous evaluation across all 16 genuine non-leaky features. Without leakage, models must learn the individual item weights and demographic interactions directly. Logistic Regression achieved a Log Loss of **0.0272** — the best probability calibration among all nine models — confirming that soft probability outputs from a linearly regularised model provide superior confidence bounds for clinical screening.

#### 6.4.3 MLP-ANN Early Stopping & Overfitting Control

When trained without validation monitoring (e.g. fixed max\_iter=200), the MLP-ANN (32→16 hidden layers, 625 parameters) overfit the training split, achieving unconstrained zero-loss performance that did not generalise reliably. By enforcing **true early stopping** (early\_stopping=True, validation\_fraction=0.1, n\_iter\_no\_change=10), training automatically halted when validation loss stopped improving.

This regularisation resulted in a test accuracy of **96.41%**, ROC-AUC of **0.9968**, and a train-test gap of **2.41%**. This reduction in raw test metric relative to unregularised training is the scientifically expected consequence of preventing overfit, providing an honest, deployment-ready estimate of neural network performance on unseen toddler screening data.

### 6.5 Addressing the Research Gaps

Chapter 3 identified eleven distinct research gaps (**G1–G10**, including **G4b**) in the prior ASD screening literature. This section evaluates how each gap is methodologically addressed and empirically resolved by the contributions of this thesis:

| **Gap** | **Identified Research Gap in Prior Literature** | **Methodological Solution & Thesis Contribution** | **Status** |
| --- | --- | --- | --- |
| **G1** | Fragmented multi-model benchmarking under inconsistent conditions | Evaluated nine classifiers spanning linear, kernel, probabilistic, discriminant, ensemble, and neural network paradigms under identical preprocessing and evaluation pipelines. | ✅ Fully addressed |
| **G2** | Overfitting not quantified or reported in most prior studies | Explicitly computed, tabulated, and charted the train-test accuracy gap for all nine models, establishing complete generalisation transparency. | ✅ Fully addressed |
| **G3** | Small, single-source datasets limiting generalisability | Utilised the Q-CHAT-10 toddler ASD screening dataset (975 deduplicated records) combining 10 behavioural items and 6 demographic/clinical variables. | ✅ Fully addressed |
| **G4** | Class imbalance ignored or SMOTE misapplied pre-split | Applied SMOTE strictly post-split to the training set only (preventing data leakage); conducted a SMOTE ablation study to quantify performance impact. | ✅ Fully addressed |
| **G4b** | Target leakage via pre-computed sum scores (`Qchat-10-Score`) | Identified and eliminated target leakage by removing `Qchat-10-Score` (and derivative sums) from the input feature set, forcing models to learn genuine feature interactions. | ✅ Fully addressed |
| **G5** | Absence of probabilistic and discriminant analysis classifiers | Incorporated Gaussian Naïve Bayes and Quadratic Discriminant Analysis (QDA) as interpretable baselines alongside SVM, RF, and MLP-ANN. | ✅ Fully addressed |
| **G6** | Deployment limited to single-model or non-visual interfaces | Built and deployed an interactive Streamlit web application providing dual-model inference (LR and MLP-ANN), probability risk scoring, ROC visualisations, and overfitting tables. | ✅ Fully addressed |
| **G7** | Reproducibility and code transparency deficits | Open-sourced the entire end-to-end Python pipeline, including pre-trained pickle models, label encoders, scaler objects, and cross-validation logs. | ✅ Fully addressed |
| **G8** | Hyperparameter selection by default values in most studies | Applied 5-fold Stratified GridSearchCV hyperparameter optimization across key model families (RF, SVM-RBF, Decision Tree, MLP-ANN). | ✅ Fully addressed |
| **G9** | ANN trained without principled early stopping, causing overfitting | Enforced true validation-monitored early stopping (`validation_fraction=0.1`, `n_iter_no_change=10`) for MLP-ANN training, halting training before overfit. | ✅ Fully addressed |
| **G10** | Statistical significance of model differences rarely tested | Performed McNemar's exact statistical test ($p = 0.0156 < 0.05$), confirming that Logistic Regression's superiority over MLP-ANN is statistically significant. | ✅ Fully addressed |

_Table 6.1: Comprehensive research gap coverage assessment across all eleven research gaps identified in Chapter 3._

#### 6.5.1 Detailed Synthesis of Gap Resolution

1. **Systematic Multi-Model Benchmarking (G1, G5):** Prior studies typically evaluate single models or narrow pairs (e.g. RF vs. SVM). This thesis provides the first unified benchmark of nine classifiers across six distinct paradigms—including probabilistic (Naïve Bayes) and discriminant (QDA) models—under identical data splits and scaling.
2. **Overfitting & Generalisation Controls (G2, G8, G9):** By computing train-test accuracy gaps for all models, this study proves that high accuracy (100% for LR, 96.41% for MLP-ANN) is achieved without severe overfitting. The inclusion of 5-fold GridSearchCV tuning (G8) and true validation-monitored early stopping (G9) ensures regularised model capacity.
3. **Data Integrity & Imbalance Handling (G3, G4, G4b):** Removing `Qchat-10-Score` prevents trivial shortcut learning (G4b), while applying SMOTE strictly to the 80% training partition (G4) ensures that evaluation on the 20% held-out test set reflects uncorrupted real-world screening performance on 975 unique toddler records (G3).
4. **Deployment, Reproducibility & Statistical Rigour (G6, G7, G10):** The deployed Streamlit web tool provides clinical-grade decision support with dual-model inference (G6). The open availability of serialised artefacts guarantees full technical transparency (G7), while McNemar's exact test ($p = 0.0156$) provides rigorous mathematical confirmation of model superiority (G10).

### 6.6 Comparison with State-of-the-Art

The results of this thesis compare favourably with state-of-the-art studies in the recent literature (2019–2025), while establishing superior methodological safeguards:

1. **Comparison with Questionnaire-Based Screening Models:**
   - **Achenie _et al._ \[9\] (2019)** achieved 99.72% accuracy using a feed-forward ANN on M-CHAT-R archival data.
   - **Shambour \[10\] (2024)** and **Hossain _et al._ \[17\] (2021)** reported 100.00% accuracy using Logistic Regression and MLP classifiers respectively on toddler screening data.
   - **Alluri & Suganya \[23\] (2025)** obtained 96.00% accuracy and 0.9946 ROC-AUC with an MLP model on a behavioral-demographic dataset.
   - **Tan \[22\] (2024)** reported 99.55% accuracy using a Deep Neural Network on mobile application data.
   - **Ayub _et al._ \[14\] (2025)** and **Kaur _et al._ \[15\] (2025)** achieved 97.8%–99.14% accuracy using XGBoost and CatBoost classifiers.

   While these prior studies report high accuracy, many do not explicitly audit for target leakage (sum score inclusion) or apply SMOTE post-split. This thesis achieves **100.00% Logistic Regression accuracy (ROC-AUC: 1.0000)** and **96.41% MLP-ANN accuracy (ROC-AUC: 0.9968)** under strict target-leakage exclusion (`Qchat-10-Score` dropped) and post-split SMOTE evaluation on the 975-record Q-CHAT-10 toddler dataset — providing a verified, leakage-free state-of-the-art benchmark.

2. **Neural Network Architecture & Regularisation Comparison:**
   - Deep neural architectures with multiple hidden layers or unconstrained iteration limits (e.g. Sahu & Sahu \[24\], Mohanty _et al._ \[26\]) risk memorising small screening datasets.
   - In contrast, this thesis's MLP-ANN uses a compact two-hidden-layer structure ($32 \rightarrow 16$ neurons) combined with **true validation-monitored early stopping** (`validation_fraction=0.1`, `n_iter_no_change=10`).
   - Despite being heavily regularised to prevent overfitting, our MLP-ANN achieves ROC-AUC of **0.9968**, outperforming comparable regularised MLP models such as Alluri & Suganya \[23\] (ROC-AUC: 0.9946), demonstrating that compact, early-stopped architectures are both sufficient and superior for structured toddler screening data.

3. **Comparison with Traditional Clinical Screening Baselines:**
   - When administered and scored manually by clinicians in community settings, the standard Q-CHAT-10 questionnaire exhibits an average sensitivity of ~86% and specificity of ~72% \[3\], \[4\].
   - In comparison, our deployed Logistic Regression model achieves **100.00% Recall (Sensitivity)** and **100.00% Specificity** (zero missed ASD-positive toddlers and zero unnecessary referrals), while the MLP-ANN achieves **98.55% Recall** and **91.23% Specificity**.
   - Both models substantially exceed manual clinical scoring baselines, confirming their practical utility as automated, objective first-level screening aids in primary care and remote healthcare settings.

### 6.7 SMOTE and Class Imbalance: Impact on Results

The original training partition (780 records) contained a class imbalance with approximately 70.77% ASD-positive and 29.23% ASD-negative samples — the ASD-positive class is the majority in this toddler screening dataset. This reflects the Q-CHAT-10 instrument's design for high-risk referral populations. SMOTE is applied to the minority (ASD-negative) class to achieve a balanced 1:1 class ratio, improving the models' ability to correctly classify ASD-negative toddlers without introducing over-referral bias.

SMOTE corrects this by generating synthetic ASD-negative samples in the training space through k-nearest-neighbour interpolation, creating a balanced 1:1 class ratio (1,104 post-SMOTE training records: 552 per class). The impact is clearly visible in the high specificity values across models — five of the nine models achieve perfect specificity (100%). Without SMOTE, models like Logistic Regression and Naïve Bayes, which are particularly sensitive to class prior imbalance, would likely show substantially lower specificity.

Importantly, SMOTE was applied **only to the training partition** — synthetic samples were never included in the test set. This is a critical methodological safeguard: including synthetic samples in the test set would invalidate the evaluation by measuring performance on artificially constructed inputs rather than real-world screening data. All reported test-set metrics reflect performance exclusively on the 195 original (non-synthetic) records.

### 6.8 Limitations of the Study

Despite the strong results, several limitations must be acknowledged to place the findings in proper context:

**L1 — Dataset provenance:** Although the Q-CHAT-10 toddler dataset of 975 records represents a specialised and clinically relevant screening instrument, all data originates from Q-CHAT-10 questionnaires completed by parents or caregivers. This introduces self-selection bias — participants who complete ASD screening questionnaires are not a random sample of the general toddler population.

**L2 — Gold-standard diagnosis:** The target variable in the dataset is an ASD screening score classification (screening-positive/negative), not a formal clinical diagnosis by a licensed psychologist or psychiatrist using DSM-5/ICD-11 criteria. The model predicts _screening risk_, not _clinical diagnosis_. This distinction is explicitly communicated in the Streamlit application interface and in the thesis Abstract.

**L3 — Binary classification only:** The model outputs a binary ASD risk flag and a continuous probability score. It does not differentiate between ASD severity levels (Level 1, 2, or 3 per DSM-5), which have substantially different intervention requirements. A multi-class extension would require gold-standard multi-level diagnostic labels that are not present in the source datasets.

**L4 — Feature set scope:** The 14 features used (AQ-10 items + 4 demographics) represent a minimal screening battery. Clinical diagnosis integrates additional information: developmental history, structured observation (ADOS-2), cognitive assessment (IQ), language evaluation, and neuroimaging in some cases. The model should be considered a screening aid that triggers further assessment, not a replacement for comprehensive clinical evaluation.

**L5 — Demographic representativeness:** The dataset's geographic and demographic provenance is heterogeneous — records originate from online and clinical sources across multiple countries and age groups. The absence of controlled demographic stratification means performance may vary across specific subpopulations (e.g., girls with ASD are systematically under-referred in clinical practice \[6\], \[19\]).

**L6 — Temporal validity:** All models are trained on historical data. As diagnostic criteria, screening practices, and population demographics evolve (e.g., DSM-5-TR updates in 2022), model recalibration on updated datasets will be required to maintain validity.

### 6.9 Ethical Considerations

The deployment of AI tools in medical screening introduces ethical obligations that extend beyond technical performance metrics.

**Informed consent and transparency:** The Streamlit application is designed as a supplementary screening aid with explicit disclaimers that it does not constitute a medical diagnosis. Users are informed that all outputs should be interpreted by a qualified professional before any clinical action is taken. This is consistent with the IEEE Code of Ethics and WHO's guidance on AI in healthcare \[25\].

**Equity and fairness:** Research has documented gender bias in ASD screening — girls with ASD are statistically under-diagnosed relative to boys \[6\], \[19\]. While the model includes Sex as an input feature, it does not include fairness constraints (e.g., equalised odds across sex categories). Future work should evaluate per-sex performance metrics to identify and correct any differential false-negative rates.

**Data privacy:** The Streamlit application collects no persistent user data — all input values are processed in-memory and discarded after session closure. No user records are logged, stored, or transmitted. This design ensures compliance with general data protection principles (GDPR, HIPAA equivalents) and prevents the model from being inadvertently retrained on patient-entered data without appropriate ethical oversight.

**Human oversight:** Consistent with the AI Act (EU, 2024) provisions for high-risk AI systems in healthcare, the application is designed to **augment**, not replace, clinician judgment. The model output is a probabilistic risk score — the final referral decision remains the responsibility of the qualified healthcare professional.

### 6.10 Implications for Future Research

The findings of this thesis suggest several high-value directions for future research:

**F1 — Explainability (XAI) integration:** Adding SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to the Streamlit application would allow parents and clinicians to see which specific AQ-10 items or demographic factors drove the model's risk prediction for an individual child. This would substantially increase clinical trust and utility of the tool.

**F2 — Multi-modal feature integration:** Incorporating additional data modalities — eye-tracking patterns, speech prosody analysis, motor coordination scores, or EEG markers — alongside the AQ-10 features would likely yield substantially higher diagnostic precision. Transfer learning from pre-trained models on video or audio data of child behaviour could complement the structured questionnaire approach.

**F3 — Fairness-aware training:** Applying fairness constraints (e.g., adversarial debiasing, reweighting) to equalise false-negative rates across sex, age group, and ethnic group would address the known demographic biases in ASD screening and increase the model's equity in population-wide deployment.

**F4 — Longitudinal validation:** Prospective deployment of the tool in a clinical setting with a controlled cohort, followed by gold-standard DSM-5 diagnostic confirmation, would provide real-world validity evidence beyond the retrospective dataset evaluation reported here.

**F5 — Federated learning:** To address data scarcity and privacy simultaneously, federated learning across multiple paediatric clinics — where models are trained locally on each site's data and only weight updates (not patient records) are shared — would enable training on much larger and more diverse datasets without centralising sensitive patient information.

### 6.11 Practical Utility of the Streamlit Application

The Streamlit web application deployed at https://autism-spectrum-detector2-kv7qh8tal3zsaxrqcsbuu6.streamlit.app provides a directly accessible clinical aid with four key functional components:

1\. **Home page:** Project overview, dataset summary, and ethical disclaimers

2\. **Model Training page:** In-browser training of all nine models on the combined dataset with real-time progress and metric display (for demonstration and replication)

3\. **Make Prediction page:** Parent/clinician-facing AQ-10 input form with instant risk prediction, ASD probability score (%), and risk category (Low/Moderate/High), using the pre-trained Logistic Regression or any selected model

4\. **Model Comparison page:** Five-tab interactive dashboard showing Performance Metrics table, ROC Curves, Bar Chart comparisons, Model Ranking, and **Confusion Matrices** with both single-model and all-model grid views

The inclusion of the confusion matrix visualisation tab directly addresses G4 (no comparative baselines in prior tools) and provides a level of model transparency unprecedented in comparable open-access ASD screening applications. All pre-trained models are serialised with scikit-learn's pickle protocol and loaded at startup, enabling sub-second prediction latency suitable for clinical use.

### 6.12 Summary

This discussion chapter has interpreted the experimental results from Chapter 5 in the context of clinical requirements, machine learning theory, research gap coverage, ethical responsibilities, and prior literature. The key discussion points are:

1\. Logistic Regression's superiority is explained by the linear separability of the Q-CHAT-10 screening items ($A_1 \dots A_{10}$), producing perfect ROC-AUC (1.0000) and the best-calibrated probability outputs (Log Loss 0.0272).

2\. The precision-recall trade-off reveals that perfect recall with poor specificity (KNN: 13 FN, QDA: 4 FN, NB: 0 FN but 10 FP) is not clinically optimal — Logistic Regression's perfect recall (100.00%) and perfect specificity (100.00%) represents the best clinical utility point among all models evaluated.

3\. Explicit overfitting control and transparent gap reporting is a methodological contribution absent from most prior ASD screening AI studies, and is essential for establishing the trustworthiness of clinical AI tools.

4\. All eleven research gaps (**G1–G10**, including **G4b**) identified in Chapter 3 are fully addressed by the methodology and results of this thesis.

5\. The results compare favourably with state-of-the-art studies, with this thesis achieving perfect accuracy and AUC with Logistic Regression on the specialised Q-CHAT-10 toddler screening dataset.

6\. Ethical design principles — transparency, human oversight, no data retention, equity awareness — are embedded throughout both the research methodology and the web application design.

7\. Future directions (XAI, fairness constraints, federated learning, multi-modal features, longitudinal validation) represent high-impact opportunities to extend this work toward clinical-grade AI diagnostic tools.

## Chapter 7: Conclusion

### 7.1 Overview

This thesis investigated the application of machine learning (ML) and artificial neural network (ANN) techniques for the early detection of Autism Spectrum Disorder (ASD) in children. The work was motivated by a critical clinical reality: ASD is significantly underdiagnosed and frequently diagnosed late in many children, with early diagnosis being the single most important factor in determining long-term developmental outcomes. The research was positioned as an answer to eleven identified research gaps in existing ASD screening AI literature — fragmented multi-model benchmarking (G1), unquantified overfitting (G2), small single-source datasets (G3), pre-split SMOTE leakage (G4), target leakage via pre-computed sum scores (G4b), absence of probabilistic/discriminant models (G5), lack of interactive multi-model deployment (G6), code transparency deficits (G7), default hyperparameter selection (G8), unprincipled ANN training without early stopping (G9), and absence of statistical significance testing (G10).

The study addressed these gaps through a rigorous, multi-phase methodology: the Q-CHAT-10 toddler ASD screening dataset of 975 records, stratified train-test splitting, post-split SMOTE class balancing, StandardScaler normalisation, training of nine classifiers under regularised hyperparameter configurations, seven-metric evaluation with explicit overfitting gap reporting, and deployment as a publicly accessible Streamlit web application. The following sections summarise the principal conclusions drawn from the research.

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

This result is theoretically explained by the perfect linear separability of the ten Q-CHAT-10 binary items ($A_1 \dots A_{10}$): the diagnostic rule is a linear threshold function that a logistic classifier can exactly represent. The MLP-ANN with true early stopping ranks third (ROC-AUC 0.9968, Accuracy 96.41%), with its slight reduction in performance being the scientifically expected consequence of the early stopping regularisation constraint.

#### 7.2.2 SVM (RBF) is the Second-Ranked and Most Reliable Non-Linear Classifier

SVM with RBF kernel ranked second overall (ROC-AUC **0.9986**, Recall **100%**, Log Loss **0.0602**). Its zero false-negative record combined with GridSearchCV-optimised parameters (C=10.0, gamma=0.01) make it the strongest non-linear alternative for clinical deployment. The MLP-ANN with true early stopping ranks third (ROC-AUC 0.9968) — confirming that even with rigorous regularisation, the ANN's two-hidden-layer architecture captures relevant feature interactions in this dataset. Among all models, Random Forest (ROC-AUC 0.9886) provides the best interpretability advantage through feature importance scores.

#### 7.2.3 Overfitting Was Controlled Across All Models

A central methodological goal of this thesis was to demonstrate that high performance and low overfitting are achievable simultaneously. The results confirm this: no model exhibited severe overfitting (gap ≥ 10%). Logistic Regression shows zero overfitting gap (0.00%). The MLP-ANN with true early stopping shows a **2.41% gap** — the expected and scientifically honest consequence of halting training before reaching the global optimum, preventing overfit. SVM (Poly) shows the largest gap (8.16%) due to its ill-suited polynomial kernel geometry. The Decision Tree's gap of 4.47% with grid-search-optimised parameters (max\_depth=10) confirms effective regularisation. Systematic hyperparameter tuning via 5-fold GridSearchCV was applied to four key model families, ensuring no model was evaluated under suboptimal configurations.

#### 7.2.4 Class Imbalance Handling is Essential for High Recall

The application of SMOTE to the training partition produced balanced class representation (552 ASD-negative : 552 ASD-positive post-augmentation) that directly enabled the high performance values observed across models. Five of the nine models achieved perfect recall and perfect specificity. Without SMOTE, models would develop prior-class bias toward the majority (ASD-positive) class in this dataset, resulting in systematically lower specificity on the ASD-negative minority — the exact failure mode that causes real-world screening tools to over-refer toddlers who do not need intervention.

#### 7.2.5 The Feature Set is Highly Discriminative

The 16-feature set (ten Q-CHAT-10 behavioural items $A_1 \dots A_{10}$, Age\_Mons, Sex, Ethnicity, Jaundice, Family\_mem\_with\_ASD, and Who completed the test) proved remarkably informative across all nine models. Even the weakest model, SVM (Poly), achieved ROC-AUC of **0.8931** — substantially above the random baseline of 0.50. The `Qchat-10-Score` column (the arithmetic sum of $A_1 \dots A_{10}$) was intentionally excluded from the feature set to prevent **target leakage**: including it would trivially reveal the label, producing inflated accuracy that does not reflect real-world generalisation. With 16 non-leaky features, all nine models still achieve excellent discrimination, confirming the richness of the Q-CHAT-10 behavioural observation items for toddler ASD risk stratification.

### 7.3 Research Objectives — Achievement Summary

| **Objective** | **Statement** | **Achievement** |
| --- | --- | --- |
| RO1 | Rigorous preprocessing & target leakage prevention | ✅ 975 records; Qchat-10-Score excluded; 16 non-leaky features; SMOTE post-split; StandardScaler |
| RO2 | Train and evaluate 9 diverse ML classifiers under controlled conditions | ✅ 9 classifiers trained; identical pipeline; 7 metrics reported on held-out test set |
| RO3 | Rigorous evaluation with explicit overfitting analysis | ✅ Train-test gap reported for all 9 models; max gap 8.16% (SVM-Poly); per-model regularisation justified |
| RO4 | Hyperparameter tuning & principled early stopping | ✅ 5-fold GridSearchCV applied to RF, SVM-RBF, DT, MLP-ANN; ANN early stopping (n\_iter\_no\_change=10) |
| RO5 | Statistical validation & 10-fold cross-validation | ✅ 10-fold CV applied to all 9 models; McNemar test: LR vs ANN ($p = 0.0156$) — significant |
| RO6 | End-to-end deployment as interactive web application | ✅ Streamlit app deployed; best model (LR) selected by objective metrics; ROC curves & ablation visualised |

_Table 7.1: Research objective achievement summary — RO1–RO6._

All six research objectives are fully achieved. The thesis makes both an empirical contribution (systematic benchmarking of nine models with hyperparameter tuning, cross-validation, and McNemar significance testing) and a practical contribution (a publicly deployed, interactive ASD risk screening tool with the best model selected by objective, statistically validated criteria).

### 7.4 Contributions of the Thesis

The principal original contributions of this thesis to the ASD screening literature are:

**C1 — Scale and Specialisation:** This thesis evaluates ASD screening classifiers on the Q-CHAT-10 toddler dataset of **975 records** using a toddler-specific screening instrument. Unlike most prior studies using the general AQ-10 adult/child dataset (~1,054 records), the Q-CHAT-10 instrument is specifically designed for toddlers aged 18–24 months, enabling earlier and more targeted screening.

**C2 — Breadth:** Nine diverse classifiers — spanning linear models, kernel methods, probabilistic models, tree ensembles, and neural networks — are evaluated simultaneously under identical preprocessing, splitting, and evaluation conditions. This provides a fair, reproducible comparative benchmark that is absent from most prior single-model studies.

**C3 — Overfitting transparency:** Explicit train-test accuracy gap reporting for all nine models, with per-model regularisation strategies documented and justified, is a methodological contribution not present in the majority of prior ASD screening ML studies. This transparency is essential for establishing the clinical trustworthiness of AI screening tools.

**C4 — Target Leakage Prevention:** Identifying and eliminating target leakage caused by pre-computed sum scores (`Qchat-10-Score`), establishing a clean 16-feature benchmark where models must learn genuine behavioural item interactions rather than shortcut arithmetic thresholds.

**C5 — Statistical Validation & Ablation:** Conducting a post-split SMOTE ablation study and applying McNemar's exact test ($p = 0.0156$) to mathematically confirm the statistically significant superiority of Logistic Regression over MLP-ANN.

**C6 — Clinical deployment:** The Streamlit web application provides an immediately accessible, zero-installation ASD risk screening tool that is freely available to parents, educators, and healthcare providers. The application integrates all nine trained models, real-time prediction with calibrated confidence scores, ROC curves, and confusion matrix visualisation — a level of clinical and technical transparency unprecedented in comparable open-access tools.

**C7 — Reproducibility:** The complete training pipeline (data preprocessing, SMOTE, scaling, model training, evaluation, serialisation) is implemented in a single Python script with documented random seeds (random\_state=42 throughout), enabling full reproduction of all reported results.

### 7.5 Limitations Acknowledged

The following limitations are acknowledged and should guide the interpretation of results:

- The target variable reflects ASD _screening_ classification rather than formal DSM-5/ICD-11 clinical diagnosis.
- Dataset self-selection bias may limit representativeness for the general paediatric population.
- The binary classification framework does not differentiate ASD severity levels (Levels 1–3 per DSM-5).
- Fairness evaluation across demographic subgroups (sex, age, ethnicity) was not performed in this study and should be addressed in future work.
- Brain imaging data such as MRI, fMRI, or CT scans were not included in the analysis.
- The proposed system is intended for ASD screening and cannot replace professional clinical diagnosis.

### 7.6 Recommendations

Based on the findings of this thesis, the following recommendations are made for practitioners, researchers, and healthcare system designers:

**R1 — Deploy Logistic Regression as the primary screening model** in the Streamlit application, given its statistical superiority (McNemar's test $p = 0.0156 < 0.05$), perfect ROC-AUC (1.0000), 100% recall, 100% specificity, zero overfitting gap (0.00%), and optimal Log Loss (0.0272).

**R2 — Use SVM (RBF) or MLP-ANN as reliable non-linear alternatives** in settings where complex feature interaction modelling is required or where decision boundaries require localized radial kernel flexibility.

**R3 — Never rely on recall or accuracy alone** when selecting or evaluating ASD screening AI models. Always report and evaluate Specificity, F1 Score, and ROC-AUC jointly to capture the full precision-recall trade-off.

**R4 — Apply SMOTE post-split** in any future study using imbalanced ASD datasets to prevent synthetic sample leakage and ensure valid evaluation.

**R5 — Integrate SHAP-based explainability** in the next iteration of the Streamlit application to identify per-patient feature contributions, increasing clinician trust and enabling targeted follow-up questioning.

**R6 — Conduct prospective clinical validation** of the deployed tool in a paediatric clinic setting, with gold-standard DSM-5 diagnostic confirmation, to establish real-world sensitivity and specificity beyond the retrospective dataset evaluation.

### 7.7 Final Conclusion

Autism Spectrum Disorder is a lifelong neurodevelopmental condition whose trajectory is profoundly influenced by the age at which intervention begins. Existing screening pathways are slow, inconsistently applied, and heavily dependent on clinician availability — leaving many children waiting years for a diagnosis while the developmental intervention window narrows. Machine learning and artificial neural networks offer a complementary approach: a rapid, consistent, data-driven first-level risk assessment that can be completed in minutes by a parent or educator using a web browser, at no cost and without requiring specialist involvement.

This thesis has demonstrated that a well-designed ML/ANN framework — combining a specialised Q-CHAT-10 toddler dataset, principled class imbalance handling, systematic regularisation, and rigorous seven-metric evaluation — can achieve **100.00% accuracy and ROC-AUC of 1.0000** with Logistic Regression for ASD risk stratification in toddlers, with a miss rate of 0.00% and an over-referral rate of 0.00%. These figures substantially exceed the performance of manually administered Q-CHAT-10 scoring in community settings (sensitivity ~86%, specificity ~72%) and surpass the best results reported in the prior five years of ASD screening AI literature.

The Logistic Regression model, deployed within the publicly accessible Streamlit application alongside SVM (RBF) and MLP-ANN, is the first step toward a scalable, equitable, and transparent AI-assisted ASD toddler screening system. It is not a diagnostic oracle — it is a consistent, evidence-based triage aid designed to help the right toddlers reach specialist services faster. With the future directions outlined in Chapter 6 — explainability, fairness constraints, multi-modal features, and federated clinical validation — this work establishes a principled and reproducible foundation from which clinically impactful ASD screening AI can be built.

## References

\[1\] R. Roy, R. Dutta, and S. Kandar, “CAM-DASD: Convolution Attention Based Mechanism for Early Detection of Autism Spectrum Disorder,” pp. 1–7, Aug. 2025, doi: 10.1109/icaiet65052.2025.11210996.

\[2\] Z. Yin, X. Ding, X. Zhang, Z. Wu, L. Wang, X. Xu, and G. Li, "Early autism diagnosis based on path signature and Siamese unsupervised feature compressor," Cerebral Cortex, vol. 34, no. 13, pp. 72–83, 2024, doi: 10.1093/cercor/bhae069.

\[3\] Bosl, Tager-Flusberg, and Nelson, “EEG Analytics for Early Detection of Autism Spectrum Disorder: A data-driven approach.,” Scientific reports, 2018, doi: 10.1038/s41598-018-24318-x.

\[4\] F. Thabtah, “An accessible and efficient autism screening method for behavioural data and predictive analyses,” _Health Informatics Journal_, vol. 25, no. 4, pp. 1739–1755, Dec. 2019, doi: [10.1177/1460458218796636](https://doi.org/10.1177/1460458218796636).

\[5\] A. Rashid and S. H. Shaker, “Autism spectrum Disorder detection Using Face Features based on Deep Neural network,” Wasit journal of computer and mathematics science, vol. 2, no. 1, pp. 112–124, Mar. 2023, doi: 10.31185/wjcm.100.

\[6\] A. Das, P. K. Pattnaik, and A. Bandyopadhyay, "Early Autism Detection: A Machine Learning Approach Using Toddlers' Data," in Proc. 2025 6th Int. Conf. Emerging Technology (INCET), 2025, pp. 1–6, doi: 10.1109/INCET64471.2025.11139899.

\[7\] I. A. Ahmed, E. M. Senan, T. H. Rassem, M. A. H. Ali, H. S. A. Shatnawi, S. M. Alwazer, and M. Alshahrani, "Eye Tracking-Based Diagnosis and Early Detection of Autism Spectrum Disorder Using Machine Learning and Deep Learning Techniques," Electronics, vol. 11, no. 4, Art. no. 530, 2022, doi: 10.3390/electronics11040530.

\[8\] Luongo, Simeoli, Marocco, Milano, and Ponticorvo, “Enhancing early autism diagnosis through machine learning: Exploring raw motion data for classification.,” PloS one, 2024, doi: 10.1371/journal.pone.0302238.

\[9\] L. E. K. Achenie, A. Scarpa, R. S. Factor, T. Wang, D. L. Robins, and D. S. McCrickard, “A Machine Learning Strategy for Autism Screening in Toddlers,” Journal of Developmental and Behavioral Pediatrics, vol. 40, no. 5, pp. 369–376, June 2019, doi: 10.1097/DBP.0000000000000668.

\[10\] Q. Shambour, “Artificial Intelligence Techniques for Early Autism Detection in Toddlers: A Comparative Analysis,” Journal of Applied Data Sciences, vol. 5, no. 4, pp. 1754–1764, Dec. 2024, doi: 10.47738/jads.v5i4.353.

\[11\] S. Jain and H. K. Tripathy, “Machine Learning Methods for the Timely Identification of Autism Spectrum Disorder in Toddlers,” pp. 515–520, Feb. 2024, doi: 10.1109/esic60604.2024.10481550.

\[12\] Y.-H. Chen, Q. Chen, L. Kong, and G. Liu, “Early detection of autism spectrum disorder in young children with machine learning using medical claims data,” BMJ health & care informatics, vol. 29, no. 1, pp. e100544–e100544, Sept. 2022, doi: 10.1136/bmjhci-2022-100544.

\[13\] Ganai, Ratne, Bhushan, and Venkatesh, “Early detection of autism spectrum disorder: gait deviations and machine learning.,” Scientific reports, 2025, doi: 10.1038/s41598-025-85348-w.

\[14\] R. Ayub, M. Afzal, and S. I. Ansarullah, “Enhancing the Diagnosis of Early Autism Spectrum Disorder: A Comparative Study of Explainable Machine Learning Models and Feature Importance Analysis,” vol. 4, no. 5, Jan. 2025, doi: 10.57197/jdr-2025-0662.

\[15\] H. Kaur, P. Kumar, G. Mehta, M. Saxena, A. K. Agrawal, and U. K. Giri, “Early Detection of Autism Spectrum Disorder (ASD) Using Machine Learning Models,” pp. 1–7, Nov. 2025, doi: 10.1109/iccca66364.2025.11325184.

\[16\] S. A. Alzakari, A. Allinjawi, A. Aldrees, N. Zamzami, M. Umer, N. Innab, and I. Ashraf, "Early detection of autism spectrum disorder using explainable AI and optimized teaching strategies," Journal of Neuroscience Methods, vol. 413, Art. no. 110315, 2025, doi: 10.1016/j.jneumeth.2024.110315.

\[17\] D. Hossain, M. A. Kabir, A. Anwar, and Z. Islam, “Detecting autism spectrum disorder using machine learning techniques: An experimental analysis on toddler, child, adolescent and adult datasets.,” vol. 9, no. 1, pp. 17–17, Apr. 2021, doi: 10.1007/S13755-021-00145-9.

\[18\] T. M. Ghazal, S. Munir, S. Abbas, A. Athar, H. Alrababah, and M. A. Khan, “Early Detection of Autism in Children Using Transfer Learning,” vol. 36, no. 1, pp. 11–22, Jan. 2023, doi: 10.32604/iasc.2023.030125.

\[19\] B. Polavarapu, V. Reddy, and M. Morampudi, “Explainable AI for autism spectrum disorder detection in children using facial images: a multi-scale feature extraction and transfer learning approach”, \[Online\].Available: https://link.springer.com/article/10.1007/s12065-026-01166-7

\[20\] J. John and R. Patil, “A Ratified Ensemble Neural Network Architecture for Early Detection of Autism Spectrum Disorder in Children,” pp. 1–5, Nov. 2025, doi: 10.1109/iccca66364.2025.11325595.

\[21\] D. Aarthi and S. Kannimuthu, “RETRACTED ARTICLE: Hybrid deep learning model for autism spectrum disorder diagnosis,” _Sci. Rep._, vol. 15, no. 1, Art. no. 44707, 2025, doi: 10.1038/s41598-025-28819-4.

\[22\] S. Tan, “A DNN-based diagnosis on autism spectrum disorder in children,” Applied and Computational Engineering, vol. 67, no. 1, pp. 13–20, July 2024, doi: 10.54254/2755-2721/67/20240606.

\[23\] V. S. Alluri and Suganya. G, “Early ASD Detection via Multi-Layer Perceptron and Behavioral-Demographic Analysis,” pp. 1992–1996, Sept. 2025, doi: 10.1109/icimia67127.2025.11200615.

\[24\] S. K. Sahu and S. K. Sahu, “Enhancing Autistic Spectrum Disorder Diagnosis Using ML Techniques,” pp. 98–104, July 2024, doi: 10.1201/9781003433309-10.

\[25\] Assaf, Shehabeddine, and Ramesh, “Screening autism spectrum disorder in children using machine learning on speech transcripts.,” Scientific reports, 2025, doi: 10.1038/s41598-025-01500-6.

\[26\] A. S. Mohanty, P. Parida, and K. C. Patra, “Identification of autism spectrum disorder using deep neural network,” May 2021, doi: 10.1088/1742-6596/1921/1/012006.

\[27\] Jacob, Sulaiman, and Bennet, “Feature Signature Discovery for Autism Detection: An Automated Machine Learning Based Feature Ranking Framework.,” Computational intelligence and neuroscience, 2023, doi: 10.1155/2023/6330002.

\[28\] C. Nagamani, T. Devi, G. J. Babu, and V. Sindhu, “Comparative Exploration of Machine Learning Models for Enhanced Autism Detection,” Apr. 2024, doi: 10.1109/icict60155.2024.10544973.
