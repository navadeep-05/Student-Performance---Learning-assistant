# AI Student Performance Learning Assistant

# 🎓 AI-Powered Student Performance & Learning Assistant

An intelligent decision-support web application that analyzes student learning data, predicts academic risk, explains the factors influencing each prediction, and provides personalized learning recommendations.

The system combines **Machine Learning, Explainable AI, Generative AI, and interactive analytics** to help faculty identify students who may need academic support at an early stage.

---

## 📌 Project Overview

Educational institutions collect large amounts of student information such as attendance, assessments, assignments, quiz performance, participation, and learning-platform activity.

However, this data is often used mainly for reporting rather than proactively identifying students who may struggle.

The **AI-Powered Student Performance & Learning Assistant** transforms student learning data into actionable insights by providing:

- 🎯 Student academic risk prediction
- 📊 Performance analysis and risk probabilities
- 🔍 Explainable AI insights using SHAP
- 💡 Personalized learning recommendations
- ✨ GenAI-powered natural-language explanations
- 🔮 What-If analysis for academic improvement scenarios
- 📥 Individual student performance reports

---

## 🎯 Project Objective

The main objective is to build a student-performance analysis system capable of answering questions such as:

> **Which students may be academically at risk?**

> **What factors are influencing a student's predicted risk?**

> **What actions could help the student improve?**

> **How could the prediction change if attendance, study habits, or academic performance improves?**

The application is designed as a **decision-support tool** and should complement, not replace, faculty judgment.

---

## ⚙️ How It Works

```text
                         HOME
                           │
                     Upload CSV
                           │
                           ▼
                    Dataset Preview
                           │
                           ▼
                   Student Analysis
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
          Random Forest              SHAP
                │                     │
                └──────────┬──────────┘
                           ▼
                      AI Insights
                           │
                           ▼
                     Gemini LLM
                           │
                           ▼
                Natural-Language
                    Explanation

                           +

                     What-If Lab
                           │
                           ▼
                  Scenario Prediction
```

---

## 🚀 Key Features

### 🏠 Home

The Home page provides a quick introduction to the application and allows faculty to upload a student dataset.

After successful upload, it displays:

- Dataset preview
- Number of students
- High-risk students
- Medium-risk students
- Low-risk students
- Basic dataset information

---

### 👤 Student Analysis

Faculty can select an individual student and examine their academic profile.

The page provides:

- Predicted academic risk
- Risk probabilities
- Important learning indicators
- Student performance information
- Personalized learning actions
- Downloadable individual student report

The system classifies students into:

- 🟢 **Low Risk**
- 🟡 **Medium Risk**
- 🔴 **High Risk**

---

### 🤖 Machine Learning Prediction

A **Random Forest Classifier** is used as the final prediction model.

Multiple machine-learning approaches were evaluated during development, and Random Forest was selected as the production model based on the experimental results.

The model estimates the student's academic risk using learning and engagement indicators rather than outcome/leakage variables.

Examples of model inputs include:

- Attendance percentage
- Assignment performance
- Assignment completion
- Quiz performance
- Quiz attempts
- Weekly study hours
- Participation
- LMS activity
- Learning resources accessed
- Video completion
- Previous GPA
- Previous failures
- Late assignments
- Discussion participation
- Support requests
- Midterm performance

---

### 🔍 Explainable AI with SHAP

Predicting risk alone is not enough.

The project integrates **SHAP (SHapley Additive exPlanations)** to help explain why the machine-learning model produced a particular prediction.

SHAP identifies which learning factors contributed most strongly to the prediction.

This makes the system more transparent and provides faculty with meaningful information behind the model's output.

---

### ✨ AI Insights

The AI Insights module converts machine-learning predictions and explainability results into concise, faculty-friendly language.

It combines:

```text
ML Prediction
      +
Risk Probabilities
      +
Important Factors / SHAP
      +
Learning Recommendations
      ↓
Gemini LLM
      ↓
Natural-Language Explanation
```

The Generative AI layer is powered through the **Google Gemini API**.

Instead of displaying only technical model outputs, the application can explain:

- Student risk status
- Important contributing factors
- Areas requiring attention
- Recommended academic focus

---

### 💡 Personalized Learning Recommendations

The system generates practical recommendations according to the student's learning indicators.

Examples include:

- Improve attendance and attend missed classes regularly.
- Practice quizzes and revise weak concepts.
- Complete pending assignments before upcoming assessments.
- Follow a structured weekly study schedule.
- Participate more actively in classroom discussions.
- Engage with the teacher for guidance on difficult topics.

Recommendations are intended to provide actionable academic guidance based on the student's current learning pattern.

---

### 🔮 What-If Analysis

The **What-If Lab** allows faculty to explore hypothetical improvement scenarios.

For example:

> **What happens to the student's predicted risk if attendance improves?**

The selected feature values can be modified and passed through the trained model again.

The application then compares:

```text
Current Student Profile
          │
          ▼
   Current Prediction

          VS

Modified Student Profile
          │
          ▼
   Scenario Prediction
```

This helps demonstrate how improvements in learning indicators may influence the model's predicted risk.

> What-If results are model-based simulations and should not be interpreted as guaranteed academic outcomes.

---

## 🧠 Model Development Pipeline

The machine-learning workflow follows:

```text
Synthetic Student Data
        │
        ▼
Data Understanding & EDA
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Selection
        │
        ▼
Train / Test Split
        │
        ▼
ML Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Random Forest
        │
        ▼
SHAP Explainability
        │
        ▼
Saved Production Model
        │
        ▼
Streamlit Application
```

The development notebooks are included in the `notebooks/` directory for transparency and reproducibility.

---

## 📊 Dataset

A synthetic student-learning dataset was created specifically for the prototype because no institution-specific dataset was provided.

The dataset represents multiple aspects of student learning, including:

- Academic performance
- Attendance
- Assignments
- Quiz activity
- Study behaviour
- LMS engagement
- Participation
- Previous academic history
- Support-seeking behaviour

The project dataset contains approximately **8,000 student records**.

### Target

The machine-learning task predicts:

```text
risk_level
```

with three classes:

```text
High
Medium
Low
```

### ⚠️ Preventing Data Leakage

Certain fields are intentionally excluded from model inputs when they directly reveal the outcome or could introduce leakage.

For example:

```text
student_id
final_score
performance_level
risk_level
```

`final_score` can be useful for constructing or evaluating the target during dataset development, but it should not be supplied as a prediction feature when the goal is to estimate student risk before the final outcome is known.

---

## 🌍 Real-World Validation

A separate validation experiment was performed using the **UCI Student Performance dataset**.

The purpose of this experiment was not to replace or silently modify the deployed production model.

Instead, it was used as an external benchmark to investigate how the project's modelling approach behaves when applied to real-world educational data with a different feature schema.

The experiment is available in:

```text
notebooks/real_world_validation.ipynb
```

with supporting outputs under:

```text
reports/real_world_validation/
```

Because the UCI dataset and the application's synthetic dataset contain different features, the original production model cannot directly consume the UCI dataset without schema alignment or retraining.

This highlights an important future direction: training and validating the system on institution-specific real-world student data.

---

## 🖥️ Application Structure

The application uses a clean multi-page Streamlit interface.

### Navigation

```text
📊 Dashboard

🏠 Home
👤 Student Analysis
🔮 What-If Lab
✨ AI Insights
🤖 Model Information
```

Each module focuses on a specific part of the analysis workflow.

---

## 🛠️ Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Prediction Model | Random Forest |
| Explainable AI | SHAP |
| Generative AI | Google Gemini |
| Visualization | Matplotlib / Plotly |
| Web Application | Streamlit |
| Model Persistence | Joblib / Pickle |
| Development | Jupyter Notebook |
| Version Control | Git & GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```text
AI-powered Student Assistant/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   ├── app.py
│   │
│   ├── assets/
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── engine.py
│   │
│   └── pages/
│       ├── home.py
│       ├── student_analysis.py
│       ├── what_if.py
│       ├── ai_insights.py
│       └── model_information.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── student_risk_model.pkl
│   └── model_metadata.json
│
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── data_preprocessing.ipynb
│   ├── machine_learning.ipynb
│   └── real_world_validation.ipynb
│
├── reports/
│   └── real_world_validation/
│
├── src/
│
├── generate_dataset.py
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

Move into the project:

```bash
cd "AI-powered Student Assistant"
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API

Create:

```text
.streamlit/secrets.toml
```

and add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "YOUR_SUPPORTED_GEMINI_MODEL"
```

> ⚠️ Never commit `secrets.toml` or API keys to GitHub.

Ensure `.gitignore` contains:

```gitignore
.streamlit/secrets.toml
__pycache__/
*.pyc
venv/
```

### 5. Start the application

```bash
streamlit run app/app.py
```

The application will open in the browser.

---

## ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**.

Deployment flow:

```text
Local Project
      │
      ▼
    Git
      │
      ▼
   GitHub
      │
      ▼
Streamlit Community Cloud
      │
      ▼
Deployed Web Application
```

For cloud deployment, configure `GEMINI_API_KEY` and `GEMINI_MODEL` through the application's **Streamlit Secrets** settings rather than committing credentials to the repository.

---

## 🔐 Security

API credentials are intentionally excluded from the repository.

Never commit:

```text
.streamlit/secrets.toml
```

or hard-code API keys directly inside Python files.

If an API key is accidentally exposed publicly, it should be revoked and replaced immediately.

---

## ⚠️ Important Considerations

This application is an educational prototype and decision-support system.

Predictions represent patterns learned by the machine-learning model and should **not** be treated as definitive judgments about students.

Academic interventions should consider:

- Faculty observations
- Student circumstances
- Institutional context
- Additional academic evidence

The system should support human decision-making rather than replace it.

---

## 🔭 Future Improvements

Potential extensions include:

- Training on institution-specific real-world student datasets
- Larger external validation studies
- Improved model calibration and hyperparameter optimization
- Additional explainability visualizations
- Longitudinal student-performance tracking
- Faculty authentication and role-based access
- Database integration
- Automated student progress reports
- Enhanced GenAI recommendations
- Cloud-based model/API architecture

---

## 🎓 Skills Demonstrated

This project demonstrates practical experience with:

`Python` • `Pandas` • `Data Preprocessing` • `EDA` • `Scikit-learn` • `Random Forest` • `Model Evaluation` • `SHAP` • `Explainable AI` • `Generative AI` • `Gemini API` • `Prompt Engineering` • `Streamlit` • `Data Visualization` • `Git` • `GitHub` • `Cloud Deployment`

---

## 👨‍💻 Project Context

Developed as a guided internship project for the **Generative AI Technical Team at SmartBridge**.

**Project:** AI-Powered Student Performance & Learning Assistant  
**Domain:** Machine Learning & Generative AI  
**Level:** Junior Intern  
**Difficulty:** Intermediate

---

## 📜 Disclaimer

This project is intended for educational, research, and prototype purposes.

The generated risk predictions, explanations, recommendations, and What-If scenarios should be interpreted as **decision-support information**, not as final academic decisions.

---

⭐ If you find this project useful, consider giving the repository a star.
