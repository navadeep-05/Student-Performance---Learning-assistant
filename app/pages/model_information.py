import streamlit as st
import pandas as pd

from core.engine import load_model, load_metadata


st.title("🤖 Model Information")

st.caption(
    "Technical overview of the student risk prediction system."
)


model = load_model()
metadata = load_metadata()


# ============================================================
# MODEL OVERVIEW
# ============================================================

col1, col2, col3 = st.columns(3)


col1.metric(
    "Model",
    metadata.get(
        "model_name",
        "Random Forest"
    )
)


col2.metric(
    "Task",
    "Risk Classification"
)


col3.metric(
    "Classes",
    len(
        metadata.get(
            "classes",
            []
        )
    )
)


st.divider()


# ============================================================
# CLASSES
# ============================================================

st.markdown(
    "### 🎯 Prediction Classes"
)


classes = metadata.get(
    "classes",
    []
)


for class_name in classes:

    st.write(
        f"• {class_name}"
    )


# ============================================================
# FEATURES
# ============================================================

st.markdown(
    "### 🧩 Model Input Features"
)


features = metadata.get(
    "features",
    []
)


feature_df = pd.DataFrame(
    {
        "Feature": features
    }
)


st.dataframe(
    feature_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FIELDS NOT USED BY MODEL
# ============================================================
st.markdown(
    "### 🚫 Fields Not Used by the Model"
)


st.markdown(
    "**Outcome / Leakage-Prone Fields**"
)

for column in [
    "final_score",
    "performance_level",
    "risk_level"
]:

    st.write(
        f"• `{column}`"
    )


st.markdown(
    "**Identifier**"
)

st.write(
    "• `student_id`"
)


# ============================================================
# ARCHITECTURE
# ============================================================

st.markdown(
    "### 🏗️ AI Pipeline"
)


st.code(
    """
Student Learning Data
        ↓
Data Validation
        ↓
Preprocessing Pipeline
        ↓
Random Forest Classifier
        ↓
Risk Prediction
        ↓
Risk Probabilities
        ↓
SHAP Explainability
        ↓
Personalized Recommendations
        ↓
What-If Simulation
        ↓
GenAI Explanation
""",
    language="text"
)


st.warning(
    "The current model was evaluated on the synthetic dataset used for this prototype. Real institutional data should be used for real-world validation before deployment."
)