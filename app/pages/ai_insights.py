import streamlit as st

from core.engine import (
    load_model,
    load_metadata,
    predict_one,
    generate_recommendations,
    shap_explanation,
    generate_genai_explanation
)


# ============================================================
# PAGE
# ============================================================


st.title("✨ AI Insights")

st.caption(
    "AI-generated explanation of the student's predicted risk and key learning factors."
)


# ============================================================
# CHECK DATA
# ============================================================

if st.session_state.data is None:

    st.info(
        "Upload a student dataset from Home first."
    )

    st.stop()


# ============================================================
# LOAD DATA / MODEL
# ============================================================

df = st.session_state.data

model = load_model()

metadata = load_metadata()

required_features = metadata.get(
    "features",
    []
)


# ============================================================
# STUDENT SELECTION
# ============================================================

st.markdown(
    "### 👤 Select Student"
)

student_ids = (
    df["student_id"]
    .astype(str)
    .tolist()
)

selected_id = st.selectbox(
    "Student",
    student_ids
)


student = (
    df[
        df["student_id"].astype(str)
        == str(selected_id)
    ]
    .iloc[0]
)


# ============================================================
# ML PREDICTION
# ============================================================

risk, probabilities = predict_one(
    model,
    student,
    required_features
)


# ============================================================
# SHAP
# ============================================================

shap_df = shap_explanation(
    model,
    student,
    required_features
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

recommendations = generate_recommendations(
    student
)


# ============================================================
# RISK SUMMARY
# ============================================================

st.divider()

st.markdown(
    "### 🧠 AI Performance Summary"
)


risk_col, prob_col = st.columns(
    [1, 2]
)


with risk_col:

    st.metric(
        "Predicted Risk",
        str(risk).upper()
    )


with prob_col:

    st.caption("Risk probabilities")

    for label, probability in probabilities.items():
        st.write(
            f"• **{str(label).title()}:** {probability * 100:.1f}%"
        )



# ============================================================
# REAL GENAI
# ============================================================

st.markdown(
    "### ✨ Natural-Language Explanation"
)


with st.spinner(
    "Generating AI explanation from ML + SHAP results using Gemini AI..."
):

    ai_response = generate_genai_explanation(
        student=student,
        risk=risk,
        probabilities=probabilities,
        recommendations=recommendations,
        shap_df=shap_df
    )


# ============================================================
# DISPLAY GENAI RESULT
# ============================================================

if ai_response is None:

    st.warning(
        "Gemini API is not configured. "
        "Add GEMINI_API_KEY to .streamlit/secrets.toml."
    )

elif str(ai_response).startswith(
    "GENAI_ERROR::"
):

    st.error(
        "Unable to generate the GenAI explanation."
    )

    st.code(
        str(ai_response).replace(
            "GENAI_ERROR::",
            ""
        )
    )

else:

    st.markdown(
        ai_response
    )


# ============================================================
# RECOMMENDED ACTIONS
# ============================================================

st.divider()

st.markdown(
    "### 💡 Recommended Actions"
)

for recommendation in recommendations:

    st.markdown(
        f"🟢 {recommendation}"
    )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander(
    "🔍 View Model & Explainability Details"
):

    st.markdown(
        "#### Risk Probabilities"
    )

    st.json(
        probabilities
    )


    if shap_df is not None and not shap_df.empty:

        st.markdown(
            "#### Top SHAP Factors"
        )

        st.dataframe(
            shap_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "SHAP explanation is unavailable."
        )


    st.caption(
        "SHAP values indicate how features influenced the model prediction. "
        "They should not be interpreted as causal effects."
    )

