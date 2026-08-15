import streamlit as st
import pandas as pd
import plotly.express as px

from core.engine import (
    predict_one,
    generate_recommendations,
    shap_explanation,
    readable_feature_name
)


st.title("👤 Student Analysis")

st.caption(
    "Deep-dive into one student's prediction, learning indicators and model explanation."
)


if st.session_state.data is None:

    st.info(
        "Upload a dataset from the sidebar first."
    )

    st.stop()


df = st.session_state.data


student_ids = (
    df["student_id"]
    .astype(str)
    .tolist()
)


selected_id = st.selectbox(
    "Select Student",
    student_ids
)


student = (
    df[
        df["student_id"].astype(str)
        == str(selected_id)
    ]
    .iloc[0]
)


required_features = st.session_state.get(
    "required_features"
)


if required_features is None:

    from core.engine import load_metadata

    metadata = load_metadata()

    required_features = metadata[
        "features"
    ]


# ============================================================
# PREDICTION
# ============================================================

from core.engine import load_model

model = load_model()


risk, probabilities = predict_one(
    model,
    student,
    required_features
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="card">
        <h2>Student {selected_id}</h2>
        <p>
            Current predicted risk:
            <strong>{risk}</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# PROBABILITIES
# ============================================================

col1, col2 = st.columns(
    [1, 2]
)


with col1:

    st.metric(
        "Predicted Risk",
        str(risk)
    )


with col2:

    probability_df = pd.DataFrame(
        {
            "Risk": list(
                probabilities.keys()
            ),
            "Probability": [
                value * 100
                for value
                in probabilities.values()
            ]
        }
    )

    fig = px.bar(
        probability_df,
        x="Risk",
        y="Probability",
        text_auto=".1f",
        title="Risk Probability"
    )

    fig.update_layout(
        height=330,
        template="plotly_white",
        yaxis_title="Probability (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PROFILE
# ============================================================

st.markdown(
    "### 📋 Learning Profile"
)


profile_features = [
    "attendance_pct",
    "assignment_avg",
    "assignment_completion_pct",
    "quiz_avg",
    "study_hours_week",
    "participation_score",
    "lms_active_days",
    "previous_gpa",
    "previous_failures",
    "late_assignments",
    "midterm_score"
]


available = [
    feature
    for feature in profile_features
    if feature in student.index
]


profile = pd.DataFrame(
    {
        "Indicator": [
            readable_feature_name(
                feature
            )
            for feature in available
        ],
        "Value": [
            student[feature]
            for feature in available
        ]
    }
)


st.dataframe(
    profile,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SHAP
# ============================================================

st.markdown(
    "### 🔍 Why did the model make this prediction?"
)


shap_df = shap_explanation(
    model,
    student,
    required_features
)


if shap_df is not None:

    chart_df = shap_df.copy()

    chart_df["Feature"] = (
        chart_df["feature"]
        .apply(
            readable_feature_name
        )
    )

    chart_df = chart_df.sort_values(
        "shap_value"
    )

    fig = px.bar(
        chart_df,
        x="shap_value",
        y="Feature",
        orientation="h",
        title="Top Model Influences"
    )

    fig.update_layout(
        height=480,
        template="plotly_white",
        xaxis_title="SHAP Contribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "SHAP describes how features influenced this model prediction. It does not establish causation."
    )

else:

    st.warning(
        "SHAP explanation could not be generated."
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.markdown(
    "### 💡 Personalized Learning Actions"
)


recommendations = generate_recommendations(
    student
)


for recommendation in recommendations:
    st.markdown(
         f"🟢 {recommendation}"
    )

# ============================================================
# DOWNLOAD INDIVIDUAL STUDENT REPORT
# ============================================================

st.markdown("### 📥 Student Performance Report")

# Get complete record of the selected student
student_report = df[
    df["student_id"].astype(str) == str(selected_id)
].copy()

# Convert the complete student record to CSV
csv_data = student_report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Student Report",
    data=csv_data,
    file_name=f"{selected_id}_performance_report.csv",
    mime="text/csv"
)