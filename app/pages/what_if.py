import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.engine import (
    load_model,
    load_metadata,
    what_if,
    generate_recommendations
)


st.title("🔮 What-If Lab")

st.caption(
    "Explore how hypothetical changes in learning indicators alter the model's estimated risk."
)


if st.session_state.data is None:

    st.info(
        "Upload a dataset from the sidebar first."
    )

    st.stop()


df = st.session_state.data

model = load_model()
metadata = load_metadata()

required_features = metadata[
    "features"
]


# ============================================================
# STUDENT
# ============================================================

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


# ============================================================
# CONTROLS
# ============================================================

st.markdown(
    "### 🎛️ Adjust Learning Indicators"
)


changes = {}


editable = [
    "attendance_pct",
    "assignment_avg",
    "assignment_completion_pct",
    "quiz_avg",
    "study_hours_week",
    "participation_score",
    "lms_active_days",
    "midterm_score"
]


for feature in editable:

    if feature not in required_features:
        continue

    current = float(
        student[feature]
    )

    if (
        feature.endswith("_pct")
        or feature in [
            "participation_score"
        ]
    ):

        new_value = st.slider(
            feature.replace(
                "_",
                " "
            ).title(),
            0.0,
            100.0,
            min(
                100.0,
                max(
                    0.0,
                    current
                )
            ),
            1.0
        )

    elif feature == "study_hours_week":

        new_value = st.slider(
            "Study Hours per Week",
            0.0,
            40.0,
            min(
                40.0,
                max(
                    0.0,
                    current
                )
            ),
            1.0
        )

    elif feature == "lms_active_days":

        new_value = st.slider(
            "LMS Active Days",
            0.0,
            31.0,
            min(
                31.0,
                max(
                    0.0,
                    current
                )
            ),
            1.0
        )

    else:

        new_value = st.number_input(
            feature.replace(
                "_",
                " "
            ).title(),
            value=current
        )

    changes[feature] = new_value


# ============================================================
# ANALYSIS
# ============================================================

result = what_if(
    model,
    student,
    changes,
    required_features
)

# ============================================================
# SCENARIO RECOMMENDATIONS
# ============================================================

scenario_student = student.copy()

for feature, value in changes.items():

    scenario_student[feature] = value


scenario_recommendations = generate_recommendations(
    scenario_student
)

original_risk = result[
    "original_prediction"
]

scenario_risk = result[
    "scenario_prediction"
]

original_prob = result[
    "original_probability"
]

scenario_prob = result[
    "scenario_probability"
]


# ============================================================
# BEFORE / AFTER
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Current Risk",
        str(original_risk)
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Risk": list(
                    original_prob.keys()
                ),
                "Probability (%)": [
                    round(
                        value * 100,
                        2
                    )
                    for value
                    in original_prob.values()
                ]
            }
        ),
        use_container_width=True,
        hide_index=True
    )


with col2:

    st.metric(
        "What-If Risk",
        str(scenario_risk)
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Risk": list(
                    scenario_prob.keys()
                ),
                "Probability (%)": [
                    round(
                        value * 100,
                        2
                    )
                    for value
                    in scenario_prob.values()
                ]
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DYNAMIC COMPARISON
# ============================================================

comparison = pd.DataFrame(
    {
        "Risk": list(
            original_prob.keys()
        ),
        "Current": [
            value * 100
            for value
            in original_prob.values()
        ],
        "What-If": [
            scenario_prob[risk] * 100
            for risk
            in original_prob
        ]
    }
)


fig = go.Figure()

fig.add_bar(
    name="Current",
    x=comparison["Risk"],
    y=comparison["Current"]
)

fig.add_bar(
    name="What-If",
    x=comparison["Risk"],
    y=comparison["What-If"]
)

fig.update_layout(
    barmode="group",
    title="Current vs Simulated Risk",
    yaxis_title="Probability (%)",
    template="plotly_white",
    height=450
)


st.plotly_chart(
    fig,
    use_container_width=True
)


if str(original_risk) != str(
    scenario_risk
):

    st.success(
        f"Estimated model prediction changes from **{original_risk} → {scenario_risk}**."
    )

else:

    st.info(
        f"The estimated model prediction remains **{original_risk}**."
    )

# ============================================================
# PERSONALIZED LEARNING ACTIONS
# ============================================================

st.divider()

st.markdown(
    "### 💡 Personalized Learning Actions"
)

for recommendation in scenario_recommendations:

    st.markdown(
        f"🟢 {recommendation}"
    )

    
st.caption(
    "What-if analysis is a model simulation, not a causal prediction."
)