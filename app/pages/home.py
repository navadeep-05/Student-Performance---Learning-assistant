import io

import pandas as pd
import streamlit as st
from pathlib import Path

from core.engine import (
    load_model,
    normalize_columns,
    validate_data,
    predict_students
)


# ============================================================
# HOME
# ============================================================

st.title(
    "🎓 AI-Powered Student Performance & Learning Assistant"
)

IMAGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "GEN AI banenr 2.png"
)

if IMAGE_PATH.exists():
    left, center, right = st.columns(
        [1, 5, 1]
    )

    with center:
        st.image(
            str(IMAGE_PATH),
            width=900
        )

# ============================================================
# INTRO
# ============================================================

st.markdown(
    """
    ### What is it?

    An intelligent system that analyzes student learning data and
    identifies performance patterns and potential academic risk.

    ### Why?

    To help faculty move from **reactive reporting** to **proactive
    academic support**.

    ### How does it work?

    **Student Data → ML Risk Prediction → SHAP Explanation → "
    "Personalized Recommendations → What-If Analysis → GenAI Explanation**
    """
)


st.divider()


# ============================================================
# QUICK FLOW
# ============================================================

st.markdown("### ⚙️ From Data to Insight")


flow = st.columns(6)

steps = [
    ("📂", "Data"),
    ("🤖", "Predict"),
    ("🔍", "Explain"),
    ("💡", "Recommend"),
    ("🔮", "What-If"),
    ("✨", "GenAI")
]

for column, (icon, label) in zip(flow, steps):

    with column:

        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding:12px 4px;
                border:1px solid rgba(128,128,128,0.25);
                border-radius:12px;
            ">
                <div style="font-size:28px;">{icon}</div>
                <b>{label}</b>
            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.markdown("### 📂 Start Your Analysis")

st.write(
    "Upload a student CSV to begin."
)


uploaded_file = st.file_uploader(
    "Upload Student Dataset",
    type=["csv"],
    help="Upload the cleaned student learning dataset."
)


if uploaded_file is not None:

    try:

        raw_data = uploaded_file.read()

        data = normalize_columns(
            pd.read_csv(
                io.BytesIO(raw_data)
            )
        )

        required_features = (
            st.session_state.required_features
        )

        missing = validate_data(
            data,
            required_features
        )

        if missing:

            st.error(
                "The uploaded dataset is missing required model features."
            )

            st.code(
                "\n".join(missing)
            )

            st.stop()


        model = load_model()


        with st.spinner(
            "Analyzing student learning data..."
        ):

            results = predict_students(
                model,
                data,
                required_features
            )


        if "student_id" not in results.columns:

            results.insert(
                0,
                "student_id",
                [
                    f"STU-{i + 1:05d}"
                    for i in range(len(results))
                ]
            )


        st.session_state.data = results

        st.session_state.uploaded_name = (
            uploaded_file.name
        )


        st.success(
            f"Successfully analyzed {len(results):,} students."
        )


    except Exception as error:

        st.error(
            "Unable to process the uploaded dataset."
        )

        st.exception(error)


# ============================================================
# DATA PREVIEW + SUMMARY
# ============================================================

if st.session_state.data is not None:

    data = st.session_state.data


    st.divider()

    st.markdown(
        "### 📊 Dataset Overview"
    )


    # --------------------------------------------------------
    # QUICK INFORMATION
    # --------------------------------------------------------

    total_students = len(data)

    total_features = len(data.columns)

    high = (
        data["predicted_risk"]
        .astype(str)
        .str.lower()
        .eq("high")
        .sum()
    )

    medium = (
        data["predicted_risk"]
        .astype(str)
        .str.lower()
        .eq("medium")
        .sum()
    )

    low = (
        data["predicted_risk"]
        .astype(str)
        .str.lower()
        .eq("low")
        .sum()
    )


    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Students",
        f"{total_students:,}"
    )

    c2.metric(
        "Features",
        f"{total_features:,}"
    )

    c3.metric(
        "🔴 High",
        f"{high:,}"
    )

    c4.metric(
        "🟡 Medium",
        f"{medium:,}"
    )

    c5.metric(
        "🟢 Low",
        f"{low:,}"
    )


    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "#### Dataset Snapshot"
    )

    info1, info2, info3, info4 = st.columns(4)


    with info1:

        st.write(
            "**Rows**"
        )

        st.write(
            f"{data.shape[0]:,}"
        )


    with info2:

        st.write(
            "**Columns**"
        )

        st.write(
            f"{data.shape[1]:,}"
        )


    with info3:

        st.write(
            "**Numeric Features**"
        )

        st.write(
            f"{data.select_dtypes(include='number').shape[1]:,}"
        )


    with info4:

        st.write(
            "**Missing Values**"
        )

        st.write(
            f"{int(data.isna().sum().sum()):,}"
        )


    # --------------------------------------------------------
    # FIRST FIVE ROWS
    # --------------------------------------------------------

    st.markdown(
        "#### Data Preview — First 5 Rows"
    )

    st.dataframe(
        data.head(5),
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "Use the sidebar to explore individual students, "
        "What-If scenarios, AI explanations and model information."
    )