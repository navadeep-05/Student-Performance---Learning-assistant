import io

import pandas as pd
import streamlit as st

from core.engine import (
    load_model,
    load_metadata,
    normalize_columns,
    validate_data,
    predict_students
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Student Performance Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "required_features" not in st.session_state:
    st.session_state.required_features = None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    return load_model()


@st.cache_data
def get_metadata():

    return load_metadata()


try:

    model = get_model()
    metadata = get_metadata()

except Exception as error:

    st.error(
        "Unable to load the trained ML model."
    )

    st.exception(error)

    st.stop()


required_features = metadata.get(
    "features",
    []
)

st.session_state.required_features = (
    required_features
)

# ============================================================
# NAVIGATION
# ============================================================

pages = [

    st.Page(
        "pages/home.py",
        title="Home",
        icon="🏠",
        default=True
    ),

    st.Page(
        "pages/student_analysis.py",
        title="Student Analysis",
        icon="👤"
    ),

    st.Page(
        "pages/what_if.py",
        title="What-If Lab",
        icon="🔮"
    ),

    st.Page(
        "pages/ai_insights.py",
        title="AI Insights",
        icon="✨"
    ),

    st.Page(
        "pages/model_information.py",
        title="Model Information",
        icon="🤖"
    )
]


# ============================================================
# CREATE NAVIGATION
# ============================================================

pg = st.navigation(
    pages,
    position="hidden"
)


# ============================================================
# CUSTOM SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 📊 Dashboard"
    )

    st.divider()

    st.page_link(
        "pages/home.py",
        label="Home",
        icon="🏠"
    )

    st.page_link(
        "pages/student_analysis.py",
        label="Student Analysis",
        icon="👤"
    )

    st.page_link(
        "pages/what_if.py",
        label="What-If Lab",
        icon="🔮"
    )

    st.page_link(
        "pages/ai_insights.py",
        label="AI Insights",
        icon="✨"
    )

    st.page_link(
        "pages/model_information.py",
        label="Model Information",
        icon="🤖"
    )


# ============================================================
# RUN SELECTED PAGE
# ============================================================

pg.run()