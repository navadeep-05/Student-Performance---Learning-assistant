from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "student_risk_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"


# ============================================================
# MODEL / METADATA
# ============================================================

def load_model():
    """Load the trained student-risk ML pipeline."""
    return joblib.load(MODEL_PATH)


def load_metadata():
    """Load model metadata, including the exact feature list."""
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ============================================================
# DATA
# ============================================================

def normalize_columns(df):
    """Normalize dataframe column names for model compatibility."""
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def validate_data(df, required_features):
    """Return required model features missing from the dataframe."""
    return [
        feature
        for feature in required_features
        if feature not in df.columns
    ]


# ============================================================
# PREDICTION
# ============================================================

def predict_students(
    model,
    df,
    required_features
):
    """Predict risk and class probabilities for every student."""

    model_input = df[
        required_features
    ].copy()

    predictions = model.predict(
        model_input
    )

    probabilities = model.predict_proba(
        model_input
    )

    classes = list(
        model.classes_
    )

    result = df.copy()

    result["predicted_risk"] = predictions

    for index, class_name in enumerate(classes):

        result[
            f"probability_{str(class_name).lower()}"
        ] = probabilities[:, index]

    return result


def predict_one(
    model,
    student,
    required_features
):
    """Predict risk and probabilities for one student."""

    model_input = (
        student[
            required_features
        ]
        .to_frame()
        .T
    )

    prediction = model.predict(
        model_input
    )[0]

    probabilities = model.predict_proba(
        model_input
    )[0]

    classes = model.classes_

    probability_dict = {
        str(cls): float(prob)
        for cls, prob in zip(
            classes,
            probabilities
        )
    }

    return (
        prediction,
        probability_dict
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(student):
    """Generate rule-based learning-support actions from student indicators."""

    recommendations = []

    def number(feature):
        try:
            value = student[feature]

            if pd.isna(value):
                return None

            return float(value)

        except (KeyError, TypeError, ValueError):
            return None

    attendance = number(
        "attendance_pct"
    )

    assignment = number(
        "assignment_avg"
    )

    completion = number(
        "assignment_completion_pct"
    )

    quiz = number(
        "quiz_avg"
    )

    study = number(
        "study_hours_week"
    )

    participation = number(
        "participation_score"
    )

    lms = number(
        "lms_active_days"
    )

    gpa = number(
        "previous_gpa"
    )

    late = number(
        "late_assignments"
    )

    if attendance is not None and attendance < 75:
        recommendations.append(
            "Improve attendance and attend missed classes regularly."
        )

    if assignment is not None and assignment < 60:
        recommendations.append(
            "Review assignment feedback and focus on improving assignment performance."
        )

    if completion is not None and completion < 70:
        recommendations.append(
            "Complete pending assignments before the next assessment."
        )

    if quiz is not None and quiz < 60:
        recommendations.append(
            "Practice quizzes regularly and revise weak topics."
        )

    if study is not None and study < 10:
        recommendations.append(
            "Increase weekly study time using a structured study schedule."
        )

    if participation is not None and participation < 60:
        recommendations.append(
            "Increase classroom participation and discussion involvement."
        )

    if lms is not None and lms < 12:
        recommendations.append(
            "Use the learning platform more consistently."
        )

    if gpa is not None and gpa < 6:
        recommendations.append(
            "Review previously difficult subjects and consider academic support."
        )

    if late is not None and late >= 3:
        recommendations.append(
            "Plan assignment deadlines in advance to reduce late submissions."
        )

    
    if not recommendations:
        recommendations.append(
            "Maintain current learning habits and continue regular revision."
        )

    teacher_action = (
    "Discuss academic difficulties with the teacher "
    "and seek guidance on weak topics."
    )

    if teacher_action not in recommendations:
        recommendations.append(teacher_action)

    return recommendations[:6]


# ============================================================
# SHAP
# ============================================================

def shap_explanation(
    model,
    student,
    required_features
):
    """
    Calculate the top SHAP factors for the student's predicted class.

    The function supports common SHAP output formats produced by
    tree-based multiclass models.
    """

    if not SHAP_AVAILABLE:
        return None

    try:

        model_input = (
            student[
                required_features
            ]
            .to_frame()
            .T
        )

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        estimator = model.named_steps[
            "model"
        ]

        transformed = preprocessor.transform(
            model_input
        )

        if hasattr(
            transformed,
            "toarray"
        ):
            transformed = transformed.toarray()

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        explainer = shap.TreeExplainer(
            estimator
        )

        shap_values = explainer.shap_values(
            transformed
        )

        prediction = estimator.predict(
            transformed
        )[0]

        classes = list(
            estimator.classes_
        )

        class_index = classes.index(
            prediction
        )

        values = np.asarray(
            shap_values
        )

        if isinstance(
            shap_values,
            list
        ):

            values = np.asarray(
                shap_values[class_index][0]
            )

        elif values.ndim == 3:

            if values.shape[2] == len(classes):

                values = values[
                    0,
                    :,
                    class_index
                ]

            elif values.shape[0] == len(classes):

                values = values[
                    class_index,
                    0,
                    :
                ]

        elif values.ndim == 2:

            values = values[0]

        else:

            return None

        explanation = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": values
            }
        )

        explanation["abs_shap"] = (
            explanation[
                "shap_value"
            ].abs()
        )

        explanation = (
            explanation
            .sort_values(
                "abs_shap",
                ascending=False
            )
            .head(10)
            .reset_index(drop=True)
        )

        return explanation

    except Exception:
        return None


# ============================================================
# WHAT-IF
# ============================================================

def what_if(
    model,
    student,
    changes,
    required_features
):
    """Compare the original prediction with a hypothetical scenario."""

    original = student.copy()
    scenario = student.copy()

    for feature, value in changes.items():
        scenario[feature] = value

    original_prediction, original_probability = (
        predict_one(
            model,
            original,
            required_features
        )
    )

    scenario_prediction, scenario_probability = (
        predict_one(
            model,
            scenario,
            required_features
        )
    )

    return {
        "original_prediction": original_prediction,
        "original_probability": original_probability,
        "scenario_prediction": scenario_prediction,
        "scenario_probability": scenario_probability
    }


# ============================================================
# GENAI CONFIGURATION
# ============================================================
def _get_secret(name):
    """Read a Streamlit secret safely, then fall back to an environment variable."""
    try:
        import streamlit as st
        value = st.secrets.get(name, None)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name)

# Gemini 2.5 Flash currently has a free API tier for text
# generation. Keep the key outside source code.
GENAI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-1.5-flash"
)

# Never send these fields to the LLM as learning indicators.
# They are identifiers, target/outcome fields, or excluded
# demographic information.
GENAI_EXCLUDED_FIELDS = {
    "student_id",
    "final_score",
    "performance_level",
    "risk_level",
    "gender"
}



def get_genai_client():
    """
    Create a Gemini client using GEMINI_API_KEY.

    Returns:
        Gemini client, or None when the SDK/key is unavailable.
    """

    api_key = _get_secret(
        "GEMINI_API_KEY"
    )

    if not api_key:
        return None

    try:

        from google import genai

        return genai.Client(
            api_key=api_key
        )

    except ImportError:
        return None


def genai_is_configured():
    """Return True when the Gemini SDK and API key are available."""

    api_key = _get_secret(
        "GEMINI_API_KEY"
    )

    if not api_key:
        return False

    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================
# AI PROMPT
# ============================================================

def _safe_student_data(student):
    """
    Build the student context sent to Gemini.

    Only model-relevant learning indicators are included.
    Identifiers, outcomes, and gender are excluded.
    """

    data = {}

    for key, value in student.to_dict().items():

        key = str(key).strip().lower()

        if key in GENAI_EXCLUDED_FIELDS:
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        if isinstance(
            value,
            (np.integer, np.floating)
        ):
            value = float(value)

        data[key] = value

    return data


def create_ai_prompt(
    student,
    risk,
    probabilities,
    recommendations,
    shap_df=None
):
    """Create the structured context used by the real Gemini LLM."""

    factors = []

    if (
        shap_df is not None
        and not shap_df.empty
    ):

        for _, row in (
            shap_df.head(5).iterrows()
        ):

            factors.append(
                {
                    "feature": readable_feature_name(
                        row["feature"]
                    ),
                    "shap_contribution": round(
                        float(
                            row["shap_value"]
                        ),
                        4
                    )
                }
            )

    student_data = _safe_student_data(
        student
    )

    return f"""
You are an educational learning-support assistant.

Your task is to explain a machine-learning student-risk prediction
to a faculty member in simple, concise and professional language.

PREDICTED RISK:
{risk}

RISK PROBABILITIES:
{probabilities}

STUDENT LEARNING INDICATORS:
{student_data}

TOP SHAP MODEL FACTORS:
{factors}

PERSONALIZED LEARNING ACTIONS:
{recommendations}

IMPORTANT RULES:
- Explain the prediction, but do not claim that any feature causes the outcome.
- SHAP values describe model influence, not causation.
- Do not mention gender, student identity, final score, performance level or risk_level as input factors.
- Do not invent missing student information.
- Do not make sensitive personal claims.
- Keep the response concise and faculty-friendly.

Return exactly this structure:

RISK SUMMARY:
Write 1-2 short sentences.

KEY FACTORS:
- Factor 1
- Factor 2
- Factor 3

RECOMMENDED FOCUS:
- Action 1
- Action 2

FACULTY NOTE:
One short sentence reminding the reader that this is a decision-support prediction.
"""


# ============================================================
# REAL GENAI EXPLANATION
# ============================================================

def generate_genai_explanation(
    student,
    risk,
    probabilities,
    recommendations,
    shap_df=None
):
    """
    Generate the student's explanation using the real Gemini API.

    Returns:
        Generated text on success.
        None if the API key/SDK is unavailable.
        A GENAI_ERROR::... string if the API call fails.
    """

    client = get_genai_client()

    if client is None:
        return None

    prompt = create_ai_prompt(
        student=student,
        risk=risk,
        probabilities=probabilities,
        recommendations=recommendations,
        shap_df=shap_df
    )

    try:

        response = client.models.generate_content(
            model=GENAI_MODEL,
            contents=prompt
        )

        text = getattr(
            response,
            "text",
            None
        )

        if not text:
            return (
                "GENAI_ERROR::"
                "Gemini returned an empty response."
            )

        return text.strip()

    except Exception as error:

        return (
            "GENAI_ERROR::"
            + str(error)
        )


# ============================================================
# UTILITY
# ============================================================

def readable_feature_name(name):
    """Convert model pipeline feature names into readable UI labels."""

    name = str(name)

    for prefix in [
        "num__",
        "cat__"
    ]:

        if name.startswith(prefix):

            name = name[
                len(prefix):
            ]

    return (
        name
        .replace(
            "_",
            " "
        )
        .title()
    )