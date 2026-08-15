import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. BASIC CONFIGURATION
# ============================================================

SEED = 42
N_STUDENTS = 8000

rng = np.random.default_rng(SEED)


# ============================================================
# 2. CREATE PROJECT DIRECTORIES
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LATENT STUDENT CHARACTERISTICS
# ============================================================
# These are hidden variables used ONLY to create realistic
# relationships between student attributes.
#
# They will NOT appear in the final dataset.

academic_ability = np.clip(
    rng.normal(0.52, 0.18, N_STUDENTS),
    0.05,
    0.95
)

engagement = np.clip(
    0.55 * academic_ability +
    rng.normal(0.28, 0.16, N_STUDENTS),
    0.03,
    0.97
)

consistency = np.clip(
    0.50 * academic_ability +
    0.35 * engagement +
    rng.normal(0.18, 0.12, N_STUDENTS),
    0.02,
    0.98
)


# ============================================================
# 4. STUDENT PROFILE
# ============================================================

student_id = [
    f"STU{100001 + i}"
    for i in range(N_STUDENTS)
]

age = rng.integers(
    18,
    25,
    N_STUDENTS
)

gender = rng.choice(
    ["Male", "Female", "Other"],
    N_STUDENTS,
    p=[0.50, 0.48, 0.02]
)

department = rng.choice(
    ["CSE", "AI&DS", "ECE", "EEE", "MECH", "CIVIL", "IT"],
    N_STUDENTS,
    p=[0.22, 0.18, 0.14, 0.12, 0.10, 0.10, 0.14]
)

semester = rng.integers(
    2,
    9,
    N_STUDENTS
)


# ============================================================
# 5. LEARNING FEATURES
# ============================================================

attendance_pct = np.clip(
    42
    + 43 * consistency
    + 8 * engagement
    + rng.normal(0, 7, N_STUDENTS),
    35,
    100
)

study_hours_week = np.clip(
    3
    + 11 * academic_ability
    + 7 * consistency
    + rng.normal(0, 3.2, N_STUDENTS),
    1,
    35
)

assignment_avg = np.clip(
    30
    + 45 * academic_ability
    + 16 * consistency
    + rng.normal(0, 8, N_STUDENTS),
    20,
    100
)

assignment_completion_pct = np.clip(
    35
    + 45 * consistency
    + 12 * engagement
    + rng.normal(0, 8, N_STUDENTS),
    20,
    100
)

quiz_avg = np.clip(
    25
    + 50 * academic_ability
    + 15 * engagement
    + rng.normal(0, 9, N_STUDENTS),
    15,
    100
)

quiz_attempts = np.clip(
    np.round(
        3
        + 10 * engagement
        + 5 * consistency
        + rng.normal(0, 2, N_STUDENTS)
    ),
    1,
    20
).astype(int)

participation_score = np.clip(
    25
    + 45 * engagement
    + 20 * consistency
    + rng.normal(0, 9, N_STUDENTS),
    10,
    100
)


# ============================================================
# 6. LMS / LEARNING ACTIVITY
# ============================================================

lms_login_count = np.clip(
    np.round(
        5
        + 24 * engagement
        + 10 * consistency
        + rng.normal(0, 5, N_STUDENTS)
    ),
    1,
    45
).astype(int)

lms_active_days = np.clip(
    np.round(
        3
        + 15 * engagement
        + 7 * consistency
        + rng.normal(0, 3, N_STUDENTS)
    ),
    1,
    30
).astype(int)

resources_accessed = np.clip(
    np.round(
        5
        + 35 * engagement
        + 20 * academic_ability
        + rng.normal(0, 9, N_STUDENTS)
    ),
    1,
    80
).astype(int)

video_completion_pct = np.clip(
    25
    + 45 * engagement
    + 20 * consistency
    + rng.normal(0, 10, N_STUDENTS),
    5,
    100
)


# ============================================================
# 7. PREVIOUS PERFORMANCE
# ============================================================

previous_gpa = np.clip(
    4
    + 5 * academic_ability
    + rng.normal(0, 0.65, N_STUDENTS),
    4,
    10
)

previous_failures = np.clip(
    np.round(
        2.2
        - 2.2 * academic_ability
        + rng.normal(0, 0.8, N_STUDENTS)
    ),
    0,
    4
).astype(int)


# ============================================================
# 8. BEHAVIORAL FEATURES
# ============================================================

late_assignments = np.clip(
    np.round(
        5.5
        - 4.8 * consistency
        + rng.normal(0, 1.6, N_STUDENTS)
    ),
    0,
    8
).astype(int)

discussion_participation = np.clip(
    np.round(
        1
        + 10 * engagement
        + 5 * consistency
        + rng.normal(0, 2.5, N_STUDENTS)
    ),
    0,
    20
)

support_requested = np.where(
    (attendance_pct < 65)
    | (quiz_avg < 50)
    | (assignment_avg < 50),

    rng.choice(
        ["Yes", "No"],
        N_STUDENTS,
        p=[0.65, 0.35]
    ),

    rng.choice(
        ["Yes", "No"],
        N_STUDENTS,
        p=[0.18, 0.82]
    )
)


# ============================================================
# 9. CURRENT / MIDTERM PERFORMANCE
# ============================================================

midterm_score = np.clip(
    0.24 * attendance_pct
    + 0.26 * assignment_avg
    + 0.30 * quiz_avg
    + 0.08 * participation_score
    + 0.12 * (previous_gpa * 10)
    + rng.normal(0, 6.5, N_STUDENTS),
    20,
    100
)


# ============================================================
# 10. FUTURE PERFORMANCE
# ============================================================
# IMPORTANT:
# final_score represents the FUTURE outcome.
#
# This is what our future ML system should learn to predict
# from currently available student information.

final_score = np.clip(
    0.20 * attendance_pct
    + 0.18 * assignment_avg
    + 0.15 * assignment_completion_pct
    + 0.18 * quiz_avg
    + 0.08 * (study_hours_week / 35 * 100)
    + 0.06 * participation_score
    + 0.05 * (lms_active_days / 30 * 100)
    + 0.05 * video_completion_pct
    + 0.03 * (previous_gpa * 10)
    - 1.8 * previous_failures
    - 0.9 * late_assignments
    + 0.02 * midterm_score
    + rng.normal(0, 6.5, N_STUDENTS),
    0,
    100
)


# ============================================================
# 11. PERFORMANCE CATEGORY
# ============================================================

performance_level = np.select(
    [
        final_score < 50,
        final_score < 65,
        final_score < 80
    ],
    [
        "Poor",
        "Average",
        "Good"
    ],
    default="Excellent"
)


# ============================================================
# 12. RISK CATEGORY
# ============================================================

risk_level = np.select(
    [
        final_score < 50,
        final_score < 65
    ],
    [
        "High",
        "Medium"
    ],
    default="Low"
)


# ============================================================
# 13. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    "student_id": student_id,

    "age": age,

    "gender": gender,

    "department": department,

    "semester": semester,

    "attendance_pct": np.round(
        attendance_pct,
        1
    ),

    "assignment_avg": np.round(
        assignment_avg,
        1
    ),

    "assignment_completion_pct": np.round(
        assignment_completion_pct,
        1
    ),

    "quiz_avg": np.round(
        quiz_avg,
        1
    ),

    "quiz_attempts": quiz_attempts,

    "study_hours_week": np.round(
        study_hours_week,
        1
    ),

    "participation_score": np.round(
        participation_score,
        1
    ),

    "lms_login_count": lms_login_count,

    "lms_active_days": lms_active_days,

    "resources_accessed": resources_accessed,

    "video_completion_pct": np.round(
        video_completion_pct,
        1
    ),

    "previous_gpa": np.round(
        previous_gpa,
        2
    ),

    "previous_failures": previous_failures,

    "late_assignments": late_assignments,

    "discussion_participation": discussion_participation,

    "support_requested": support_requested,

    "midterm_score": np.round(
        midterm_score,
        1
    ),

    "final_score": np.round(
        final_score,
        1
    ),

    "performance_level": performance_level,

    "risk_level": risk_level
})


# ============================================================
# 14. SAVE CLEAN DATASET
# ============================================================

clean_path = PROCESSED_DIR / "student_learning_data_clean.csv"

df.to_csv(
    clean_path,
    index=False
)


# ============================================================
# 15. CREATE RAW DATASET
# ============================================================
# We intentionally add a small amount of missing data and
# duplicate rows so our preprocessing pipeline has something
# realistic to handle.

raw_df = df.copy()

missing_columns = [
    "attendance_pct",
    "assignment_avg",
    "quiz_avg",
    "study_hours_week",
    "participation_score",
    "lms_active_days",
    "video_completion_pct",
    "previous_gpa",
    "midterm_score"
]

for column in missing_columns:

    indexes = rng.choice(
        raw_df.index,
        size=int(0.01 * N_STUDENTS),
        replace=False
    )

    raw_df.loc[indexes, column] = np.nan


# Add 20 duplicate records
duplicates = raw_df.sample(
    20,
    random_state=42
)

raw_df = pd.concat(
    [raw_df, duplicates],
    ignore_index=True
)


raw_path = RAW_DIR / "student_data_raw.csv"

raw_df.to_csv(
    raw_path,
    index=False
)


# ============================================================
# 16. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SYNTHETIC STUDENT DATASET CREATED")
print("=" * 60)

print(f"\nClean records : {len(df):,}")
print(f"Raw records   : {len(raw_df):,}")
print(f"Columns       : {len(df.columns)}")

print("\nRisk Distribution:")
print(df["risk_level"].value_counts())

print("\nPerformance Distribution:")
print(df["performance_level"].value_counts())

print("\nClean Dataset:")
print(clean_path)

print("\nRaw Dataset:")
print(raw_path)

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)