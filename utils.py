import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "Age",
    "Total_Reels_Watched",
    "Coffee_Consumed_Per_Day",
    "Focus_Sessions_Count",
    "Study_Hours",
    "Is_Late_Night",
    "Device_Type",
]

TARGET_COLUMN = "Brainrot_Stage"

STAGE_ORDER = ["Healthy", "Casual", "Advanced", "Critical"]

STAGE_COLORS = {
    "Healthy": "#22C55E",
    "Casual": "#EAB308",
    "Advanced": "#F97316",
    "Critical": "#F43F5E",
}

STAGE_ICONS = {
    "Healthy": "",
    "Casual": "",
    "Advanced": "",
    "Critical": "",
}

STAGE_LABELS_AR = {
    "Healthy": "Healthy",
    "Casual": "Casual",
    "Advanced": "Advanced",
    "Critical": "Critical",
}

STAGE_RECOMMENDATIONS_AR = {
    "Healthy": "Keep up this healthy pattern. You have an excellent balance between screen time, focus sessions, and study hours. Maintain your current routine.",
    "Casual": "Distraction is still at a mild stage. Try gradually reducing your daily short-form video count and add one extra deep-focus session per week.",
    "Advanced": "Digital distraction is starting to noticeably affect your productivity. Consider setting a daily screen-time limit for short-form videos, increasing your deep-focus sessions, and reducing caffeine intake after 6 PM.",
    "Critical": "This is a critical stage that requires immediate attention: reduce short-form video consumption, especially before bed, try the Pomodoro technique to increase focus sessions, and avoid using your phone after midnight. If you feel this is affecting your daily life, it may help to speak with a specialist.",
}


def load_raw_data(path: str) -> pd.DataFrame:
    """Loads raw data from a specified CSV file path."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs missing value imputation across demographic and behavioral features 
    to preserve overall sample size.
    """
    df = df.copy()

    # Handle Age imputation or range calculation if Min_Age / Max_Age present
    if "Age" not in df.columns:
        if "Min_Age" in df.columns and "Max_Age" in df.columns:
            df["Age"] = (df["Min_Age"] + df["Max_Age"]) / 2.0
        elif "min_age" in df.columns and "max_age" in df.columns:
            df["Age"] = (df["min_age"] + df["max_age"]) / 2.0

    if "Age" in df.columns and df["Age"].isnull().any():
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Device_Type" in df.columns and df["Device_Type"].isnull().any():
        df["Device_Type"] = df["Device_Type"].fillna(df["Device_Type"].mode()[0])

    for col in ["Age_Group", "Region", "Username"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    for col in ["Is_Smoker", "Base_Focus_Level"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts target features and performs one-hot encoding on categorical attributes."""
    X = df[FEATURE_COLUMNS].copy()
    X = pd.get_dummies(X, columns=["Device_Type"], prefix="Device")
    return X


def align_columns(X: pd.DataFrame, trained_columns: list) -> pd.DataFrame:
    """Aligns production/inference dataframe schema with training metadata columns."""
    return X.reindex(columns=trained_columns, fill_value=0)


def get_dark_chart_layout(title="", height=380):
    """Returns a standardized Plotly dark theme layout dictionary."""
    return dict(
        title=dict(text=title, font=dict(color="#E2E8F0", size=15, family="Inter, sans-serif")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=30),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#94A3B8"),
            title=dict(font=dict(color="#E2E8F0", size=13))
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#94A3B8"),
            title=dict(font=dict(color="#E2E8F0", size=13))
        ),
        height=height,
    )


def process_batch_predictions(input_df: pd.DataFrame, model, scaler, trained_columns: list, uses_scaled: bool) -> pd.DataFrame:
    """Processes bulk student records and generates stage predictions and confidence scores."""
    df_clean = clean_data(input_df)
    
    # Check for missing required columns
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {', '.join(missing_cols)}")

    X_features = prepare_features(df_clean)
    X_aligned = align_columns(X_features, trained_columns)

    if uses_scaled:
        X_model_input = scaler.transform(X_aligned)
    else:
        X_model_input = X_aligned

    predictions = model.predict(X_model_input)
    probabilities = model.predict_proba(X_model_input)

    output_df = input_df.copy()
    output_df["Predicted_Stage_Code"] = predictions
    output_df["Predicted_Brainrot_Stage"] = [STAGE_ORDER[p] for p in predictions]
    output_df["Prediction_Confidence"] = [np.max(prob) for prob in probabilities]

    for idx, stage in enumerate(STAGE_ORDER):
        output_df[f"Prob_{stage}"] = probabilities[:, idx]

    return output_df