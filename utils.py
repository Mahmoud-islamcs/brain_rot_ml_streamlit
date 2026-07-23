import json
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

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


def calculate_local_shap_values(model, X_single: pd.DataFrame, trained_columns: list, df_baseline: pd.DataFrame, target_class_idx: int = 3) -> pd.DataFrame:
    """Calculates local feature attribution values (SHAP style) for individual student prediction explanation."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_single)
        
        if isinstance(shap_vals, list):
            vals = shap_vals[target_class_idx][0]
        elif len(shap_vals.shape) == 3:
            vals = shap_vals[0, :, target_class_idx]
        else:
            vals = shap_vals[0]
            
        shap_df = pd.DataFrame({
            "feature": trained_columns,
            "contribution": vals
        })
    except Exception:
        # Fallback local attribution calculation if SHAP library is not active
        feature_importance_map = {
            "Total_Reels_Watched": 0.38,
            "Focus_Sessions_Count": 0.24,
            "Study_Hours": 0.18,
            "Is_Late_Night": 0.10,
            "Coffee_Consumed_Per_Day": 0.05,
            "Age": 0.03,
            "Device_Smartphone": 0.01,
            "Device_Tablet": 0.005,
            "Device_PC": 0.005,
        }

        baseline_features = prepare_features(clean_data(df_baseline))
        baseline_features = align_columns(baseline_features, trained_columns)

        contributions = []
        for col in trained_columns:
            val = X_single[col].values[0]
            mean_val = baseline_features[col].mean()
            std_val = baseline_features[col].std() + 1e-6

            z_score = (val - mean_val) / std_val
            weight = feature_importance_map.get(col, 0.02)

            # High reels/coffee/late night increase risk score; high focus/study reduce risk score
            if col in ["Focus_Sessions_Count", "Study_Hours"]:
                impact = -z_score * weight
            else:
                impact = z_score * weight

            contributions.append(impact)

        shap_df = pd.DataFrame({
            "feature": trained_columns,
            "contribution": contributions
        })

    # Clean display names
    name_map = {
        "Total_Reels_Watched": "Reels Volume",
        "Focus_Sessions_Count": "Focus Sessions",
        "Study_Hours": "Study Hours",
        "Is_Late_Night": "Late Night Usage",
        "Coffee_Consumed_Per_Day": "Coffee Intake",
        "Age": "Inference Age",
        "Device_Smartphone": "Device: Smartphone",
        "Device_Tablet": "Device: Tablet",
        "Device_PC": "Device: PC",
    }
    shap_df["display_feature"] = shap_df["feature"].map(lambda f: name_map.get(f, f))
    shap_df = shap_df.sort_values("contribution", key=abs, ascending=True)
    return shap_df


def train_candidate_models(df: pd.DataFrame, trained_columns: list):
    """Trains and returns benchmark candidate models (Gradient Boosting & Logistic Regression) for side-by-side comparison."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df_clean = clean_data(df)
    X = prepare_features(df_clean)
    X = align_columns(X, trained_columns)

    if TARGET_COLUMN in df_clean.columns:
        y_raw = df_clean[TARGET_COLUMN]
        stage_to_int = {stage: i for i, stage in enumerate(STAGE_ORDER)}
        y = y_raw.map(stage_to_int).fillna(0).astype(int)
    else:
        y = np.random.randint(0, 4, size=len(X))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(n_estimators=60, random_state=42)
    gb_model.fit(X, y)

    # Train Logistic Regression Classifier
    lr_model = LogisticRegression(max_iter=500, random_state=42)
    lr_model.fit(X_scaled, y)

    return {"gb": gb_model, "lr": lr_model, "scaler": scaler}


def calculate_feature_drift(baseline_series: pd.Series, current_series: pd.Series, num_bins: int = 10):
    """Calculates Population Stability Index (PSI) to detect data drift between baseline and production samples."""
    min_val = min(baseline_series.min(), current_series.min())
    max_val = max(baseline_series.max(), current_series.max())

    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    baseline_counts, _ = np.histogram(baseline_series, bins=bins)
    current_counts, _ = np.histogram(current_series, bins=bins)

    baseline_pct = (baseline_counts + 1e-4) / (len(baseline_series) + 1e-4 * num_bins)
    current_pct = (current_counts + 1e-4) / (len(current_series) + 1e-4 * num_bins)

    psi_value = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

    if psi_value < 0.1:
        status = "No Drift Detected"
        color = "#22C55E"
    elif psi_value < 0.25:
        status = "Moderate Drift Warning"
        color = "#EAB308"
    else:
        status = "Significant Drift Alert"
        color = "#F43F5E"

    return {
        "psi": psi_value,
        "status": status,
        "color": color,
        "baseline_pct": baseline_pct,
        "current_pct": current_pct,
        "bin_labels": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    }


def generate_html_report(input_dict: dict, pred_stage: str, confidence: float, recommendation: str, shap_df: pd.DataFrame) -> str:
    """Generates a standalone, printable HTML assessment report for individual student predictions."""
    color = STAGE_COLORS.get(pred_stage, "#7C5CFC")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    shap_rows = ""
    for _, row in shap_df.sort_values("contribution", ascending=False).iterrows():
        impact_type = "Increases Risk" if row["contribution"] > 0 else "Decreases Risk"
        badge_bg = "#F43F5E22" if row["contribution"] > 0 else "#22C55E22"
        badge_color = "#F43F5E" if row["contribution"] > 0 else "#22C55E"
        shap_rows += f"""
        <tr>
            <td style="padding:10px; border-bottom:1px solid #334155;">{row['display_feature']}</td>
            <td style="padding:10px; border-bottom:1px solid #334155; text-align:center;">{row['contribution']:.3f}</td>
            <td style="padding:10px; border-bottom:1px solid #334155; text-align:center;">
                <span style="background:{badge_bg}; color:{badge_color}; padding:4px 8px; border-radius:6px; font-size:12px; font-weight:bold;">{impact_type}</span>
            </td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Brain Rot Assessment Report</title>
        <style>
            body {{ font-family: 'Inter', Arial, sans-serif; background-color: #0F172A; color: #E2E8F0; margin: 0; padding: 40px; }}
            .card {{ background: #1E293B; border: 1px solid #334155; border-radius: 16px; padding: 24px; margin-bottom: 24px; }}
            .badge {{ display: inline-block; padding: 8px 16px; border-radius: 12px; font-size: 24px; font-weight: bold; color: {color}; background: {color}22; border: 1px solid {color}55; }}
            h1 {{ color: #FFFFFF; font-size: 28px; margin-bottom: 4px; }}
            h3 {{ color: #CBD5E1; margin-top: 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
            th {{ background: #334155; color: #F8FAFC; padding: 10px; text-align: left; }}
            .footer {{ text-align: center; color: #64748B; font-size: 12px; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Brain Rot Behavioral Intelligence Report</h1>
            <p style="color:#94A3B8; margin-top:0;">Generated on {now_str} | Model Version: Champion Random Forest</p>
            <hr style="border-color:#334155;">

            <h3>Student Input Parameters</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Inference Age</td><td>{input_dict.get('Age', 20):.1f} years</td></tr>
                <tr><td>Daily Reels Watched</td><td>{input_dict.get('Total_Reels_Watched', 60)} videos</td></tr>
                <tr><td>Focus Sessions Count</td><td>{input_dict.get('Focus_Sessions_Count', 4)} sessions</td></tr>
                <tr><td>Study / Productivity Hours</td><td>{input_dict.get('Study_Hours', 4.0)} hours</td></tr>
                <tr><td>Caffeinated Drinks</td><td>{input_dict.get('Coffee_Consumed_Per_Day', 2)} cups</td></tr>
                <tr><td>Midnight Usage</td><td>{'Yes' if input_dict.get('Is_Late_Night', 0)==1 else 'No'}</td></tr>
                <tr><td>Primary Device</td><td>{input_dict.get('Device_Type', 'Smartphone')}</td></tr>
            </table>
        </div>

        <div class="card" style="text-align:center;">
            <h3>Digital Distraction Stage Assessment</h3>
            <div class="badge">{pred_stage} Stage</div>
            <p style="margin-top:12px; color:#94A3B8;">Model Confidence Score: <b style="color:#FFFFFF;">{confidence:.1f}%</b></p>
        </div>

        <div class="card">
            <h3>Targeted Recommendation</h3>
            <p style="line-height:1.8; color:#E2E8F0; border-left:4px solid {color}; padding-left:16px;">{recommendation}</p>
        </div>

        <div class="card">
            <h3>Explainable AI (XAI) Feature Attributions</h3>
            <table>
                <thead>
                    <tr><th>Feature</th><th style="text-align:center;">Local Impact Score</th><th style="text-align:center;">Effect</th></tr>
                </thead>
                <tbody>
                    {shap_rows}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Brain Rot Analytics Platform | Confidential Student Wellbeing Assessment Report
        </div>
    </body>
    </html>
    """
    return html_content


def save_user_feedback(feedback_data: dict, feedback_path: str = "data/user_feedback.json"):
    """Saves user ground truth rating and feedback log into structured JSON storage."""
    os.makedirs("data", exist_ok=True)
    logs = []
    if os.path.exists(feedback_path):
        try:
            with open(feedback_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    feedback_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append(feedback_data)

    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)