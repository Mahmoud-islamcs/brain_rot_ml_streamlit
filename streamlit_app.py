import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time

from utils import (
    load_raw_data,
    clean_data,
    prepare_features,
    align_columns,
    STAGE_ORDER,
    STAGE_COLORS,
)
from translations import t, TRANSLATIONS, STAGE_LABELS, STAGE_RECOMMENDATIONS

# PAGE CONFIG
st.set_page_config(
    page_title="Brain Rot Analytics",
    page_icon="https://cdn-icons-png.flaticon.com/512/2103/2103633.png", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# LANGUAGE STATE
if "lang" not in st.session_state:
    st.session_state.lang = "en"


def toggle_language():
    st.session_state.lang = "arz" if st.session_state.lang == "en" else "en"


lang = st.session_state.lang

# GLASSMORPHISM DARK THEME CSS
st.markdown(
    """
    <style>

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Metric-style glass card */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #FFFFFF;
    }
    .metric-card .metric-label {
        font-size: 13px;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Result card */
    .result-card {
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(16px);
        margin: 16px 0 24px 0;
    }
    .result-title {
        font-size: 22px;
        color: #E5E7EB;
        margin-bottom: 6px;
    }
    .result-stage {
        font-size: 38px;
        font-weight: 900;
        margin: 8px 0;
    }
    .recommendation-box {
        background: rgba(255,255,255,0.04);
        border-left: 4px solid #7C5CFC;
        border-radius: 10px;
        padding: 16px 20px;
        color: #E2E8F0;
        line-height: 1.8;
    }

    /* Buttons */
    div.stButton > button {
        # background: linear-gradient(90deg, #7C5CFC 0%, #22D3EE 100%);
        # color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        width: 100%;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        opacity: 0.85;
        transform: scale(1.01);
    }

    /* Inputs */
    .stSlider, .stNumberInput, .stSelectbox, .stRadio {
        color: #F5F7FA;
    }
    
    /* Hide default streamlit footer/menu clutter (keep header) */
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# RTL SUPPORT (applied only when Arabic is active)
if lang == "arz":
    st.markdown(
        """
        <style>
        .main .block-container {
            direction: rtl;
            text-align: right;
        }
        section[data-testid="stSidebar"] {
            direction: rtl;
            text-align: right;
        }
        .glass-card, .result-card, .recommendation-box {
            direction: rtl;
            text-align: right;
        }
        .recommendation-box {
            border-left: none;
            border-right: 4px solid #7C5CFC;
        }
        div[data-testid="stMetricValue"], .metric-card {
            direction: rtl;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# CACHED LOADERS
@st.cache_data
def get_data():
    df = load_raw_data("data/BrainRot_Final_Dataset.csv")
    df = clean_data(df)
    return df

@st.cache_resource
def get_model_artifacts():
    model = joblib.load("model/brainrot_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    metadata = joblib.load("model/metadata.pkl")
    return model, scaler, metadata

df = get_data()
model, scaler, metadata = get_model_artifacts()
trained_columns = metadata["trained_columns"]
uses_scaled = metadata["uses_scaled"]
best_model_name = metadata["best_model_name"]

# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center;">
            <h2 style="margin-bottom:0;">{t('sidebar_title', lang)}</h2>
            <p style="margin-top:-4px; font-size:0.9rem; color:#9CA3AF;">
                {t('sidebar_subtitle', lang)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(t("lang_toggle_button", lang), use_container_width=True, key="btn_lang_toggle"):
        toggle_language()
        st.rerun()

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Predict"

    if st.button(t("nav_predict", lang), use_container_width=True):
        st.session_state.page = "Predict"

    if st.button(t("nav_insights", lang), use_container_width=True):
        st.session_state.page = "Insights"

    if st.button(t("nav_about", lang), use_container_width=True):
        st.session_state.page = "About"

    page = st.session_state.page

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    with st.container():
        st.write(f"**{t('sidebar_model_label', lang)}:** {best_model_name}")
        st.write(f"**{t('sidebar_records_label', lang)}:** {len(df):,}")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown(t("made_by", lang), unsafe_allow_html=True)


# PAGE 1: PREDICT
if page == "Predict":
    st.title(t("predict_title", lang))
    st.markdown(t("predict_intro", lang))

    device_options = [
        t("device_smartphone", lang),
        t("device_tablet", lang),
        t("device_pc", lang),
    ]
    device_values = ["Smartphone", "Tablet", "PC"]

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        reels = st.slider(t("label_reels", lang), 0, 270, 60)
        coffee = st.slider(t("label_coffee", lang), 0, 10, 2)
        device_display = st.selectbox(t("label_device", lang), device_options)
        device = device_values[device_options.index(device_display)]
        q_col, a_col = st.columns([2, 1])
        with q_col:
            st.write("")
            st.write(t("label_late_night_question", lang))
        with a_col:
            late_night_options = [t("label_yes", lang), t("label_no", lang)]
            late_night_display = st.radio(
                "", late_night_options, horizontal=True, label_visibility="collapsed"
            )
            late_night = "Yes" if late_night_display == t("label_yes", lang) else "No"
    with col2:
        study_hours = st.slider(t("label_study_hours", lang), 0.0, 11.0, 4.0, step=0.5)
        focus_sessions = st.slider(t("label_focus_sessions", lang), 0, 11, 4)
        age = st.number_input(t("label_age", lang), min_value=8, max_value=25, value=20, step=1)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button(
        t("predict_button", lang),
        type="primary",
        use_container_width=True,
        key="btn_predict_main"
    )

    if predict_clicked:
        with st.spinner(t("spinner_text", lang)):
            time.sleep(1.5) 
            input_df = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "Total_Reels_Watched": reels,
                        "Coffee_Consumed_Per_Day": coffee,
                        "Focus_Sessions_Count": focus_sessions,
                        "Study_Hours": study_hours,
                        "Is_Late_Night": 1 if late_night == "Yes" else 0,
                        "Device_Type": device,
                    }
                ]
            )
            X_input = prepare_features(input_df)
            X_input = align_columns(X_input, trained_columns)

            if uses_scaled:
                X_input_for_model = scaler.transform(X_input)
            else:
                X_input_for_model = X_input

            pred_int = model.predict(X_input_for_model)[0]
            proba = model.predict_proba(X_input_for_model)[0]
            pred_stage = STAGE_ORDER[pred_int]

            color = STAGE_COLORS[pred_stage]
            stage_label = STAGE_LABELS[lang][pred_stage]
            recommendation = STAGE_RECOMMENDATIONS[lang][pred_stage]

        st.markdown(
            f"""
            <div class='result-card' style='background: linear-gradient(135deg, {color}22, {color}08); border-color:{color}55;'>
                <div class='result-title'>{t('result_title', lang)}</div>
                <div class='result-stage' style='color:{color};'>{stage_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class='recommendation-box'><b>{t('recommendation_label', lang)}:</b> {recommendation}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {t('proba_title', lang)}")

        proba_df = pd.DataFrame(
            {
                "StageKey": STAGE_ORDER,
                "Stage": [STAGE_LABELS[lang][s] for s in STAGE_ORDER],
                "Probability": proba,
            }
        ).sort_values("Probability", ascending=True)

        fig = go.Figure(
            go.Bar(
                x=proba_df["Probability"],
                y=proba_df["Stage"],
                orientation="h",
                marker_color=[STAGE_COLORS[s] for s in proba_df["StageKey"]],
                text=[f"{p:.1%}" for p in proba_df["Probability"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB",
            xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)


# PAGE 3: INSIGHTS
elif page == "Insights":
    st.title(t("insights_title", lang))
    st.markdown(f"{t('insights_subtitle', lang)} **{best_model_name}**")

    dark_layout = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E5E7EB",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {t('feature_importance_title', lang)}")
        try:
            fi_df = pd.read_csv("model/feature_importance.csv")
            fig = px.bar(fi_df, x="importance", y="feature", orientation="h",color="importance", color_continuous_scale="Purples")
            fig.update_layout(**dark_layout, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.info(t("feature_importance_missing", lang))

    with col2:
        st.markdown(f"#### {t('confusion_matrix_title', lang)}")
        try:
            st.image("model/confusion_matrix.png", use_container_width=True)
        except Exception:
            st.info(t("confusion_matrix_missing", lang))

    st.markdown(f"#### {t('classification_report_title', lang)}")
    try:
        with open("model/classification_report.txt") as f:
            st.code(f.read(), language="text")
    except FileNotFoundError:
        st.info(t("classification_report_missing", lang))

    st.markdown(f"#### {t('correlation_title', lang)}")
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=["ActivityID", "UserKey", "DateKey", "StateKey", "HabitKey"], errors="ignore"
    )
    corr = numeric_df.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title=t("correlation_heatmap_title", lang))
    fig.update_layout(**dark_layout, height=550)
    st.plotly_chart(fig, use_container_width=True)


# PAGE 4: ABOUT
else:
    st.title(t("about_title", lang))

    benefits_list = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get("about_benefits", [])
    benefits_html = "".join(f"<li>{item}</li>" for item in benefits_list)

    st.markdown(
        f"""
        <div class='glass-card'>

        <h3>{t('about_project_heading', lang)}</h3>
        <p style='color:#CBD5E1; line-height:1.9;'>
        {t('about_project_text', lang)}
        </p>

        <h3>{t('about_dataset_heading', lang)}</h3>
        <p style='color:#CBD5E1; line-height:1.9;'>
        {t('about_dataset_text', lang)}
        </p>

        <h3>{t('about_benefits_heading', lang)}</h3>
        <ul style='color:#CBD5E1; line-height:1.9; padding-left: 20px;'>
            {benefits_html}
        </ul>

        <h3>{t('about_committee_heading', lang)}</h3>
        <p style='color:#CBD5E1; line-height:1.9;'>
        <b>{t('about_committee_p1_heading', lang)}</b><br>
        {t('about_committee_p1', lang)}
        <br><br>
        <b>{t('about_committee_p2_heading', lang)}</b><br>
        {t('about_committee_p2', lang)}
        <br><br>
        <b>{t('about_committee_p3_heading', lang)}</b><br>
        {t('about_committee_p3', lang)}
        </p>

        <h3>{t('about_leakage_heading', lang)}</h3>
        <p style='color:#CBD5E1; line-height:1.9;'>
        {t('about_leakage_text', lang)}
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )