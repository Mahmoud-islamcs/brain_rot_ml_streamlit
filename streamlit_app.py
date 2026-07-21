# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
import streamlit as st

from utils import load_raw_data, clean_data
from views.predict import render_predict_page
from views.explorer import render_explorer_page
from views.insights import render_insights_page
from views.batch import render_batch_page
from views.about import render_about_page

# PAGE CONFIG
st.set_page_config(
    page_title="Brain Rot Analytics",
    page_icon="https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# GLASSMORPHISM DARK THEME CSS
st.markdown(
    """
    <style>

    /* Main container background */
    .stApp {
        background-color: #0E1117;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Metric-style glass card */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.06);
    }
    .metric-card .metric-value {
        font-size: 26px;
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
        padding: 30px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(16px);
        margin: 16px 0 24px 0;
    }
    .result-title {
        font-size: 20px;
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
        padding: 18px 22px;
        color: #E2E8F0;
        line-height: 1.8;
    }

    /* Primary and Secondary Buttons */
    div.stButton > button {
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 10px 18px;
        width: 100%;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        border-color: #7C5CFC;
        color: #7C5CFC;
        transform: translateY(-1px);
    }

    /* Active navigation button visual indicator */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7C5CFC 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
    }

    /* Inputs styling */
    .stSlider, .stNumberInput, .stSelectbox, .stRadio, .stMultiSelect {
        color: #F5F7FA;
    }
    
    /* Footer cleanup */
    footer {visibility: hidden;}
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
        """
        <div style="text-align:center; padding: 10px 0;">
            <h2 style="margin-bottom:0; color:#FFFFFF;">BrainRot Analytics</h2>
            <p style="margin-top:2px; font-size:0.85rem; color:#9CA3AF;">
                AI-Powered Behavioral Intelligence Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Predict"

    pages_list = [
        ("Predict", "Predict"),
        ("Dataset Explorer", "Dataset Explorer"),
        ("Insights", "Insights"),
        ("Batch Prediction", "Batch Prediction"),
        ("About", "About"),
    ]

    for label, page_key in pages_list:
        is_active = (st.session_state.page == page_key)
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, use_container_width=True, type=btn_type, key=f"nav_{page_key}"):
            st.session_state.page = page_key

    page = st.session_state.page

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
    st.write(f"**Model:** {best_model_name}")
    st.write(f"**Dataset Size:** {len(df):,} records")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
    st.markdown("Developed by [Mahmoud Islam](https://www.linkedin.com/in/mahmoud-islam-analytics/)", unsafe_allow_html=True)


# ROUTING TO MODULAR PAGES
if page == "Predict":
    render_predict_page(df, model, scaler, trained_columns, uses_scaled, best_model_name)

elif page == "Dataset Explorer":
    render_explorer_page(df)

elif page == "Insights":
    render_insights_page(df, best_model_name)

elif page == "Batch Prediction":
    render_batch_page(model, scaler, trained_columns, uses_scaled)

elif page == "About":
    render_about_page(best_model_name, len(df))