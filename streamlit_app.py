# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
import streamlit as st

from utils import load_raw_data, clean_data, train_candidate_models
from views.predict import render_predict_page
from views.explorer import render_explorer_page
from views.insights import render_insights_page
from views.geospatial import render_geospatial_page
from views.batch import render_batch_page
from views.about import render_about_page
from views.sidebar import render_sidebar

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

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.04);
        border-radius: 8px 8px 0 0;
        color: #94A3B8;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(124, 92, 252, 0.2) !important;
        color: #FFFFFF !important;
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

@st.cache_resource
def get_candidate_models_cache(_df, trained_columns, _main_model, uses_scaled):
    models = train_candidate_models(_df, trained_columns)
    models["trained_columns"] = trained_columns
    models["main_model"] = _main_model
    models["uses_scaled"] = uses_scaled
    return models


df = get_data()
model, scaler, metadata = get_model_artifacts()
trained_columns = metadata["trained_columns"]
uses_scaled = metadata["uses_scaled"]
best_model_name = metadata["best_model_name"]

candidate_models_dict = get_candidate_models_cache(df, trained_columns, model, uses_scaled)


# RENDER EXECUTIVE SIDEBAR NAVIGATION
page = render_sidebar(df, best_model_name)


# ROUTING TO MODULAR PAGES
if page == "Predict":
    render_predict_page(df, model, scaler, trained_columns, uses_scaled, best_model_name)

elif page == "Dataset Explorer":
    render_explorer_page(df)

elif page == "Insights":
    render_insights_page(df, best_model_name, candidate_models_dict)

elif page == "Geospatial Analysis":
    render_geospatial_page(df)

elif page == "Batch Prediction":
    render_batch_page(model, scaler, trained_columns, uses_scaled)

elif page == "About":
    render_about_page(best_model_name, len(df))