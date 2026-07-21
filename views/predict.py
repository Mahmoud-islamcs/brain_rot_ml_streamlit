import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import (
    prepare_features,
    align_columns,
    get_dark_chart_layout,
    STAGE_ORDER,
    STAGE_COLORS,
    STAGE_LABELS_AR,
    STAGE_RECOMMENDATIONS_AR,
)


def render_predict_page(df, model, scaler, trained_columns, uses_scaled, best_model_name):
    st.title("Mental Health & Wellbeing Predictor")
    st.markdown(
        "Enter student daily habits below to predict current digital distraction stage "
        "(Brain Rot Stage), using our trained machine learning model."
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Input Student Behavioral Attributes")

    col1, col2 = st.columns(2)

    with col1:
        reels = st.slider("Daily Short-Form Videos Watched (Reels/Shorts/TikTok)", 0, 270, 60)
        coffee = st.slider("Daily Caffeinated Drinks Consumed", 0, 10, 2)
        device = st.selectbox("Primary Device Used", ["Smartphone", "Tablet", "PC"])

        q_col, a_col = st.columns([2, 1])
        with q_col:
            st.write("")
            st.write("Do you use your phone after midnight?")
        with a_col:
            late_night = st.radio("", ["Yes", "No"], horizontal=True, label_visibility="collapsed")

    with col2:
        study_hours = st.slider("Daily Effective Study/Productivity Hours", 0.0, 11.0, 4.0, step=0.5)
        focus_sessions = st.slider("Daily Deep Focus Sessions (Uninterrupted)", 0, 11, 4)

        # Replacing single age input with dual-value Range Slider
        age_range = st.slider("Age Range (Min Age to Max Age)", 8, 25, (18, 22), step=1)
        min_age, max_age = age_range
        effective_age = (min_age + max_age) / 2.0

        st.info(
            f"Selected Age Range: {min_age} - {max_age} years. "
            f"Model Inference Age (Calculated Average): {effective_age:.1f} years."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Metric Containers to highlight key inputs before prediction
    st.markdown("<h4 style='color:#E2E8F0; margin-top:10px;'>Input Parameters Overview</h4>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{effective_age:.1f}</div>
                <div class='metric-label'>Inference Age</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{reels}</div>
                <div class='metric-label'>Reels / Day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{focus_sessions}</div>
                <div class='metric-label'>Focus Sessions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{study_hours}h</div>
                <div class='metric-label'>Study Hours</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col5:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{coffee}</div>
                <div class='metric-label'>Coffee / Day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "Execute AI Prediction",
        type="primary",
        use_container_width=True,
        key="btn_predict_main",
    )

    if predict_clicked:
        with st.spinner("Analyzing student behavioral metrics..."):
            time.sleep(1.0)
            input_df = pd.DataFrame(
                [
                    {
                        "Age": effective_age,
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
            label_ar = STAGE_LABELS_AR[pred_stage]
            recommendation = STAGE_RECOMMENDATIONS_AR[pred_stage]
            confidence = proba[pred_int] * 100

        # Structured Output Section
        st.markdown("---")
        st.markdown("<h3 style='color:#E2E8F0;'>Prediction Results & Assessment</h3>", unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown(
                f"""
                <div class='result-card' style='background: linear-gradient(135deg, {color}22, {color}08); border-color:{color}55;'>
                    <div class='result-title'>Predicted Digital Distraction Stage</div>
                    <div class='result-stage' style='color:{color};'>{label_ar}</div>
                    <div style='font-size: 14px; color: #94A3B8; margin-top: 6px;'>
                        Model Confidence: <b style='color:#FFFFFF;'>{confidence:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class='recommendation-box'>
                    <b style='color:{color}; font-size:16px;'>Targeted Recommendation:</b><br>
                    {recommendation}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown("#### Class Probability Distribution")
            proba_df = pd.DataFrame(
                {"Stage": STAGE_ORDER, "Probability": proba}
            ).sort_values("Probability", ascending=True)

            fig = go.Figure(
                go.Bar(
                    x=proba_df["Probability"],
                    y=proba_df["Stage"],
                    orientation="h",
                    marker_color=[STAGE_COLORS[s] for s in proba_df["Stage"]],
                    text=[f"{p:.1%}" for p in proba_df["Probability"]],
                    textposition="outside",
                )
            )

            layout_config = get_dark_chart_layout(height=280)
            layout_config["xaxis"]["range"] = [0, 1.15]
            layout_config["xaxis"]["tickformat"] = ".0%"
            fig.update_layout(**layout_config)

            st.plotly_chart(fig, use_container_width=True)
