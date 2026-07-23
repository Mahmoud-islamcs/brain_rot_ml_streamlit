import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_utils import (
    prepare_features,
    align_columns,
    get_dark_chart_layout,
    calculate_local_shap_values,
    generate_html_report,
    save_user_feedback,
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
        reels = st.slider("Daily Short-Form Videos Watched (Reels/Shorts/TikTok)", 0, 270, 60, key="pred_reels")
        coffee = st.slider("Daily Caffeinated Drinks Consumed", 0, 10, 2, key="pred_coffee")
        device = st.selectbox("Primary Device Used", ["Smartphone", "Tablet", "PC"], key="pred_device")

        q_col, a_col = st.columns([2, 1])
        with q_col:
            st.write("")
            st.write("Do you use your phone after midnight?")
        with a_col:
            late_night = st.radio("", ["Yes", "No"], horizontal=True, label_visibility="collapsed", key="pred_late")

    with col2:
        study_hours = st.slider("Daily Effective Study/Productivity Hours", 0.0, 11.0, 4.0, step=0.5, key="pred_study")
        focus_sessions = st.slider("Daily Deep Focus Sessions (Uninterrupted)", 0, 11, 4, key="pred_focus")

        age_range = st.slider("Age Range (Min Age to Max Age)", 8, 25, (18, 22), step=1, key="pred_age")
        min_age, max_age = age_range
        effective_age = (min_age + max_age) / 2.0

        st.info(
            f"Selected Age Range: {min_age} - {max_age} years. "
            f"Model Inference Age (Calculated Average): {effective_age:.1f} years."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Metric Containers
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

    if predict_clicked or "last_prediction" in st.session_state:
        if predict_clicked:
            with st.spinner("Analyzing student behavioral metrics..."):
                time.sleep(0.6)
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

                pred_int = int(model.predict(X_input_for_model)[0])
                proba = model.predict_proba(X_input_for_model)[0]
                pred_stage = STAGE_ORDER[pred_int]

                st.session_state.last_prediction = {
                    "input_dict": input_df.to_dict(orient="records")[0],
                    "X_input": X_input,
                    "pred_int": pred_int,
                    "pred_stage": pred_stage,
                    "proba": proba,
                }

        pred_data = st.session_state.last_prediction
        pred_stage = pred_data["pred_stage"]
        pred_int = pred_data["pred_int"]
        proba = pred_data["proba"]
        input_dict = pred_data["input_dict"]
        X_input = pred_data["X_input"]

        color = STAGE_COLORS[pred_stage]
        label_ar = STAGE_LABELS_AR[pred_stage]
        recommendation = STAGE_RECOMMENDATIONS_AR[pred_stage]
        confidence = proba[pred_int] * 100

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

        # 1. ADVANCED EXPLAINABLE AI (XAI) SECTION
        st.markdown("---")
        st.subheader("Advanced Local Model Explainability (XAI)")
        st.markdown(
            "Understand exactly which features increased or decreased distraction risk "
            "for this specific student prediction."
        )

        shap_df = calculate_local_shap_values(model, X_input, trained_columns, df, target_class_idx=pred_int)

        # Build Plotly Waterfall Chart
        fig_waterfall = go.Figure(
            go.Waterfall(
                name="Local Feature Impact",
                orientation="v",
                measure=["relative"] * len(shap_df),
                x=shap_df["display_feature"],
                y=shap_df["contribution"],
                connector={"line": {"color": "rgba(255,255,255,0.2)"}},
                increasing={"marker": {"color": "#F43F5E"}},
                decreasing={"marker": {"color": "#22C55E"}},
                text=[f"{v:+.3f}" for v in shap_df["contribution"]],
                textposition="outside",
            )
        )
        waterfall_layout = get_dark_chart_layout(title="Local Feature Attribution (SHAP Waterfall Plot)", height=400)
        waterfall_layout["xaxis"]["tickangle"] = -25
        fig_waterfall.update_layout(**waterfall_layout)

        st.plotly_chart(fig_waterfall, use_container_width=True)

        # 2. SCENARIO ANALYSIS ("WHAT-IF" TOOL)
        st.markdown("---")
        st.subheader("Interactive Scenario Analysis (What-If Simulator)")
        st.markdown(
            "Tweak habits dynamically to see how changes in daily behavior immediately alter the predicted distraction stage."
        )

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        wif_col1, wif_col2, wif_col3 = st.columns(3)

        with wif_col1:
            wif_reels = st.slider("Simulated Reels Watched", 0, 270, reels, key="wif_reels")
        with wif_col2:
            wif_focus = st.slider("Simulated Focus Sessions", 0, 11, focus_sessions, key="wif_focus")
        with wif_col3:
            wif_study = st.slider("Simulated Study Hours", 0.0, 11.0, float(study_hours), step=0.5, key="wif_study")

        wif_df = pd.DataFrame([
            {
                "Age": effective_age,
                "Total_Reels_Watched": wif_reels,
                "Coffee_Consumed_Per_Day": coffee,
                "Focus_Sessions_Count": wif_focus,
                "Study_Hours": wif_study,
                "Is_Late_Night": 1 if late_night == "Yes" else 0,
                "Device_Type": device,
            }
        ])

        wif_X = prepare_features(wif_df)
        wif_X = align_columns(wif_X, trained_columns)
        wif_X_model = scaler.transform(wif_X) if uses_scaled else wif_X

        wif_pred_int = model.predict(wif_X_model)[0]
        wif_proba = model.predict_proba(wif_X_model)[0]
        wif_stage = STAGE_ORDER[wif_pred_int]
        wif_color = STAGE_COLORS[wif_stage]

        wif_res1, wif_res2 = st.columns(2)
        with wif_res1:
            st.markdown(
                f"""
                <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:16px; text-align:center;'>
                    <div style='color:#94A3B8; font-size:13px;'>Original Input Stage</div>
                    <div style='font-size:24px; font-weight:800; color:{color};'>{pred_stage}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with wif_res2:
            st.markdown(
                f"""
                <div style='background:rgba(255,255,255,0.03); border:1px solid {wif_color}88; border-radius:12px; padding:16px; text-align:center;'>
                    <div style='color:#94A3B8; font-size:13px;'>Simulated Target Stage</div>
                    <div style='font-size:24px; font-weight:800; color:{wif_color};'>{wif_stage}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # 3. AUTOMATED REPORT EXPORT & FEEDBACK LOOP
        st.markdown("---")
        rep_col, feed_col = st.columns(2)

        with rep_col:
            st.subheader("Export Assessment Report")
            st.markdown("Download a printable HTML report of this student's assessment and XAI attributions.")

            html_report = generate_html_report(input_dict, pred_stage, confidence, recommendation, shap_df)
            st.download_button(
                label="Download Printable Assessment Report (HTML)",
                data=html_report,
                file_name=f"student_mental_health_report_{int(time.time())}.html",
                mime="text/html",
                key="btn_download_report",
            )

        with feed_col:
            st.subheader("Model Feedback & Ground Truth")
            st.markdown("Rate prediction accuracy or submit actual ground truth for future retraining.")

            user_rating = st.select_slider("Prediction Accuracy Rating", options=[1, 2, 3, 4, 5], value=5)
            actual_stage = st.selectbox("Ground Truth Stage (Actual)", STAGE_ORDER, index=int(pred_int))
            comments = st.text_input("Optional Comments", value="")

            if st.button("Submit Feedback", key="btn_submit_feedback"):
                feedback_payload = {
                    "inputs": input_dict,
                    "predicted_stage": pred_stage,
                    "confidence": float(confidence),
                    "user_rating": user_rating,
                    "actual_stage": actual_stage,
                    "comments": comments,
                }
                save_user_feedback(feedback_payload)
                st.success("Thank you! Feedback successfully logged for automated MLOps retraining.")
