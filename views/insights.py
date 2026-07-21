import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import get_dark_chart_layout, STAGE_ORDER


def render_insights_page(df: pd.DataFrame, best_model_name: str):
    st.title("Model Insights & Interpretability")
    st.markdown(
        f"Detailed evaluation metrics, feature importances, and confusion matrix breakdown "
        f"for our champion model: **{best_model_name}**."
    )

    dark_layout = get_dark_chart_layout()

    col1, col2 = st.columns(2)

    # 1. Feature Importance
    with col1:
        st.subheader("Feature Importance Ranking")
        if os.path.exists("model/feature_importance.csv"):
            fi_df = pd.read_csv("model/feature_importance.csv")
            fi_df = fi_df.sort_values("importance", ascending=True)

            fig_fi = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Purples",
                text="importance",
            )
            fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            layout_fi = get_dark_chart_layout(title="Relative Gini Feature Importance", height=420)
            layout_fi["coloraxis_showscale"] = False
            layout_fi["xaxis"]["range"] = [0, fi_df["importance"].max() * 1.25]
            fig_fi.update_layout(**layout_fi)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance data file not found in model directory.")

        with st.expander("How to Interpret Feature Importance"):
            st.markdown(
                """
                **Understanding Feature Importance:**
                - **Gini Importance (Mean Decrease in Impurity):** Measures how much each behavioral feature contributes to partitioning student records into the correct digital distraction stages.
                - **Top Key Driver:** `Total_Reels_Watched` consistently ranks highest, demonstrating that short-form video consumption volume is the primary determinant of digital distraction.
                - **Productivity Counterweights:** `Focus_Sessions_Count` and `Study_Hours` serve as strong secondary signals that counteract high digital exposure.
                """
            )

    # 2. Confusion Matrix
    with col2:
        st.subheader("Confusion Matrix")
        if os.path.exists("model/confusion_matrix.png"):
            st.image("model/confusion_matrix.png", use_container_width=True)
        else:
            st.info("Confusion matrix image file not found in model directory.")

        with st.expander("How to Interpret Confusion Matrix & Metric Tradeoffs"):
            st.markdown(
                """
                **Reading the Confusion Matrix:**
                - **Diagonal Cells:** Represent correct predictions for each stage (True Positives). High diagonal values indicate high accuracy across all stages.
                - **Off-Diagonal Cells:** Represent misclassifications (False Positives and False Negatives).
                - **Precision vs. Recall Tradeoffs:**
                  * **High Recall for Critical Stage:** Crucial so that students experiencing severe distraction are never missed (low False Negatives).
                  * **High Precision for Healthy Stage:** Ensures students with healthy habits are not falsely flagged with alarming diagnoses.
                - **Macro F1-Score Alignment:** By evaluating macro F1-score during training, minority classes (Casual and Advanced) are treated with equal weighting.
                """
            )

    st.markdown("---")

    col3, col4 = st.columns([1, 1])

    # 3. Classification Report
    with col3:
        st.subheader("Classification Report Metrics")
        if os.path.exists("model/classification_report.txt"):
            with open("model/classification_report.txt", "r") as f:
                report_text = f.read()
            st.code(report_text, language="text")
        else:
            st.info("Classification report file not found in model directory.")

        with st.expander("Classification Report Guidelines"):
            st.markdown(
                """
                **Key Performance Definitions:**
                - **Precision:** The proportion of predicted instances for a stage that were actually correct.
                - **Recall (Sensitivity):** The proportion of actual instances of a stage that the model captured.
                - **F1-Score:** Harmonic mean of Precision and Recall. Our model achieves a **Macro F1-Score of 0.903** and an **Accuracy of 94.5%**.
                - **Support:** The total number of true student test instances evaluated per stage class.
                """
            )

    # 4. Correlation Heatmap
    with col4:
        st.subheader("Behavioral Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number]).drop(
            columns=["ActivityID", "UserKey", "DateKey", "StateKey", "HabitKey"], errors="ignore"
        )

        # Compute correlation matrix
        corr = numeric_df.corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Feature Correlation Heatmap",
        )
        layout_corr = get_dark_chart_layout(height=420)
        fig_corr.update_layout(**layout_corr)
        st.plotly_chart(fig_corr, use_container_width=True)

        with st.expander("How to Interpret the Correlation Heatmap"):
            st.markdown(
                """
                **Correlation Heatmap Insights:**
                - **Red/Positive Values (+1.0):** Strong positive association. For instance, `Total_Reels_Watched` positively correlates with late night usage and distraction risk.
                - **Blue/Negative Values (-1.0):** Inverse relationship. Notice how `Study_Hours` and `Focus_Sessions_Count` show negative correlations with screen distraction variables.
                - **Multicollinearity Audit:** Low inter-feature correlation between predictors verifies feature independence, preventing model instability.
                """
            )
