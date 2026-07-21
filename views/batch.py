import io
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import (
    process_batch_predictions,
    get_dark_chart_layout,
    STAGE_COLORS,
    STAGE_ORDER,
    FEATURE_COLUMNS,
)


def render_batch_page(model, scaler, trained_columns: list, uses_scaled: bool):
    st.title("Batch Prediction Engine")
    st.markdown(
        "Upload a CSV file containing multiple student behavioral records to generate bulk "
        "digital distraction predictions and download the annotated dataset."
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Batch Dataset Upload")

    st.markdown(
        """
        **Required CSV Columns:**
        `Age` (or `Min_Age` and `Max_Age`), `Total_Reels_Watched`, `Coffee_Consumed_Per_Day`, 
        `Focus_Sessions_Count`, `Study_Hours`, `Is_Late_Night` (0 or 1), `Device_Type` (Smartphone/Tablet/PC).
        """
    )

    # Sample template download
    sample_df = pd.DataFrame(
        [
            {
                "Age": 20,
                "Total_Reels_Watched": 90,
                "Coffee_Consumed_Per_Day": 3,
                "Focus_Sessions_Count": 2,
                "Study_Hours": 3.0,
                "Is_Late_Night": 1,
                "Device_Type": "Smartphone",
            },
            {
                "Age": 22,
                "Total_Reels_Watched": 30,
                "Coffee_Consumed_Per_Day": 1,
                "Focus_Sessions_Count": 6,
                "Study_Hours": 6.5,
                "Is_Late_Night": 0,
                "Device_Type": "PC",
            },
            {
                "Min_Age": 18,
                "Max_Age": 24,
                "Total_Reels_Watched": 180,
                "Coffee_Consumed_Per_Day": 5,
                "Focus_Sessions_Count": 1,
                "Study_Hours": 1.5,
                "Is_Late_Night": 1,
                "Device_Type": "Smartphone",
            },
        ]
    )

    sample_csv = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Sample CSV Template",
        data=sample_csv,
        file_name="sample_student_batch_input.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file for batch processing", type=["csv"], key="batch_file_uploader"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded dataset with {len(input_df):,} rows and {len(input_df.columns)} columns.")

            st.markdown("##### Input Data Preview")
            st.dataframe(input_df.head(5), use_container_width=True)

            if st.button("Generate Batch Predictions", type="primary", use_container_width=True):
                with st.spinner("Processing batch predictions..."):
                    results_df = process_batch_predictions(
                        input_df, model, scaler, trained_columns, uses_scaled
                    )

                st.markdown("---")
                st.subheader("Batch Prediction Results")

                # Overview summary metrics
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)

                with b_col1:
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <div class='metric-value'>{len(results_df):,}</div>
                            <div class='metric-label'>Total Records Evaluated</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with b_col2:
                    critical_cnt = (results_df["Predicted_Brainrot_Stage"] == "Critical").sum()
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <div class='metric-value' style='color:#F43F5E;'>{critical_cnt:,}</div>
                            <div class='metric-label'>Critical Stage Cases</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with b_col3:
                    healthy_cnt = (results_df["Predicted_Brainrot_Stage"] == "Healthy").sum()
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <div class='metric-value' style='color:#22C55E;'>{healthy_cnt:,}</div>
                            <div class='metric-label'>Healthy Stage Cases</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with b_col4:
                    avg_conf = results_df["Prediction_Confidence"].mean() * 100
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <div class='metric-value'>{avg_conf:.1f}%</div>
                            <div class='metric-label'>Mean Confidence</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Batch distribution chart
                stage_counts = (
                    results_df["Predicted_Brainrot_Stage"]
                    .value_counts()
                    .reindex(STAGE_ORDER, fill_value=0)
                    .reset_index()
                )
                stage_counts.columns = ["Stage", "Count"]

                fig_batch = px.bar(
                    stage_counts,
                    x="Stage",
                    y="Count",
                    color="Stage",
                    color_discrete_map=STAGE_COLORS,
                    text="Count",
                    title="Batch Predicted Stage Breakdown",
                )
                fig_batch.update_traces(textposition="outside")
                fig_batch.update_layout(**get_dark_chart_layout(height=380))
                st.plotly_chart(fig_batch, use_container_width=True)

                # Results Dataframe Display
                st.subheader("Annotated Predictions Data Table")
                st.dataframe(results_df, use_container_width=True)

                # Export output CSV
                out_csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predicted Batch Results CSV",
                    data=out_csv,
                    file_name="batch_brain_rot_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing CSV batch file: {str(e)}")
