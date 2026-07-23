import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from app_utils import get_dark_chart_layout, STAGE_COLORS, STAGE_ORDER


def render_explorer_page(df: pd.DataFrame):
    st.title("Dataset Explorer & Statistical Analytics")
    st.markdown(
        "Interactively explore distributions, behavior correlations, and summary statistics "
        "across the 5,000 Egyptian student survey dataset."
    )

    # Filtering controls container
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Interactive Data Filters")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        min_age_val = int(df["Age"].min()) if "Age" in df.columns else 8
        max_age_val = int(df["Age"].max()) if "Age" in df.columns else 25
        selected_age = st.slider(
            "Age Range Filter",
            min_age_val,
            max_age_val,
            (min_age_val, max_age_val),
            key="exp_age_slider",
        )

    with filter_col2:
        device_options = list(df["Device_Type"].unique()) if "Device_Type" in df.columns else []
        selected_devices = st.multiselect(
            "Device Type Filter",
            device_options,
            default=device_options,
            key="exp_device_multi",
        )

    with filter_col3:
        late_night_options = ["All", "Late Night Users", "Non-Late Night Users"]
        selected_late = st.selectbox("Midnight Phone Usage", late_night_options, index=0)

    with filter_col4:
        stage_col = "Brainrot_Stage" if "Brainrot_Stage" in df.columns else None
        stage_options = list(df[stage_col].unique()) if stage_col else []
        selected_stages = st.multiselect(
            "Brainrot Stage Filter",
            stage_options,
            default=stage_options,
            key="exp_stage_multi",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Filter application
    filtered_df = df.copy()

    if "Age" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["Age"] >= selected_age[0]) & (filtered_df["Age"] <= selected_age[1])
        ]

    if "Device_Type" in filtered_df.columns and selected_devices:
        filtered_df = filtered_df[filtered_df["Device_Type"].isin(selected_devices)]

    if "Is_Late_Night" in filtered_df.columns:
        if selected_late == "Late Night Users":
            filtered_df = filtered_df[filtered_df["Is_Late_Night"] == 1]
        elif selected_late == "Non-Late Night Users":
            filtered_df = filtered_df[filtered_df["Is_Late_Night"] == 0]

    if stage_col and selected_stages:
        filtered_df = filtered_df[filtered_df[stage_col].isin(selected_stages)]

    # Key Summary Metrics
    st.markdown("<h4 style='color:#E2E8F0; margin-top:10px;'>Filtered Sample Overview</h4>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{len(filtered_df):,}</div>
                <div class='metric-label'>Filtered Records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col2:
        avg_reels = filtered_df["Total_Reels_Watched"].mean() if "Total_Reels_Watched" in filtered_df.columns else 0
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{avg_reels:.1f}</div>
                <div class='metric-label'>Avg Reels Watched / Day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col3:
        avg_study = filtered_df["Study_Hours"].mean() if "Study_Hours" in filtered_df.columns else 0
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{avg_study:.1f}h</div>
                <div class='metric-label'>Avg Study Hours / Day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f_col4:
        avg_focus = filtered_df["Focus_Sessions_Count"].mean() if "Focus_Sessions_Count" in filtered_df.columns else 0
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{avg_focus:.1f}</div>
                <div class='metric-label'>Avg Focus Sessions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Statistical Summary Table
    st.subheader("Summary Statistics Table")
    numeric_cols = [
        col for col in ["Age", "Total_Reels_Watched", "Coffee_Consumed_Per_Day", "Focus_Sessions_Count", "Study_Hours"]
        if col in filtered_df.columns
    ]

    if numeric_cols and len(filtered_df) > 0:
        stats_df = filtered_df[numeric_cols].describe().T
        stats_df = stats_df[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        stats_df.columns = ["Mean", "Std Dev", "Min", "25% Quantile", "Median (50%)", "75% Quantile", "Max"]
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("No matching data available for statistical summary.")

    # Interactive Plots Section
    st.markdown("---")
    st.subheader("Interactive Variable Visualizations")

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.markdown("##### Distribution Plot (Histogram)")
        var_to_plot = st.selectbox(
            "Select Variable for Histogram",
            numeric_cols,
            index=0 if "Total_Reels_Watched" in numeric_cols else 0,
            key="hist_var",
        )
        color_by_stage = st.checkbox("Color by Brainrot Stage", value=True, key="hist_color")

        fig_hist = px.histogram(
            filtered_df,
            x=var_to_plot,
            color=stage_col if (color_by_stage and stage_col) else None,
            color_discrete_map=STAGE_COLORS if (color_by_stage and stage_col) else None,
            marginal="box",
            nbins=30,
            opacity=0.8,
            title=f"Histogram & Box Distribution: {var_to_plot}",
        )
        fig_hist.update_layout(**get_dark_chart_layout(height=420))
        st.plotly_chart(fig_hist, use_container_width=True)

    with plot_col2:
        st.markdown("##### Comparative Boxplot")
        box_var = st.selectbox(
            "Select Metric for Boxplot Comparison",
            numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0,
            key="box_var",
        )
        group_var = st.selectbox(
            "Group By Feature",
            [col for col in ["Brainrot_Stage", "Device_Type", "Is_Late_Night"] if col in filtered_df.columns],
            index=0,
            key="box_group",
        )

        fig_box = px.box(
            filtered_df,
            x=group_var,
            y=box_var,
            color=group_var if group_var == stage_col else None,
            color_discrete_map=STAGE_COLORS if group_var == stage_col else None,
            points="outliers",
            title=f"Boxplot: {box_var} by {group_var}",
        )
        fig_box.update_layout(**get_dark_chart_layout(height=420))
        st.plotly_chart(fig_box, use_container_width=True)

    # Raw Data Table View and Download
    st.markdown("---")
    st.subheader("Filtered Dataset View")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Dataset CSV",
        data=csv_data,
        file_name="filtered_brain_rot_dataset.csv",
        mime="text/csv",
    )
