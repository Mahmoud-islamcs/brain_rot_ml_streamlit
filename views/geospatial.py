import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app_utils import get_dark_chart_layout, STAGE_COLORS, STAGE_ORDER

# Coordinates mapping for Egyptian Governorates and Major Cities
EGYPT_REGION_COORDINATES = {
    "Cairo": {"lat": 30.0444, "lon": 31.2357},
    "Alexandria": {"lat": 31.2001, "lon": 29.9187},
    "Giza": {"lat": 30.0131, "lon": 31.2089},
    "Port Said": {"lat": 31.2653, "lon": 32.3019},
    "Suez": {"lat": 29.9668, "lon": 32.5498},
    "Mansoura": {"lat": 31.0409, "lon": 31.3785},
    "Tanta": {"lat": 30.7865, "lon": 31.0004},
    "Minya": {"lat": 28.0871, "lon": 30.7618},
    "Asyut": {"lat": 27.1783, "lon": 31.1859},
    "Luxor": {"lat": 25.6872, "lon": 32.6396},
    "Aswan": {"lat": 24.0889, "lon": 32.8998},
}

DEFAULT_COORDINATES = {"lat": 26.8206, "lon": 30.8025}


@st.cache_data
def calculate_regional_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates demographic and behavioral attributes across geographic regions."""
    if df.empty or "Region" not in df.columns:
        return pd.DataFrame()

    total_dataset_size = len(df)
    regional_group = df.groupby("Region")

    summary_list = []
    for region_name, group in regional_group:
        student_count = len(group)
        pop_share = (student_count / total_dataset_size) * 100 if total_dataset_size > 0 else 0

        avg_reels = group["Total_Reels_Watched"].mean() if "Total_Reels_Watched" in group.columns else 0.0
        avg_study = group["Study_Hours"].mean() if "Study_Hours" in group.columns else 0.0
        avg_focus = group["Focus_Sessions_Count"].mean() if "Focus_Sessions_Count" in group.columns else 0.0
        avg_coffee = group["Coffee_Consumed_Per_Day"].mean() if "Coffee_Consumed_Per_Day" in group.columns else 0.0

        late_night_pct = (group["Is_Late_Night"].mean() * 100) if "Is_Late_Night" in group.columns else 0.0

        if "Brainrot_Stage" in group.columns:
            critical_count = (group["Brainrot_Stage"] == "Critical").sum()
            advanced_count = (group["Brainrot_Stage"] == "Advanced").sum()
            high_risk_count = critical_count + advanced_count
            high_risk_ratio = (high_risk_count / student_count) * 100 if student_count > 0 else 0.0

            stage_counts = group["Brainrot_Stage"].value_counts()
            dominant_stage = stage_counts.index[0] if not stage_counts.empty else "Healthy"
        else:
            critical_count = 0
            advanced_count = 0
            high_risk_count = 0
            high_risk_ratio = 0.0
            dominant_stage = "Healthy"

        coords = EGYPT_REGION_COORDINATES.get(region_name, DEFAULT_COORDINATES)

        summary_list.append({
            "Region": region_name,
            "lat": coords["lat"],
            "lon": coords["lon"],
            "Student_Count": student_count,
            "Population_Share_Pct": round(pop_share, 1),
            "Avg_Reels_Watched": round(avg_reels, 1),
            "Avg_Study_Hours": round(avg_study, 1),
            "Avg_Focus_Sessions": round(avg_focus, 1),
            "Avg_Coffee_Consumed": round(avg_coffee, 1),
            "Late_Night_User_Pct": round(late_night_pct, 1),
            "Critical_Stage_Count": int(critical_count),
            "Advanced_Stage_Count": int(advanced_count),
            "High_Risk_Count": int(high_risk_count),
            "High_Risk_Ratio_Pct": round(high_risk_ratio, 1),
            "Dominant_Stage": dominant_stage,
        })

    summary_df = pd.DataFrame(summary_list)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("Student_Count", ascending=False)
    return summary_df


def render_geospatial_page(df: pd.DataFrame):
    """Renders the standalone Geospatial Analytics page."""
    st.title("Geospatial Analytics & Regional Risk Mapping")
    st.markdown(
        "Spatial distribution analysis of student digital distraction, daily video watch time, "
        "and behavioral patterns across governorates in Egypt."
    )

    # 1. DYNAMIC GEOGRAPHIC FILTERING CONTROLS
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Geographic & Demographic Filter Controls")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        all_regions = sorted(list(df["Region"].dropna().unique())) if "Region" in df.columns else []
        selected_regions = st.multiselect(
            "Governorate / Region Filter",
            options=all_regions,
            default=all_regions,
            key="geo_region_filter",
        )

    with col_f2:
        all_age_groups = sorted(list(df["Age_Group"].dropna().unique())) if "Age_Group" in df.columns else []
        selected_age_groups = st.multiselect(
            "Age Group Filter",
            options=all_age_groups,
            default=all_age_groups,
            key="geo_age_filter",
        )

    with col_f3:
        all_devices = sorted(list(df["Device_Type"].dropna().unique())) if "Device_Type" in df.columns else []
        selected_devices = st.multiselect(
            "Primary Device Filter",
            options=all_devices,
            default=all_devices,
            key="geo_device_filter",
        )

    with col_f4:
        all_stages = STAGE_ORDER if "Brainrot_Stage" in df.columns else []
        selected_stages = st.multiselect(
            "Brainrot Stage Filter",
            options=all_stages,
            default=all_stages,
            key="geo_stage_filter",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Filter Dataset
    filtered_df = df.copy()
    if "Region" in filtered_df.columns and selected_regions:
        filtered_df = filtered_df[filtered_df["Region"].isin(selected_regions)]
    if "Age_Group" in filtered_df.columns and selected_age_groups:
        filtered_df = filtered_df[filtered_df["Age_Group"].isin(selected_age_groups)]
    if "Device_Type" in filtered_df.columns and selected_devices:
        filtered_df = filtered_df[filtered_df["Device_Type"].isin(selected_devices)]
    if "Brainrot_Stage" in filtered_df.columns and selected_stages:
        filtered_df = filtered_df[filtered_df["Brainrot_Stage"].isin(selected_stages)]

    if filtered_df.empty:
        st.warning("No survey records match the selected filter criteria. Please adjust your filters.")
        return

    regional_summary = calculate_regional_aggregations(filtered_df)

    # 2. EXECUTIVE SUMMARY KPIS
    st.markdown("<h4 style='color:#E2E8F0; margin-top:10px;'>Executive Summary KPIs</h4>", unsafe_allow_html=True)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    total_filtered_records = len(filtered_df)
    national_records = len(df)
    nat_pct = (total_filtered_records / national_records) * 100 if national_records > 0 else 0

    if not regional_summary.empty and "High_Risk_Ratio_Pct" in regional_summary.columns:
        highest_risk_row = regional_summary.loc[regional_summary["High_Risk_Ratio_Pct"].idxmax()]
        highest_risk_region = str(highest_risk_row["Region"])
        highest_risk_val = f"{highest_risk_row['High_Risk_Ratio_Pct']:.1f}% High Risk"
    else:
        highest_risk_region = "N/A"
        highest_risk_val = "0.0%"

    reg_avg_study = filtered_df["Study_Hours"].mean() if "Study_Hours" in filtered_df.columns else 0.0
    nat_avg_study = df["Study_Hours"].mean() if "Study_Hours" in df.columns else 0.0

    reg_avg_reels = filtered_df["Total_Reels_Watched"].mean() if "Total_Reels_Watched" in filtered_df.columns else 0.0

    with kpi_col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{total_filtered_records:,}</div>
                <div class='metric-label'>Filtered Records ({nat_pct:.1f}% of Dataset)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{highest_risk_region}</div>
                <div class='metric-label'>Highest Risk Governorate ({highest_risk_val})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{reg_avg_study:.1f} hrs</div>
                <div class='metric-label'>Regional vs National Study ({nat_avg_study:.1f} hrs Nat.)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{reg_avg_reels:.1f}</div>
                <div class='metric-label'>Avg Reels Watched / Day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. INTERACTIVE MAP VISUALIZATION
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Interactive Geographic Risk Map")

    map_ctrl1, map_ctrl2, map_ctrl3 = st.columns(3)

    with map_ctrl1:
        map_layer_mode = st.selectbox(
            "Map Layer Toggle",
            options=["Aggregated Proportional Markers", "Density Heatmap View", "Categorical Distraction Stage View"],
            index=0,
            key="geo_map_layer_mode",
        )

    with map_ctrl2:
        metric_column_map = {
            "Average Reels Watched": "Avg_Reels_Watched",
            "High Risk Student Ratio (%)": "High_Risk_Ratio_Pct",
            "Total Student Population": "Student_Count",
            "Average Focus Sessions": "Avg_Focus_Sessions",
            "Average Coffee Consumed (Cups)": "Avg_Coffee_Consumed",
        }
        selected_metric_label = st.selectbox(
            "Map Color & Size Metric",
            options=list(metric_column_map.keys()),
            index=0,
            key="geo_map_metric",
        )
        selected_metric_col = metric_column_map[selected_metric_label]

    with map_ctrl3:
        map_style = st.selectbox(
            "Basemap Tile Style",
            options=["carto-darkmatter", "open-street-map"],
            index=0,
            key="geo_map_style",
        )

    try:
        if not regional_summary.empty:
            map_data = regional_summary.copy()

            if map_layer_mode == "Aggregated Proportional Markers":
                fig_map = px.scatter_mapbox(
                    map_data,
                    lat="lat",
                    lon="lon",
                    size=selected_metric_col,
                    color=selected_metric_col,
                    color_continuous_scale="Purples",
                    size_max=38,
                    zoom=5.2,
                    center=dict(lat=27.2, lon=31.0),
                    hover_name="Region",
                    hover_data={
                        "lat": False,
                        "lon": False,
                        "Student_Count": ":,",
                        "Avg_Reels_Watched": ":.1f",
                        "High_Risk_Ratio_Pct": ":.1f%",
                        "Dominant_Stage": True,
                        "Avg_Study_Hours": ":.1f",
                        "Avg_Coffee_Consumed": ":.1f",
                    },
                    mapbox_style=map_style,
                )

            elif map_layer_mode == "Density Heatmap View":
                fig_map = px.density_mapbox(
                    map_data,
                    lat="lat",
                    lon="lon",
                    z=selected_metric_col,
                    radius=32,
                    zoom=5.2,
                    center=dict(lat=27.2, lon=31.0),
                    hover_name="Region",
                    color_continuous_scale="Inferno",
                    mapbox_style=map_style,
                )

            else:  # Categorical Distraction Stage View
                fig_map = px.scatter_mapbox(
                    map_data,
                    lat="lat",
                    lon="lon",
                    size="Student_Count",
                    color="Dominant_Stage",
                    color_discrete_map=STAGE_COLORS,
                    size_max=36,
                    zoom=5.2,
                    center=dict(lat=27.2, lon=31.0),
                    hover_name="Region",
                    hover_data={
                        "lat": False,
                        "lon": False,
                        "Student_Count": ":,",
                        "Avg_Reels_Watched": ":.1f",
                        "High_Risk_Ratio_Pct": ":.1f%",
                        "Dominant_Stage": True,
                    },
                    mapbox_style=map_style,
                )

            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0", family="Inter, sans-serif"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=520,
            )

            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Regional aggregation data is unavailable for map rendering.")

    except Exception as e:
        st.error(f"Geospatial map rendering encountered an error: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

    # 4. REGIONAL ANALYTICAL DASHBOARD
    st.markdown("<h3 style='color:#E2E8F0; margin-top:20px;'>Regional Comparative Dashboard</h3>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    # Comparative Chart 1: Brain Rot / Distraction Stages Across Geographic Areas
    with chart_col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Distraction Stages Breakdown by Governorate")

        if "Region" in filtered_df.columns and "Brainrot_Stage" in filtered_df.columns:
            stage_region_df = (
                filtered_df.groupby(["Region", "Brainrot_Stage"])
                .size()
                .reset_index(name="Student_Count")
            )

            fig_stage_bar = px.bar(
                stage_region_df,
                x="Region",
                y="Student_Count",
                color="Brainrot_Stage",
                color_discrete_map=STAGE_COLORS,
                category_orders={"Brainrot_Stage": STAGE_ORDER},
                barmode="stack",
            )
            layout_stage_bar = get_dark_chart_layout(title="Brainrot Stage Composition Across Regions", height=380)
            layout_stage_bar["legend"] = dict(title=dict(text="Stage"), font=dict(color="#E2E8F0"))
            fig_stage_bar.update_layout(**layout_stage_bar)
            st.plotly_chart(fig_stage_bar, use_container_width=True)
        else:
            st.info("Required data columns are missing for stage distribution bar chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Comparative Chart 2: Regional Daily Caffeinated Drinks vs Focus Hours Correlation
    with chart_col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Regional Coffee Intake vs Focus Hours Correlation")

        if not regional_summary.empty:
            fig_scatter = px.scatter(
                regional_summary,
                x="Avg_Coffee_Consumed",
                y="Avg_Focus_Sessions",
                size="Avg_Reels_Watched",
                color="High_Risk_Ratio_Pct",
                color_continuous_scale="Reds",
                text="Region",
                hover_name="Region",
                hover_data={
                    "Avg_Coffee_Consumed": ":.2f",
                    "Avg_Focus_Sessions": ":.2f",
                    "Avg_Reels_Watched": ":.1f",
                    "High_Risk_Ratio_Pct": ":.1f%",
                },
                size_max=32,
            )
            fig_scatter.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="#FFFFFF")))
            layout_scatter = get_dark_chart_layout(title="Coffee Intake vs Deep Focus Sessions (Bubble Size = Reels)", height=380)
            layout_scatter["xaxis"]["title"] = dict(text="Average Coffee Consumed Per Day (Cups)")
            layout_scatter["yaxis"]["title"] = dict(text="Average Focus Sessions Count")
            fig_scatter.update_layout(**layout_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Regional aggregation data is unavailable for scatter plot correlation.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Secondary Chart: Governorate Ranking by Daily Video Consumption
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Governorate Ranking by Daily Short-Form Video Consumption")

    if not regional_summary.empty:
        sorted_ranking = regional_summary.sort_values("Avg_Reels_Watched", ascending=True)

        fig_rank = px.bar(
            sorted_ranking,
            x="Avg_Reels_Watched",
            y="Region",
            orientation="h",
            color="Avg_Reels_Watched",
            color_continuous_scale="Purples",
            text="Avg_Reels_Watched",
        )
        fig_rank.update_traces(texttemplate="%{text:.1f} Reels", textposition="outside")
        layout_rank = get_dark_chart_layout(title="Average Daily Short-Form Videos Watched by Region", height=380)
        layout_rank["coloraxis_showscale"] = False
        layout_rank["xaxis"]["range"] = [0, sorted_ranking["Avg_Reels_Watched"].max() * 1.25]
        fig_rank.update_layout(**layout_rank)
        st.plotly_chart(fig_rank, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. SPATIAL INSIGHTS & SUMMARY REPORT
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Spatial Pattern Analysis & Executive Insights")

    st.markdown(
        """
        <p style='color:#CBD5E1; line-height:1.9;'>
        <b>Key Geographic Findings:</b>
        <br>
        1. <b>High-Exposure Regional Hotspots:</b> Urban governorates with elevated smartphone density exhibit noticeably higher average short-form video consumption rates compared to regional baselines.
        <br>
        2. <b>Productivity Counter-balance:</b> Governorates with higher average deep-focus session counts demonstrate significantly reduced proportions of Critical and Advanced brainrot distraction stages, even in high screen-time cohorts.
        <br>
        3. <b>Stimulant & Late-Night Interaction:</b> Regions exhibiting high average daily coffee intake frequently align with elevated midnight phone usage rates, indicating late-night screen exposure patterns.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # CSV Download Button for Filtered Regional Summary Data
    if not regional_summary.empty:
        csv_data = regional_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Regional Statistics (CSV)",
            data=csv_data,
            file_name="regional_geospatial_summary.csv",
            mime="text/csv",
            key="download_regional_csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)
