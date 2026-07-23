import pandas as pd
import streamlit as st


def render_sidebar(df: pd.DataFrame, best_model_name: str) -> str:
    """Renders the executive sidebar navigation, status badges, global controls, and branding footer."""
    with st.sidebar:
        # 1. ENHANCED PROJECT HEADER
        st.markdown(
            """
            <div style="padding: 10px 0 14px 0; text-align: center;">
                <h2 style="margin: 0; color: #FFFFFF; font-size: 1.45rem; font-weight: 800; letter-spacing: -0.5px;">
                    BrainRot Analytics
                </h2>
                <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94A3B8; font-weight: 500;">
                    Enterprise AI Behavioral Intelligence
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. MODEL STATUS & INFO BADGE
        dataset_size = len(df) if df is not None else 0
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px 14px; margin-bottom: 16px;">
                <div style="font-size: 0.7rem; color: #7C5CFC; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700; margin-bottom: 6px;">
                    Pipeline Status: Active
                </div>
                <div style="font-size: 0.82rem; color: #E2E8F0; margin-bottom: 3px;">
                    <span style="color: #94A3B8;">Champion Model:</span> <b style="color: #FFFFFF;">{best_model_name}</b>
                </div>
                <div style="font-size: 0.82rem; color: #E2E8F0;">
                    <span style="color: #94A3B8;">Dataset Size:</span> <b style="color: #FFFFFF;">{dataset_size:,} records</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 10px 0 14px 0;'>", unsafe_allow_html=True)

        # 3. CATEGORIZED NAVIGATION ARCHITECTURE
        if "page" not in st.session_state:
            st.session_state.page = "Predict"

        navigation_categories = {
            "Main Applications": [
                ("Predict", "Predict"),
                ("Batch Prediction", "Batch Prediction"),
            ],
            "Analytics & Insights": [
                ("Dataset Explorer", "Dataset Explorer"),
                ("Insights", "Insights"),
                ("Geospatial Analysis", "Geospatial Analysis"),
            ],
            "System Information": [
                ("About", "About"),
            ],
        }

        for category_name, page_items in navigation_categories.items():
            st.markdown(
                f"""
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #64748B; font-weight: 700; margin: 12px 0 6px 4px; letter-spacing: 1px;">
                    {category_name}
                </div>
                """,
                unsafe_allow_html=True,
            )

            for label, page_key in page_items:
                is_active = (st.session_state.page == page_key)
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, use_container_width=True, type=btn_type, key=f"nav_{page_key}"):
                    st.session_state.page = page_key

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 14px 0 10px 0;'>", unsafe_allow_html=True)

        # 4. GLOBAL CONTROL PANEL INTEGRATION
        with st.expander("Global Data Control Panel", expanded=False):
            st.markdown(
                "<p style='font-size: 0.78rem; color: #94A3B8; margin-bottom: 8px;'>"
                "Apply baseline filters across session state."
                "</p>",
                unsafe_allow_html=True,
            )

            age_options = ["All"] + (sorted(list(df["Age_Group"].dropna().unique())) if df is not None and "Age_Group" in df.columns else [])
            selected_global_age = st.selectbox(
                "Global Age Group",
                options=age_options,
                index=0,
                key="global_age_group_filter",
            )

            device_options = ["All"] + (sorted(list(df["Device_Type"].dropna().unique())) if df is not None and "Device_Type" in df.columns else [])
            selected_global_device = st.selectbox(
                "Global Primary Device",
                options=device_options,
                index=0,
                key="global_device_type_filter",
            )

            if st.button("Reset Global Filters", use_container_width=True, key="reset_global_filters_btn"):
                st.session_state.global_age_group_filter = "All"
                st.session_state.global_device_type_filter = "All"
                st.rerun()

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 14px 0;'>", unsafe_allow_html=True)

        # 5. STYLIZED FOOTER & BRANDING
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 12px; text-align: center;">
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 8px;">
                    Developed by <b style="color: #FFFFFF;">Mahmoud Islam</b>
                </div>
                <div style="display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;">
                    <a href="https://www.linkedin.com/in/mahmoud-islam-analytics/" target="_blank" style="font-size: 0.75rem; color: #7C5CFC; background: rgba(124,92,252,0.12); padding: 5px 12px; border-radius: 8px; text-decoration: none; border: 1px solid rgba(124,92,252,0.3); font-weight: 600;">
                        LinkedIn Profile
                    </a>
                    <a href="https://github.com/Mahmoud-islamcs/Neuro-Digital-Analytics" target="_blank" style="font-size: 0.75rem; color: #CBD5E1; background: rgba(255,255,255,0.05); padding: 5px 12px; border-radius: 8px; text-decoration: none; border: 1px solid rgba(255,255,255,0.15); font-weight: 600;">
                        GitHub Repo
                    </a>
                </div>
            </div>
            <div style="text-align: center; margin-top: 10px; font-size: 0.7rem; color: #475569; font-weight: 500;">
                v2.0 - Production Build | Enterprise Edition
            </div>
            """,
            unsafe_allow_html=True,
        )

        return st.session_state.page
