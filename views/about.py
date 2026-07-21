import streamlit as st


def render_about_page(best_model_name: str, total_records: int):
    st.title("About Brain Rot Analytics")
    st.markdown(
        "Comprehensive architectural overview, dataset specifications, methodology, "
        "and contact information for the Brain Rot Behavioral Intelligence project."
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("1. Model Architecture Overview")
    st.markdown(
        f"""
        <p style='color:#CBD5E1; line-height:1.9;'>
        <b>"Brain Rot Analytics"</b> is an AI-powered behavioral intelligence platform built as part of a 
        Data Science graduation project. It studies the relationship between daily digital habits—specifically 
        short-form video consumption—and student cognitive wellbeing.
        <br><br>
        <b>Champion Algorithm:</b> <code>{best_model_name}</code><br>
        <b>Evaluation Benchmark:</b> Evaluated across Logistic Regression, Random Forest, and XGBoost using 
        <code>RandomizedSearchCV</code> hyperparameter tuning.<br>
        <b>Primary Metric:</b> <b>Macro F1-Score of 0.903</b> (Overall Accuracy of <b>94.5%</b>). Macro F1-score 
        was selected to ensure robust classification performance across all minority distraction classes.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("2. Dataset Distribution & Imbalance Strategy")
    st.markdown(
        f"""
        <p style='color:#CBD5E1; line-height:1.9;'>
        The underlying dataset comprises <b>{total_records:,} student records</b> with 30 attributes collected across Egyptian student surveys.
        <br><br>
        <b>Missing Value Imputation:</b> Missing data across 51 rows (in Age, Region, Device Type) were imputed using median and mode strategies rather than row deletion, preserving sample size.
        <br>
        <b>Class Imbalance Handling:</b> Class distribution is imbalanced (Healthy: 61%, Critical: 18%, Advanced: 12.5%, Casual: 8.5%). Strict class weighting (<code>class_weight='balanced'</code>) and stratified split were applied during model training.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("3. Methodology & Data Leakage Prevention")
    st.markdown(
        """
        <p style='color:#CBD5E1; line-height:1.9;'>
        <b>Feature Engineering & Selection:</b> Only <b>7 core behavioral features</b> were passed to the prediction model:
        <i>Age, Total_Reels_Watched, Coffee_Consumed_Per_Day, Focus_Sessions_Count, Study_Hours, Is_Late_Night, Device_Type</i>.
        <br><br>
        <b>Strict Leakage Exclusion:</b> Target-derived columns such as <code>Brainrot_Exposure_Score</code>, 
        <code>Wellbeing_Score</code>, <code>Attention_Span_Level</code>, and <code>Aura_Color_Code</code> were strictly excluded 
        to guarantee true generalization and prevent artificial score inflation.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("4. Contact & Social Links")
    st.markdown(
        """
        <p style='color:#CBD5E1; line-height:1.9;'>
        Developed by <b>Mahmoud Islam</b> - Data Scientist & AI Engineer.
        </p>
        <div style='display:flex; gap:16px; margin-top:12px; flex-wrap:wrap;'>
            <a href='https://www.linkedin.com/in/mahmoud-islam-analytics/' target='_blank' style='background:rgba(124, 92, 252, 0.15); border:1px solid #7C5CFC; color:#E2E8F0; padding:10px 18px; border-radius:10px; text-decoration:none; font-weight:600;'>LinkedIn Profile</a>
            <a href='https://github.com/Mahmoud-islamcs' target='_blank' style='background:rgba(255, 255, 255, 0.05); border:1px solid rgba(255, 255, 255, 0.2); color:#E2E8F0; padding:10px 18px; border-radius:10px; text-decoration:none; font-weight:600;'>GitHub Repository</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
