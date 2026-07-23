import streamlit as st


def render_about_page(best_model_name: str, total_records: int):
    st.title("About Brain Rot Analytics")
    st.markdown(
        "Comprehensive architectural overview, dataset specifications, methodology, MLOps governance, "
        "and contact information for the Brain Rot Behavioral Intelligence project."
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("1. Model Architecture Overview")
    st.markdown(
        f"""
        <p style='color:#CBD5E1; line-height:1.9;'>
        <b>"Brain Rot Analytics"</b> is an enterprise-grade AI behavioral intelligence platform. 
        It studies the relationship between daily digital habits—specifically short-form video consumption—and 
        student cognitive wellbeing.
        <br><br>
        <b>Champion Algorithm:</b> <code>{best_model_name}</code><br>
        <b>Evaluation Benchmark:</b> Benchmarked across Logistic Regression, Random Forest, and Gradient Boosting / XGBoost using 
        <code>RandomizedSearchCV</code> hyperparameter optimization.<br>
        <b>Primary Metric:</b> <b>Macro F1-Score of 0.903</b> (Overall Accuracy of <b>94.5%</b>). Macro F1-score 
        was chosen to guarantee balanced classification across all minority distraction classes.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("2. Explainable AI (XAI) & MLOps Governance")
    st.markdown(
        """
        <p style='color:#CBD5E1; line-height:1.9;'>
        <b>SHAP Local Explainability:</b> Incorporates interactive SHAP (SHapley Additive exPlanations) waterfall attributions 
        to break down single-prediction decisions into exact positive and negative feature risk contributions.
        <br><br>
        <b>Data Drift Analytics (PSI):</b> Tracks Population Stability Index (PSI) and distribution shifts between 
        baseline training data and production inference streams to maintain long-term model governance.
        <br>
        <b>User Feedback Loop:</b> Logged user ratings and ground truth validations stored in structured JSON format for automated retraining pipelines.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("3. Dataset Distribution & Imbalance Strategy")
    st.markdown(
        f"""
        <p style='color:#CBD5E1; line-height:1.9;'>
        The underlying dataset comprises <b>{total_records:,} student records</b> with 30 attributes collected across Egyptian student surveys.
        <br><br>
        <b>Missing Value Imputation:</b> Imputed using median and mode strategies to preserve full sample size without dropping rows.
        <br>
        <b>Class Imbalance Handling:</b> Enforced strict class weighting (<code>class_weight='balanced'</code>) and stratified train/test split.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("4. Methodology & Data Leakage Prevention")
    st.markdown(
        """
        <p style='color:#CBD5E1; line-height:1.9;'>
        <b>Feature Engineering & Selection:</b> Only <b>7 core behavioral features</b> were passed to the prediction model:
        <i>Age, Total_Reels_Watched, Coffee_Consumed_Per_Day, Focus_Sessions_Count, Study_Hours, Is_Late_Night, Device_Type</i>.
        <br><br>
        <b>Strict Leakage Exclusion:</b> Target-derived columns such as <code>Brainrot_Exposure_Score</code>, 
        <code>Wellbeing_Score</code>, <code>Attention_Span_Level</code>, and <code>Aura_Color_Code</code> were strictly excluded.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("5. Contact & Social Links")
    st.markdown(
        """
        <p style='color:#CBD5E1; line-height:1.9;'>
        Developed by <b>Mahmoud Islam</b> - Data Scientist & AI Engineer.
        </p>
        <div style='display:flex; gap:16px; margin-top:12px; flex-wrap:wrap;'>
            <a href='https://www.linkedin.com/in/mahmoud-islam-analytics/' target='_blank' style='background:rgba(124, 92, 252, 0.15); border:1px solid #7C5CFC; color:#E2E8F0; padding:10px 18px; border-radius:10px; text-decoration:none; font-weight:600;'>LinkedIn Profile</a>
            <a href='https://github.com/Mahmoud-islamcs/Neuro-Digital-Analytics' target='_blank' style='background:rgba(255, 255, 255, 0.05); border:1px solid rgba(255, 255, 255, 0.2); color:#E2E8F0; padding:10px 18px; border-radius:10px; text-decoration:none; font-weight:600;'>GitHub Repository</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
