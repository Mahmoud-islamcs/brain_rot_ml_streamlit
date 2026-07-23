# BrainRot Analytics - AI-Powered Behavioral Intelligence Platform

## Executive Overview

BrainRot Analytics is an enterprise-grade AI behavioral intelligence platform designed to study, quantify, and predict digital distraction stages among students. Leveraging machine learning models trained on survey data from 5,000 Egyptian students, the system evaluates the complex interactions between daily short-form video consumption (Reels/TikTok/Shorts), study hours, caffeine intake, deep-focus sessions, midnight phone usage, age, and primary device types.

The core objective is to identify individuals experiencing mild, advanced, or critical digital distraction early, providing explainable model outputs and targeted interventions to foster healthy digital habits and preserve student wellbeing.

---

## Live Application

Access the live production deployment hosted on Streamlit Community Cloud:

**Live Demo URL:** [https://brainrot-analytics-egy.streamlit.app/](https://brainrot-analytics-egy.streamlit.app/)

---

## Key Features

1. **Real-Time Prediction Engine**
   - Interactive behavioral input interface with age sliders, habit toggles, and device selectors.
   - Instant classification into four distraction stages: Healthy, Casual, Advanced, and Critical.
   - Automated model confidence score calculation and downloadable assessment reports (HTML format).

2. **Explainable AI (XAI) & Interpretability**
   - Local feature attribution scores (SHAP style) breaking down positive and negative risk contributors for every individual prediction.
   - Global feature importance rankings highlighting primary behavioral drivers across the student population.

3. **Geospatial Analytics & Interactive Risk Mapping**
   - Dynamic map component built with Plotly Express and dark-themed basemap tiles (`carto-darkmatter`).
   - Regional data aggregation across Egyptian governorates (Cairo, Alexandria, Giza, Port Said, Suez, Mansoura, Tanta, Minya, Asyut, Luxor, Aswan).
   - Interactive map layer toggles (Aggregated Proportional Markers, Density Heatmap View, and Categorical Stage Markers).
   - Exportable regional summary statistics in CSV format.

4. **Batch Prediction Engine**
   - Bulk CSV dataset processing capability for institutional analysis.
   - Automated missing value imputation, column alignment, class probability distribution, and downloadable batch results.

5. **Dynamic Data Explorer & Statistical Dashboard**
   - Multi-dimensional distribution charts, correlation heatmaps, and habit interaction scatters.
   - Interactive filtering by Governorate, Age Group, Primary Device, and Distraction Stage.

6. **MLOps Governance & Drift Analytics**
   - Population Stability Index (PSI) calculation to track data drift between baseline training distributions and production inference data.
   - Benchmark model comparator comparing Random Forest, Gradient Boosting, and Logistic Regression algorithms.

---

## Technology Stack

- **Core Programming Language:** Python 3.10+
- **Web Application Framework:** Streamlit (Custom Glassmorphism Dark Theme)
- **Machine Learning & Modeling:** Scikit-Learn, Joblib, NumPy, Pandas
- **Data Visualization & GIS:** Plotly Express, Plotly Graph Objects
- **Model Explainability (XAI):** SHAP (SHapley Additive exPlanations)
- **Deployment Platform:** Streamlit Community Cloud

---

## Installation & Setup Guide

Follow these steps to set up and run the BrainRot Analytics platform on your local environment.

### Prerequisites

Ensure you have Git and Python 3.10 or later installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mahmoud-islamcs/brain_rot_ml_streamlit.git
cd brain_rot_ml_streamlit
```

### Step 2: Create and Activate a Virtual Environment

On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

On macOS / Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## Project Directory Structure

```text
brain_rot_ml_streamlit/
│
├── data/
│   └── BrainRot_Final_Dataset.csv     # Raw student survey dataset (5,000 records)
│
├── model/
│   ├── brainrot_model.pkl             # Trained Random Forest Champion Model
│   ├── scaler.pkl                     # StandardScaler artifact
│   ├── metadata.pkl                   # Training metadata & feature names
│   ├── feature_importance.csv         # Gini feature importances
│   ├── confusion_matrix.png           # Model evaluation confusion matrix
│   └── classification_report.txt     # Precision, recall, and F1-score breakdown
│
├── views/
│   ├── predict.py                     # Single-student prediction view & XAI report
│   ├── explorer.py                    # Dataset distribution & statistical analytics
│   ├── insights.py                    # Feature importance & MLOps governance
│   ├── geospatial.py                  # Regional map analytics & governorate insights
│   ├── batch.py                       # Bulk CSV upload & batch prediction view
│   ├── about.py                       # Architecture & methodology documentation
│   └── sidebar.py                     # Executive sidebar navigation renderer
│
├── requirements.txt                   # Production Python dependencies
├── utils.py                           # Helper utilities, clean data, & calculations
└── streamlit_app.py                   # Main application entry point & router
```

---

## Navigation & Usage Guide

1. **Predict:** Input individual student daily habits to compute immediate digital distraction stage predictions, model confidence, and SHAP feature attributions. Download formal HTML reports.
2. **Dataset Explorer:** Filter and analyze distribution patterns across 5,000 student records using multi-select filter controls.
3. **Insights:** Inspect champion model feature importance, confusion matrix tradeoffs, MLOps data drift alerts (PSI), and candidate model benchmarks.
4. **Geospatial Analysis:** Explore geographic distraction hotspots across Egyptian governorates using dark-themed interactive maps, layer toggles, and regional summary exports.
5. **Batch Prediction:** Upload multi-student CSV datasets for bulk prediction generation and download detailed output spreadsheets.
6. **About:** Review project methodology, feature engineering specifications, and author contact links.

---

## License & Contact

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Author & Developer Information
- **Author:** Mahmoud Islam
- **LinkedIn Profile:** [https://www.linkedin.com/in/mahmoud-islam-analytics/](https://www.linkedin.com/in/mahmoud-islam-analytics/)
- **GitHub Repository:** [https://github.com/Mahmoud-islamcs/Neuro-Digital-Analytics](https://github.com/Mahmoud-islamcs/Neuro-Digital-Analytics)
