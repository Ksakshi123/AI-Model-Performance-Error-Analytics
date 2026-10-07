import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_csv, generate_sample_data
from analytics import (
    calculate_metrics,
    calculate_errors,
    find_error_drivers
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Error Analytics",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
}

.dashboard-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    color: #64748b;
    font-size: 16px;
    margin-bottom: 25px;
}

.metric-card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e2e8f0;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="dashboard-title">AI Model Performance & Error Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Analyze model predictions, identify errors, and discover potential error patterns.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

st.sidebar.markdown("---")

use_sample = st.sidebar.button(
    "Load Sample Dataset",
    use_container_width=True
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if uploaded_file is not None:

    df = load_csv(uploaded_file)

elif use_sample:

    df = generate_sample_data()

else:

    df = generate_sample_data()


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📁 Dataset Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Rows",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Total Columns",
        len(df.columns)
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Numeric Columns",
        len(df.select_dtypes(include=np.number).columns)
    )

st.dataframe(
    df.head(100),
    use_container_width=True,
    height=300
)


# --------------------------------------------------
# NUMERIC COLUMNS
# --------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


if len(numeric_columns) < 2:

    st.warning(
        "Your dataset needs at least two numeric columns "
        "for prediction error analysis."
    )

    st.stop()


# --------------------------------------------------
# COLUMN SELECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🎯 Prediction Configuration</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    actual_column = st.selectbox(
        "Actual / Target Column",
        numeric_columns
    )

with col2:

    prediction_options = [
        col for col in numeric_columns
        if col != actual_column
    ]

    predicted_column = st.selectbox(
        "Predicted Column",
        prediction_options
    )


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

metrics = calculate_metrics(
    df,
    actual_column,
    predicted_column
)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "MAE",
        f"{metrics['MAE']:.4f}"
    )

with col2:
    st.metric(
        "MSE",
        f"{metrics['MSE']:.4f}"
    )

with col3:
    st.metric(
        "RMSE",
        f"{metrics['RMSE']:.4f}"
    )

with col4:
    st.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )


# --------------------------------------------------
# ERROR CALCULATION
# --------------------------------------------------

error_df = calculate_errors(
    df,
    actual_column,
    predicted_column
)


# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Error Analysis</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# Actual vs Predicted

with col1:

    st.subheader("Actual vs Predicted")

    chart_data = error_df[
        [actual_column, predicted_column]
    ].copy()

    chart_data.columns = [
        "Actual",
        "Predicted"
    ]

    st.scatter_chart(
        chart_data,
        x="Actual",
        y="Predicted"
    )


# Error distribution

with col2:

    st.subheader("Error Distribution")

    error_chart = error_df[
        ["Absolute Error"]
    ]

    st.bar_chart(
        error_chart.head(50)
    )


# --------------------------------------------------
# ERROR STATISTICS
# --------------------------------------------------

st.subheader("Error Statistics")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Average Error",
        f"{error_df['Error'].mean():.4f}"
    )

with col2:

    st.metric(
        "Maximum Error",
        f"{error_df['Absolute Error'].max():.4f}"
    )

with col3:

    high_errors = (
        error_df["Absolute Error"]
        > error_df["Absolute Error"].quantile(0.90)
    ).sum()

    st.metric(
        "High Error Cases",
        int(high_errors)
    )


# --------------------------------------------------
# WORST PREDICTIONS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🚨 Largest Prediction Errors</div>',
    unsafe_allow_html=True
)

worst_errors = error_df.sort_values(
    "Absolute Error",
    ascending=False
).head(20)

st.dataframe(
    worst_errors,
    use_container_width=True
)


# --------------------------------------------------
# ERROR DRIVERS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🔍 Potential Error Drivers</div>',
    unsafe_allow_html=True
)

drivers = find_error_drivers(
    df,
    error_df,
    actual_column,
    predicted_column
)

if len(drivers) > 0:

    st.dataframe(
        drivers,
        use_container_width=True
    )

else:

    st.info(
        "No additional numeric feature columns were found "
        "for error-driver analysis."
    )


# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📥 Export</div>',
    unsafe_allow_html=True
)

csv_data = error_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Error Report",
    data=csv_data,
    file_name="error_analysis_report.csv",
    mime="text/csv"
)