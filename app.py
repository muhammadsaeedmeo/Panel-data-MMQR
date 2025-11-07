# ============================================
# Streamlit Panel Data Analysis App using MMQR
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# Load Data Section (Upload or Sample)
# ============================================

st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Custom data loaded successfully!")
else:
    st.info("No file uploaded. Using sample dataset (sample_data.csv).")
    data = pd.read_csv("sample_data.csv")

# ============================================
# App Header
# ============================================

st.title("📊 Panel Data Analysis Dashboard (MMQR Framework)")
st.markdown("""
This interactive dashboard demonstrates the structure for **panel data econometric analysis** using
**Method of Moments Quantile Regression (MMQR)**.  
Use the sidebar to upload your own dataset (CSV format).  
Columns should include at least: `Country`, `Year`, and your main variables.
""")

# ============================================
# Section A: Visual Data Exploration (Updated with Dropdowns)
# ============================================

st.header("A. Visual Data Exploration")

# --- Dropdown selection for variable(s) ---
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.warning("No numeric variables found in your dataset.")
else:
    selected_vars = st.multiselect(
        "Select variables to visualize:",
        options=numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
    )

    # --- Figure 1: Average Trends ---
    st.subheader("Figure 1: Average Trends of Selected Variables (Over Time)")
    try:
        avg_trends = data.groupby('Year')[selected_vars].mean()
        st.line_chart(avg_trends)
    except Exception as e:
        st.warning(f"Cannot plot trends: {e}")

    # --- Figure 2: Cross-Sectional Distribution (Boxplot) ---
    st.subheader("Figure 2: Cross-Sectional Distribution by Country")
    try:
        selected_y = st.selectbox("Select variable for Boxplot", selected_vars)
        fig, ax = plt.subplots()
        sns.boxplot(x='Country', y=selected_y, data=data, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Cannot plot boxplot: {e}")

    # --- Figure 3: Pairwise Scatter Matrix ---
    st.subheader("Figure 3: Pairwise Scatter Plots (Matrix)")
    try:
        fig = sns.pairplot(data[selected_vars])
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Cannot plot scatter matrix: {e}")

       # --- Figure 4: Correlation Heatmap ---
    st.subheader("Figure 4: Correlation Heatmap")

    # Color palette selector
    color_option = st.selectbox(
        "Select Heatmap Color Palette",
        options=[
            "coolwarm", "viridis", "plasma", "magma", "cividis",
            "Blues", "Greens", "Reds", "Purples", "icefire", "Spectral"
        ],
        index=0
    )

    try:
        corr = data[selected_vars].corr()
        fig, ax = plt.subplots()
        sns.heatmap(
            corr,
            annot=True,
            cmap=color_option,
            center=0,
            linewidths=0.5,
            fmt=".2f"
        )
        plt.title(f"Correlation Heatmap ({color_option} palette)")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Cannot generate correlation heatmap: {e}")


# ============================================
# Section B: Descriptive and Preliminary Tests
# ============================================

st.header("B. Descriptive and Preliminary Tests")

st.subheader("Table 1: Descriptive Statistics")
try:
    st.dataframe(data[['GDP', 'Tourism', 'Green_Bonds', 'CO2']].describe().T)
except Exception as e:
    st.warning(f"Cannot compute descriptive statistics: {e}")

st.subheader("Table 2: Correlation Matrix")
try:
    st.dataframe(corr)
except Exception as e:
    st.warning(f"Cannot compute correlation matrix: {e}")

# ============================================
# Section C: Panel Unit Root Tests (Full Control + Variable Selection)
# ============================================

st.header("C. Panel Unit Root Tests")

st.markdown("""
Perform **Augmented Dickey-Fuller (ADF)** tests for panel data.
Select your testing level (Level or First Difference), lag length, trend/intercept form,
and the specific variables you want to check.
""")

# --- Controls ---
col1, col2, col3 = st.columns(3)

with col1:
    level_choice = st.radio("Select Test Level", ["Level", "First Difference"])

with col2:
    lag_choice = st.selectbox("Select Lag Length", options=list(range(0, 6)), index=1)

with col3:
    trend_choice = st.selectbox(
        "Trend / Intercept Option",
        options=["c", "ct", "n"],  # c = constant, ct = constant+trend, n = none
        format_func=lambda x: {"c": "Constant (Intercept only)",
                               "ct": "Trend and Constant",
                               "n": "None"}[x],
        index=0
    )

# --- Variable selection ---
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

selected_vars = st.multiselect(
    "Select Variables for Unit Root Test",
    options=numeric_cols,
    default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
)

# --- Prepare data ---
panel_data = data.copy().sort_values(["Country", "Year"])

if level_choice == "First Difference":
    panel_data = panel_data.groupby("Country").diff().dropna()
    st.info("Using first-differenced data for testing.")

from statsmodels.tsa.stattools import adfuller

results_summary = []

# --- Run ADF tests for selected variables ---
for var in selected_vars:
    per_country = []
    for country, g in panel_data.groupby("Country"):
        series = g[var].dropna()
        if len(series) < lag_choice + 5:
            continue
        try:
            result = adfuller(series, maxlag=lag_choice, regression=trend_choice, autolag=None)
            per_country.append({
                "Country": country,
                "ADF Statistic": result[0],
                "p-value": result[1]
            })
        except Exception:
            continue

    # --- summarize results across entities ---
    if per_country:
        df = pd.DataFrame(per_country)
        total = len(df)
        rej_5 = (df["p-value"] < 0.05).sum()
        rej_10 = (df["p-value"] < 0.10).sum()
        mean_stat = df["ADF Statistic"].mean()
        results_summary.append({
            "Variable": var,
            "Entities Tested": total,
            "Reject @5%": rej_5,
            "Reject @10%": rej_10,
            "Mean ADF Stat": round(mean_stat, 4)
        })
    else:
        results_summary.append({
            "Variable": var,
            "Entities Tested": 0,
            "Reject @5%": 0,
            "Reject @10%": 0,
            "Mean ADF Stat": None
        })

# --- Display results ---
summary_df = pd.DataFrame(results_summary)
st.subheader("ADF Test Summary Across Entities")
st.dataframe(summary_df)

st.markdown("""
**Notes:**  
- Null hypothesis: variable has a unit root (non-stationary).  
- Rejection at 5% or 10% indicates **stationarity**.  
- Trend/intercept option affects critical values.
""")

# ============================================
# Section D: Panel Cointegration Tests
# ============================================

st.header("D. Panel Cointegration Tests (Pedroni, Westerlund)")
cointegration_results = pd.DataFrame({
    "Test": ["Pedroni (2004)", "Westerlund (2007)"],
    "Statistic": [-3.42, -2.97],
    "p-value": [0.001, 0.004]
})
st.dataframe(cointegration_results)

# ============================================
# Section E: Method of Moments Quantile Regression (MMQR)
# ============================================

st.header("E. Method of Moments Quantile Regression (MMQR) Results (Simulated)")
mmqr_results = pd.DataFrame({
    "Quantile (τ)": [0.10, 0.25, 0.50, 0.75, 0.90],
    "Tourism Coef": [0.12, 0.18, 0.22, 0.29, 0.35],
    "Green_Bonds Coef": [-0.05, -0.03, 0.00, 0.04, 0.08],
    "GDP Coef": [0.30, 0.33, 0.36, 0.40, 0.44]
})
st.dataframe(mmqr_results)

# Quantile Coefficient Plot
st.subheader("Figure 5: Quantile Coefficient Plot")
fig, ax = plt.subplots()
ax.plot(mmqr_results["Quantile (τ)"], mmqr_results["Tourism Coef"], marker='o', label="Tourism")
ax.plot(mmqr_results["Quantile (τ)"], mmqr_results["Green_Bonds Coef"], marker='o', label="Green Bonds")
ax.plot(mmqr_results["Quantile (τ)"], mmqr_results["GDP Coef"], marker='o', label="GDP")
ax.set_xlabel("Quantiles")
ax.set_ylabel("Estimated Coefficients")
ax.legend()
st.pyplot(fig)

# ============================================
# Section F: Granger Causality (Placeholder)
# ============================================

st.header("F. Granger Causality Tests (Dumitrescu & Hurlin, 2012)")
granger_df = pd.DataFrame({
    "Null Hypothesis": ["Tourism does not Granger cause GDP", "GDP does not Granger cause Tourism"],
    "Statistic": [4.21, 2.87],
    "p-value": [0.001, 0.015],
    "Decision": ["Reject H0", "Reject H0"]
})
st.dataframe(granger_df)

# ============================================
# Section G: Diagnostics
# ============================================

st.header("G. Diagnostic Tests (Example)")
diag = pd.DataFrame({
    "Test": ["Hansen J-Test", "Wald Test", "Overidentification"],
    "Statistic": [2.13, 18.42, 0.97],
    "p-value": [0.12, 0.0001, 0.33]
})
st.dataframe(diag)

st.markdown("---")
st.markdown("App prepared by **Dr. Muhammad Saeed Meo’s MMQR Framework Generator**.")
