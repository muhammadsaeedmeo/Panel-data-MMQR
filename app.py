# ============================================
# Streamlit Panel Data Analysis App using MMQR but no p
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.formula.api import quantreg
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy import stats
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ============================================
# App Configuration
# ============================================

st.set_page_config(page_title="Panel Data Analysis Dashboard", layout="wide")

# ============================================
# Section A: Data Upload
# ============================================

st.title("📊 Panel Data Analysis Dashboard (MMQR Framework)")

st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom data loaded successfully!")
else:
    st.info("No file uploaded. Using sample dataset.")
    np.random.seed(42)
    countries = [f"Country_{i}" for i in range(1, 11)]
    years = list(range(2000, 2020))
    sample = []
    for c in countries:
        for y in years:
            gdp = np.random.normal(100, 20)
            tourism = gdp * 0.3 + np.random.normal(0, 5)
            sample.append({
                "Country": c,
                "Year": y,
                "GDP": gdp,
                "Tourism": tourism,
                "Investment": np.random.normal(50, 10),
                "Trade": np.random.normal(60, 15)
            })
    df = pd.DataFrame(sample)

st.header("A. Data Overview")
st.dataframe(df.head())
st.write(f"Dataset shape: {df.shape}")

if "Country" not in df.columns or "Year" not in df.columns:
    st.error("❌ Required columns 'Country' and/or 'Year' missing.")
    st.stop()

# ======================================================================
# 📊 SECTION: DESCRIPTIVE STATISTICS AND DISTRIBUTION ANALYSIS (Enhanced)
# ======================================================================

st.subheader("Descriptive Statistics and Distribution Analysis")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_col = st.selectbox(
    "Select a variable (or choose 'All Variables - Combined Summary Plot')",
    options=["All Variables - Combined Summary Plot"] + numeric_cols
)

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(col):
    data = df[col].dropna()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    
    sns.histplot(data, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Histogram + KDE")
    
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot")
    
    sns.boxplot(y=data, ax=axes[2], color="mediumseagreen")
    axes[2].set_title("Box Plot")
    
    sns.violinplot(y=data, ax=axes[3], color="salmon")
    axes[3].set_title("Violin Plot")
    
    plt.tight_layout()
    st.pyplot(fig)

    if len(data) > 3:
        stat, p = stats.shapiro(data)
        if p > 0.05:
            st.info(f"**{col}** appears normally distributed (p = {p:.3f}).")
        else:
            st.warning(f"**{col}** deviates from normality (p = {p:.3f}).")
    else:
        st.write("Sample too small for normality test.")
    st.markdown("---")


# ---- Combined Plot for All Variables ----
def combined_distribution_plot(df, numeric_cols):
    n = len(numeric_cols)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()
        sns.kdeplot(data, fill=True, ax=axes[i], color=sns.color_palette("husl", n)[i])
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Density")
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Combined Distribution of All Variables", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    st.pyplot(fig)
    st.markdown("**Note:** The density plots show each variable’s overall distribution pattern for quick comparison.")


# ---- Logic ----
if selected_col == "All Variables - Combined Summary Plot":
    combined_distribution_plot(df, numeric_cols)
else:
    st.subheader(f"Descriptive Analysis for {selected_col}")
    plot_distribution(selected_col)

# ============================================
# Section C: Correlation Analysis
# ============================================

st.header("C. Correlation Analysis")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    col1, col2 = st.columns(2)
    with col1:
        dep_var = st.selectbox("Select Dependent Variable", options=numeric_cols)
    with col2:
        indep_vars = st.multiselect(
            "Select Independent Variable(s)",
            options=[c for c in numeric_cols if c != dep_var],
            default=[c for c in numeric_cols if c != dep_var][:3]
        )

    color_option = st.selectbox(
        "Heatmap Color Palette",
        options=["coolwarm","viridis","plasma","magma","cividis","Blues","Greens","Reds"],
        index=0
    )

    if indep_vars:
        selected_vars = [dep_var] + indep_vars
        corr = df[selected_vars].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap=color_option, center=0, linewidths=0.5, fmt=".2f", ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)

        def interpret_corr(v):
            v = abs(v)
            if v < 0.20: return "very weak"
            if v < 0.40: return "weak"
            if v < 0.60: return "moderate"
            if v < 0.80: return "strong"
            return "very strong"

        st.subheader("Correlation Interpretation")
        for var in indep_vars:
            val = corr.loc[dep_var, var]
            st.write(f"- {dep_var} and {var}: {val:.2f} ({interpret_corr(val)} {'positive' if val>0 else 'negative'})")
else:
    st.warning("No numeric variables for correlation.")

# ============================================
# Section 4: Method of Moments Quantile Regression (MMQR)
# ============================================

st.header("Section 4: Method of Moments Quantile Regression (MMQR) with Location–Scale Decomposition")

# --- Step 1: Normalize variables ---
numeric_cols = [dep_var] + indep_vars
df_norm = df.copy()
for col in numeric_cols:
    df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()

st.markdown("All variables have been standardized (mean = 0, SD = 1).")

# --- Step 2: Location parameters (τ = 0.5) ---
ref_q = 0.5
formula = f"{dep_var} ~ {' + '.join(indep_vars)}"
location_model = quantreg(formula, df_norm).fit(q=ref_q)
location_params = location_model.params
location_se = location_model.bse
location_p = location_model.pvalues

# --- Step 3: Scale parameters (τ = 0.25–0.75) ---
q_low, q_high = 0.25, 0.75
model_low = quantreg(formula, df_norm).fit(q=q_low)
model_high = quantreg(formula, df_norm).fit(q=q_high)

scale_params = (model_high.params - model_low.params) / (q_high - q_low)
scale_se = np.sqrt((model_high.bse**2 + model_low.bse**2) / ((q_high - q_low) ** 2))
t_vals = scale_params / scale_se
df_resid = len(df_norm) - len(indep_vars) - 1
scale_p = 2 * (1 - stats.t.cdf(abs(t_vals), df=df_resid))

# --- Step 4: Display Location and Scale Tables ---
st.subheader(f"Location Parameters (τ = {ref_q})")
loc_table = pd.DataFrame({
    "Coefficient": location_params.round(3),
    "Std. Error": location_se.round(3),
    "P-Value": location_p.round(3)
})
st.dataframe(loc_table)

st.subheader(f"Scale Parameters (τ = {q_low}–{q_high})")
scale_table = pd.DataFrame({
    "Coefficient": scale_params.round(3),
    "Std. Error": scale_se.round(3),
    "P-Value": scale_p.round(3)
})
st.dataframe(scale_table)

# --- Step 5: MMQR Coefficients, SEs, and P-values Across Quantiles ---
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
mmqr_coef, mmqr_se, mmqr_p = {}, {}, {}

for q in quantiles:
    coeff_q = location_params + scale_params * q
    mmqr_coef[q] = coeff_q

    se_q = np.sqrt(location_se**2 + (q**2) * scale_se**2)
    mmqr_se[q] = se_q

    t_vals = coeff_q / se_q
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=df_resid))
    mmqr_p[q] = p_vals

# --- Step 6: Combine into one wide table (Coef–Std–P order per quantile) ---
combined_cols = {}
for q in quantiles:
    combined_cols[(f"τ={q}", "Coefficient")] = mmqr_coef[q].round(3)
    combined_cols[(f"τ={q}", "Std. Error")] = mmqr_se[q].round(3)
    combined_cols[(f"τ={q}", "P-Value")] = mmqr_p[q].round(3)

mmqr_df = pd.concat(combined_cols, axis=1)

# --- Proper multiindex formatting ---
mmqr_df.columns = pd.MultiIndex.from_tuples(mmqr_df.columns)
st.subheader("MMQR Results: Coefficient – Std. Error – P-Value by Quantile")
st.dataframe(mmqr_df)

# --- Step 7: Plot Coefficient Dynamics ---
st.subheader("Coefficient Dynamics by Quantile")
fig, ax = plt.subplots(figsize=(8, 5))
for var in indep_vars:
    ax.plot(quantiles, [mmqr_coef[q][var] for q in quantiles],
            marker="o", label=var)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Quantile (τ)")
ax.set_ylabel("Coefficient (standardized)")
ax.set_title("MMQR Coefficient Dynamics Across Quantiles")
ax.legend()
st.pyplot(fig)

# --- Step 8: Download Results ---
out_text = []
out_text.append("=== LOCATION PARAMETERS ===\n")
out_text.append(loc_table.to_string())
out_text.append("\n\n=== SCALE PARAMETERS ===\n")
out_text.append(scale_table.to_string())
out_text.append("\n\n=== MMQR RESULTS (Coef, Std.Err, P-Value by Quantile) ===\n")
out_text.append(mmqr_df.to_string())

st.download_button(
    "📥 Download MMQR Full Table (RTF)",
    data="\n".join(out_text),
    file_name="MMQR_Results_ByQuantile.rtf",
    mime="application/rtf"
)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
