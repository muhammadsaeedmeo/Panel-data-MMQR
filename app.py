# ============================================
# Streamlit Panel Data Analysis App using MMQR
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

# ===================================================
# 🟩 SECTION 4: MMQR estimation and results
# ===================================================

# ... your code for estimating quantiles, storing results in mmqr_results ...
# mmqr_results[q] = {"model": model, "mmqr_coefficients": ..., "pvalues": ...}

# ⬇️ Place the following block immediately AFTER all MMQR quantile regressions
# ===================================================
# LOCATION & SCALE RESULTS SECTION
# ===================================================

# [Paste the big location–scale block I gave you here]
# ========================
# Location & Scale results (display + CSV)
# ========================
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from tqdm import trange

# --- Parameters for scale calculation ---
tau_low = 0.25
tau_high = 0.75
tau_diff = tau_high - tau_low
ref_q = reference_quantile if 'reference_quantile' in globals() else 0.5
bootstrap_for_scale = bootstrap_ci if 'bootstrap_ci' in globals() else False
n_boot = int(n_bootstrap) if 'n_bootstrap' in globals() else 200

# --- Ensure mmqr_results exist (rebuild if not) ---
if 'mmqr_results' not in globals() or not isinstance(mmqr_results, dict):
    mmqr_results = {}
    for q in quantiles:
        try:
            model = smf.quantreg(f"{dep_var} ~ {' + '.join(indep_vars)}", data=df).fit(q=q)
            mmqr_results[q] = {
                "model": model,
                "mmqr_coefficients": model.params,
                "pvalues": model.pvalues,
                "coefficients": model.params
            }
        except Exception as e:
            st.warning(f"Failed to run quantile {q}: {e}")

# --- 1) Location parameters (reference quantile) ---
st.subheader(f"Table: Location parameters (reference quantile τ = {ref_q})")
if ref_q not in mmqr_results:
    try:
        model_ref = smf.quantreg(f"{dep_var} ~ {' + '.join(indep_vars)}", data=df).fit(q=ref_q)
        mmqr_results[ref_q] = {"model": model_ref, "mmqr_coefficients": model_ref.params,
                               "pvalues": model_ref.pvalues, "coefficients": model_ref.params}
    except Exception as e:
        st.error(f"Cannot estimate reference quantile {ref_q}: {e}")
        model_ref = None
else:
    model_ref = mmqr_results[ref_q]["model"]

location_rows = []
if model_ref is not None:
    for var in model_ref.params.index:
        coef = float(model_ref.params[var])
        try:
            se = float(model_ref.bse[var])
        except Exception:
            se = np.nan
        try:
            pval = float(model_ref.pvalues[var])
        except Exception:
            pval = np.nan
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        location_rows.append({
            "Variable": "Intercept" if var.lower() in ["_cons", "intercept"] else var,
            "Coefficient": round(coef, 3),
            "Std. Error": round(se, 3) if np.isfinite(se) else "NA",
            "P-Value": round(pval, 3) if np.isfinite(pval) else "NA",
            "Signif": stars
        })
location_df = pd.DataFrame(location_rows)
st.dataframe(location_df, use_container_width=True)

# --- 2) Scale parameters: (Q_high - Q_low) / (tau_high - tau_low) ---
st.subheader(f"Table: Scale parameters (based on τ={tau_low} and τ={tau_high})")

# Ensure models at tau_low and tau_high exist
for q in (tau_low, tau_high):
    if q not in mmqr_results:
        try:
            m = smf.quantreg(f"{dep_var} ~ {' + '.join(indep_vars)}", data=df).fit(q=q)
            mmqr_results[q] = {"model": m, "mmqr_coefficients": m.params,
                               "pvalues": m.pvalues, "coefficients": m.params}
        except Exception as e:
            st.error(f"Cannot estimate quantile {q}: {e}")

# Extract params
model_low = mmqr_results.get(tau_low, {}).get("model", None)
model_high = mmqr_results.get(tau_high, {}).get("model", None)

scale_rows = []
if model_low is None or model_high is None:
    st.error("Missing low/high quantile models; scale parameters cannot be computed.")
else:
    params_low = model_low.params
    params_high = model_high.params

    # Delta-method conservative variance: Var(scale) ≈ (Var(high) + Var(low)) / tau_diff^2
    var_high = (model_high.bse ** 2).to_dict()
    var_low = (model_low.bse ** 2).to_dict()

    # Optional bootstrap to estimate SEs and p-values for scale
    boot_scale = None
    if bootstrap_for_scale and n_boot > 0:
        st.info(f"Bootstrapping scale SEs ({n_boot} draws)...")
        boot_scale = {var: [] for var in params_high.index}
        for i in range(n_boot):
            boot_samp = df.sample(n=len(df), replace=True)
            try:
                bh = smf.quantreg(f"{dep_var} ~ {' + '.join(indep_vars)}", data=boot_samp).fit(q=tau_high)
                bl = smf.quantreg(f"{dep_var} ~ {' + '.join(indep_vars)}", data=boot_samp).fit(q=tau_low)
                bscale = (bh.params - bl.params) / tau_diff
                for var in params_high.index:
                    boot_scale[var].append(bscale.get(var, np.nan))
            except Exception:
                continue

    for var in params_high.index:
        scale_val = float((params_high[var] - params_low[var]) / tau_diff)
        # compute se:
        se_scale = np.nan
        pval_scale = np.nan

        # prefer bootstrap SE/pval if available
        if boot_scale is not None and len(boot_scale.get(var, [])) > 0:
            arr = np.array(boot_scale[var])
            # remove nans
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                se_scale = float(np.std(arr, ddof=1))
                # two-sided p-value from bootstrap distribution
                # p = proportion of bootstrap estimates more extreme than observed (two-sided)
                pval_scale = 2 * min(np.mean(arr >= scale_val), np.mean(arr <= scale_val))
        else:
            # delta-method approximate (conservative; assumes cov=0)
            vh = var_high.get(var, np.nan)
            vl = var_low.get(var, np.nan)
            if np.isfinite(vh) and np.isfinite(vl):
                var_scale = (vh + vl) / (tau_diff ** 2)
                if var_scale >= 0:
                    se_scale = float(np.sqrt(var_scale))
                    # Student-t p-value with df = n - k - 1
                    dfree = max(len(df) - len(indep_vars) - 1, 1)
                    tstat = scale_val / se_scale if se_scale > 0 else 0.0
                    from scipy import stats
                    pval_scale = float(2 * (1 - stats.t.cdf(abs(tstat), df=dfree)))
        stars = ''
        if np.isfinite(pval_scale):
            if pval_scale < 0.01:
                stars = '***'
            elif pval_scale < 0.05:
                stars = '**'
            elif pval_scale < 0.1:
                stars = '*'

        scale_rows.append({
            "Variable": "Intercept" if var.lower() in ["_cons", "intercept"] else var,
            "Scale Coef": round(scale_val, 3),
            "Std. Error": round(se_scale, 3) if np.isfinite(se_scale) else "NA",
            "P-Value": round(pval_scale, 3) if np.isfinite(pval_scale) else "NA",
            "Signif": stars
        })

scale_df = pd.DataFrame(scale_rows)
st.dataframe(scale_df, use_container_width=True)

# === Add location & scale to download bundle ===
# Prepare combined download table (location + scale + mmqr)
download_rows = []

# location
for r in location_rows:
    download_rows.append({
        "Variable": r["Variable"],
        "Type": "Location",
        "Coefficient": r["Coefficient"],
        "StdError": r["Std. Error"],
        "PValue": r["P-Value"],
        "Significance": r["Signif"],
        "Quantile": ref_q
    })

# scale
for r in scale_rows:
    download_rows.append({
        "Variable": r["Variable"],
        "Type": "Scale",
        "Coefficient": r["Scale Coef"],
        "StdError": r["Std. Error"],
        "PValue": r["P-Value"],
        "Significance": r["Signif"],
        "Quantile": "NA"
    })

# mmqr coefficients (existing structure)
first_q = next((q for q in quantiles if q in mmqr_results), None)
if first_q is not None:
    coef_names = mmqr_results[first_q]["coefficients"].index.tolist()
    for var in coef_names:
        for q in quantiles:
            if q in mmqr_results:
                coef = mmqr_results[q]["mmqr_coefficients"].get(var, np.nan)
                try:
                    se = mmqr_results[q]["model"].bse.get(var, np.nan)
                except Exception:
                    se = np.nan
                try:
                    pval = mmqr_results[q]["pvalues"].get(var, np.nan)
                except Exception:
                    pval = np.nan
                download_rows.append({
                    "Variable": "Intercept" if var.lower() in ["_cons", "intercept"] else var,
                    "Type": f"MMQR_τ={q}",
                    "Coefficient": round(float(coef), 3) if np.isfinite(coef) else "NA",
                    "StdError": round(float(se), 3) if np.isfinite(se) else "NA",
                    "PValue": round(float(pval), 3) if np.isfinite(pval) else "NA",
                    "Significance": '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else '',
                    "Quantile": q
                })

download_df = pd.DataFrame(download_rows)
csv_out = "MMQR_Combined_Location_Scale_MMQR.csv"
download_df.to_csv(csv_out, index=False)
st.success(f"Combined results saved to {csv_out}")
with open(csv_out, "rb") as f:
    st.download_button("⬇️ Download Combined Location/Scale/MMQR (CSV)", data=f, file_name=csv_out, mime="text/csv")


# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
