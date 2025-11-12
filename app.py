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

# ============================================
# Corrected MMQR Implementation - Machado & Santos Silva (2019)
# With Proper Standard Errors and P-values
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist
from statsmodels.regression.quantile_regression import QuantReg as quantreg

st.header("Method of Moments Quantile Regression (MMQR) - Machado & Santos Silva (2019)")

if 'dep_var' not in locals() or 'indep_vars' not in locals() or not indep_vars:
    st.warning("Please complete the correlation analysis first to select variables.")
else:
    st.subheader("MMQR Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        quantiles = st.text_input("Quantiles (comma-separated)", "0.05,0.25,0.50,0.75,0.95")
        quantiles = [float(q.strip()) for q in quantiles.split(",")]
    
    with col2:
        bootstrap_inference = st.checkbox("Bootstrap Inference (Recommended)", True)
        n_bootstrap = st.slider("Bootstrap Samples", 100, 1000, 500) if bootstrap_inference else 100
        reference_quantile = st.selectbox("Reference Quantile for Location", [0.25, 0.50, 0.75], index=1)

    def mmqr_estimation(data, y_var, x_vars, quantiles, reference_quantile=0.5, bootstrap=True, n_boot=500):
        """
        Correct MMQR implementation following Machado & Santos Silva (2019)
        
        Key corrections:
        1. Uses Φ^(-1)(τ) as the transformation function h(τ)
        2. Analytical derivation of MMQR coefficients (no re-estimation)
        3. Proper bootstrap for joint inference on location and scale
        4. Correct standard errors and p-values
        """
        
        # Prepare data
        formula = f"{y_var} ~ {' + '.join(x_vars)}"
        n_obs = len(data)
        n_params = len(x_vars) + 1  # Including intercept
        
        # ==========================================
        # Step 1: Estimate Location Parameters (α)
        # ==========================================
        st.info(f"Estimating location parameters at τ = {reference_quantile}...")
        location_model = quantreg(formula, data).fit(q=reference_quantile, vcov='robust')
        alpha = location_model.params
        alpha_se = location_model.bse
        alpha_pvalues = location_model.pvalues
        
        # ==========================================
        # Step 2: Estimate Scale Parameters (δ)
        # ==========================================
        st.info("Estimating scale parameters using symmetric quantiles...")
        
        # Use symmetric quantiles around median
        tau_high = 0.75
        tau_low = 0.25
        
        model_high = quantreg(formula, data).fit(q=tau_high, vcov='robust')
        model_low = quantreg(formula, data).fit(q=tau_low, vcov='robust')
        
        # CRITICAL: Use inverse normal CDF for h(τ)
        h_high = norm.ppf(tau_high)  # Φ^(-1)(0.75) ≈ 0.674
        h_low = norm.ppf(tau_low)    # Φ^(-1)(0.25) ≈ -0.674
        
        # Scale parameters: δ = [β(τ_high) - β(τ_low)] / [h(τ_high) - h(τ_low)]
        delta = (model_high.params - model_low.params) / (h_high - h_low)
        
        # ==========================================
        # Step 3: Bootstrap Inference (Recommended)
        # ==========================================
        if bootstrap:
            st.info(f"Running bootstrap with {n_boot} replications...")
            
            # Storage for bootstrap samples
            boot_alpha = []
            boot_delta = []
            boot_beta = {tau: [] for tau in quantiles}
            
            progress_bar = st.progress(0)
            
            for b in range(n_boot):
                try:
                    # Resample with replacement
                    boot_data = data.sample(n=n_obs, replace=True)
                    
                    # Estimate location
                    boot_loc = quantreg(formula, boot_data).fit(q=reference_quantile)
                    boot_alpha.append(boot_loc.params.values)
                    
                    # Estimate scale
                    boot_high = quantreg(formula, boot_data).fit(q=tau_high)
                    boot_low = quantreg(formula, boot_data).fit(q=tau_low)
                    boot_delta_b = (boot_high.params - boot_low.params) / (h_high - h_low)
                    boot_delta.append(boot_delta_b.values)
                    
                    # MMQR coefficients for all quantiles
                    for tau in quantiles:
                        h_tau = norm.ppf(tau)
                        beta_tau = boot_loc.params + boot_delta_b * h_tau
                        boot_beta[tau].append(beta_tau.values)
                    
                except Exception as e:
                    continue
                
                if (b + 1) % 50 == 0:
                    progress_bar.progress((b + 1) / n_boot)
            
            progress_bar.progress(1.0)
            
            # Convert to arrays
            boot_alpha = np.array(boot_alpha)
            boot_delta = np.array(boot_delta)
            
            # Calculate bootstrap standard errors
            alpha_se_boot = np.std(boot_alpha, axis=0)
            delta_se_boot = np.std(boot_delta, axis=0)
            
            # Calculate bootstrap p-values (two-sided)
            alpha_pvalues_boot = []
            for i, var in enumerate(alpha.index):
                t_stat = alpha[var] / alpha_se_boot[i] if alpha_se_boot[i] > 0 else 0
                p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n_obs - n_params))
                alpha_pvalues_boot.append(p_val)
            
            delta_pvalues_boot = []
            for i, var in enumerate(delta.index):
                t_stat = delta[var] / delta_se_boot[i] if delta_se_boot[i] > 0 else 0
                p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n_obs - n_params))
                delta_pvalues_boot.append(p_val)
            
            # Use bootstrap inference
            alpha_se = pd.Series(alpha_se_boot, index=alpha.index)
            delta_se = pd.Series(delta_se_boot, index=delta.index)
            alpha_pvalues = pd.Series(alpha_pvalues_boot, index=alpha.index)
            delta_pvalues = pd.Series(delta_pvalues_boot, index=delta.index)
            
        else:
            # Delta method for scale parameters (less reliable)
            st.warning("Using delta method for scale inference. Bootstrap is recommended.")
            
            delta_se = {}
            delta_pvalues = {}
            
            for var in delta.index:
                # Variance using delta method
                var_high = model_high.bse[var] ** 2
                var_low = model_low.bse[var] ** 2
                
                # Conservative covariance estimate
                cov_hl = 0.5 * np.sqrt(var_high * var_low)
                
                var_delta = (var_high + var_low - 2 * cov_hl) / ((h_high - h_low) ** 2)
                se_delta = np.sqrt(var_delta)
                
                delta_se[var] = se_delta
                
                # T-statistic and p-value
                t_stat = delta[var] / se_delta if se_delta > 0 else 0
                p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n_obs - n_params))
                delta_pvalues[var] = p_val
            
            delta_se = pd.Series(delta_se)
            delta_pvalues = pd.Series(delta_pvalues)
        
        # ==========================================
        # Step 4: MMQR Coefficients (Analytical)
        # ==========================================
        results = {}
        
        for tau in quantiles:
            # Transform quantile using inverse normal CDF
            h_tau = norm.ppf(tau)
            
            # MMQR coefficients: β(τ) = α + δ · Φ^(-1)(τ)
            beta_tau = alpha + delta * h_tau
            
            # Standard errors using delta method or bootstrap
            if bootstrap and tau in boot_beta and len(boot_beta[tau]) > 0:
                boot_beta_array = np.array(boot_beta[tau])
                beta_se = pd.Series(np.std(boot_beta_array, axis=0), index=beta_tau.index)
                
                # Bootstrap p-values
                beta_pvalues = []
                for i, var in enumerate(beta_tau.index):
                    t_stat = beta_tau[var] / beta_se[var] if beta_se[var] > 0 else 0
                    p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n_obs - n_params))
                    beta_pvalues.append(p_val)
                beta_pvalues = pd.Series(beta_pvalues, index=beta_tau.index)
            else:
                # Delta method: Var[β(τ)] = Var[α] + h(τ)² · Var[δ] + 2·h(τ)·Cov[α,δ]
                # Simplified: assume Cov[α,δ] ≈ 0
                beta_se = np.sqrt(alpha_se**2 + (h_tau**2) * delta_se**2)
                
                beta_pvalues = []
                for var in beta_tau.index:
                    t_stat = beta_tau[var] / beta_se[var] if beta_se[var] > 0 else 0
                    p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n_obs - n_params))
                    beta_pvalues.append(p_val)
                beta_pvalues = pd.Series(beta_pvalues, index=beta_tau.index)
            
            # Confidence intervals
            ci_lower = beta_tau - 1.96 * beta_se
            ci_upper = beta_tau + 1.96 * beta_se
            
            results[tau] = {
                'coefficients': beta_tau,
                'std_errors': beta_se,
                'pvalues': beta_pvalues,
                'conf_int_lower': ci_lower,
                'conf_int_upper': ci_upper,
                'h_tau': h_tau
            }
        
        return {
            'results': results,
            'location': {
                'params': alpha,
                'std_errors': alpha_se,
                'pvalues': alpha_pvalues
            },
            'scale': {
                'params': delta,
                'std_errors': delta_se,
                'pvalues': delta_pvalues
            },
            'quantiles': quantiles,
            'n_obs': n_obs,
            'reference_quantile': reference_quantile
        }

    # ==========================================
    # Run MMQR Estimation
    # ==========================================
    try:
        mmqr_output = mmqr_estimation(
            df, dep_var, indep_vars, quantiles, 
            reference_quantile, bootstrap_inference, n_bootstrap
        )
        
        results = mmqr_output['results']
        location = mmqr_output['location']
        scale = mmqr_output['scale']
        
        st.success("✅ MMQR estimation completed successfully!")
        
        # ==========================================
        # Table 1: Location Parameters
        # ==========================================
        st.subheader(f"Table 1: Location Parameters (α) at τ = {reference_quantile}")
        
        location_data = []
        for var in location['params'].index:
            coef = location['params'][var]
            se = location['std_errors'][var]
            pval = location['pvalues'][var]
            t_stat = coef / se if se > 0 else 0
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            
            location_data.append({
                'Variable': var,
                'Coefficient': f"{coef:.4f}",
                'Std. Error': f"{se:.4f}",
                'T-Statistic': f"{t_stat:.3f}",
                'P-Value': f"{pval:.4f}",
                '95% CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                'Sig.': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            })
        
        location_df = pd.DataFrame(location_data)
        st.dataframe(location_df, use_container_width=True)
        
        # ==========================================
        # Table 2: Scale Parameters
        # ==========================================
        st.subheader("Table 2: Scale Parameters (δ)")
        
        scale_data = []
        for var in scale['params'].index:
            coef = scale['params'][var]
            se = scale['std_errors'][var]
            pval = scale['pvalues'][var]
            t_stat = coef / se if se > 0 else 0
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            
            # Economic interpretation
            if var in location['params']:
                loc_coef = location['params'][var]
                if abs(loc_coef) > 1e-6:
                    relative_effect = abs(coef) / abs(loc_coef)
                    interpretation = (
                        'Strong heterogeneity' if pval < 0.05 and relative_effect > 0.5 else
                        'Moderate heterogeneity' if pval < 0.05 and relative_effect > 0.2 else
                        'Weak heterogeneity' if pval < 0.05 else
                        'No significant heterogeneity'
                    )
                else:
                    interpretation = 'Location near zero'
            else:
                interpretation = 'N/A'
            
            scale_data.append({
                'Variable': var,
                'Coefficient': f"{coef:.4f}",
                'Std. Error': f"{se:.4f}",
                'T-Statistic': f"{t_stat:.3f}",
                'P-Value': f"{pval:.4f}",
                '95% CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                'Sig.': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else '',
                'Interpretation': interpretation
            })
        
        scale_df = pd.DataFrame(scale_data)
        st.dataframe(scale_df, use_container_width=True)
        
        # ==========================================
        # Table 3: MMQR Coefficients
        # ==========================================
        st.subheader("Table 3: MMQR Coefficients β(τ) = α + δ·Φ⁻¹(τ)")
        
        var_names = results[quantiles[0]]['coefficients'].index
        
        mmqr_table = []
        for var in var_names:
            row = {'Variable': var}
            
            for tau in quantiles:
                coef = results[tau]['coefficients'][var]
                se = results[tau]['std_errors'][var]
                pval = results[tau]['pvalues'][var]
                
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                row[f'τ={tau}'] = f"{coef:.4f}{sig}"
                row[f'SE({tau})'] = f"({se:.4f})"
            
            mmqr_table.append(row)
        
        mmqr_df = pd.DataFrame(mmqr_table)
        st.dataframe(mmqr_df, use_container_width=True)
        
        st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.1")
        
        # ==========================================
        # Interpretation Guide
        # ==========================================
        st.subheader("📊 Interpretation Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Location Parameters (α):**")
            st.write("- Central tendency effect at reference quantile")
            st.write("- Baseline impact across the distribution")
            
            st.write("**Scale Parameters (δ):**")
            st.write("- Heterogeneity in effects across quantiles")
            st.write("- Positive δ: effects increase at higher quantiles")
            st.write("- Negative δ: effects decrease at higher quantiles")
        
        with col2:
            st.write("**MMQR Coefficients β(τ):**")
            st.write("- Marginal effects at specific quantiles")
            st.write("- β(τ) = α + δ·Φ⁻¹(τ)")
            st.write("- Captures full distributional heterogeneity")
        
        # Significant heterogeneity summary
        st.write("**Heterogeneity Summary:**")
        sig_hetero = [var for var in scale['params'].index 
                     if var != 'Intercept' and scale['pvalues'][var] < 0.1]
        
        if sig_hetero:
            st.success(f"Significant heterogeneity detected in: **{', '.join(sig_hetero)}**")
            for var in sig_hetero:
                direction = "increasing" if scale['params'][var] > 0 else "decreasing"
                st.write(f"- **{var}**: {direction} effects across quantiles (p={scale['pvalues'][var]:.4f})")
        else:
            st.info("No significant heterogeneity detected. Effects are relatively constant across quantiles.")
        
        # ==========================================
        # Visualizations
        # ==========================================
        st.subheader("Figure 1: MMQR Coefficient Trajectories")
        
        plot_vars = [v for v in var_names if v != 'Intercept']
        
        for var in plot_vars:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Extract coefficients and confidence intervals
            taus = quantiles
            coefs = [results[tau]['coefficients'][var] for tau in taus]
            ci_lower = [results[tau]['conf_int_lower'][var] for tau in taus]
            ci_upper = [results[tau]['conf_int_upper'][var] for tau in taus]
            
            # Plot MMQR trajectory
            ax.plot(taus, coefs, 'o-', linewidth=2.5, markersize=8, 
                   color='#2E86AB', label='MMQR Coefficients')
            
            # Confidence band
            ax.fill_between(taus, ci_lower, ci_upper, alpha=0.2, color='#2E86AB', 
                           label='95% CI')
            
            # Location parameter (horizontal line)
            loc_val = location['params'][var]
            ax.axhline(y=loc_val, color='red', linestyle='--', linewidth=2, 
                      alpha=0.7, label=f'Location (α={loc_val:.3f})')
            
            # Scale significance
            scale_pval = scale['pvalues'][var]
            scale_coef = scale['params'][var]
            scale_sig = '***' if scale_pval < 0.01 else '**' if scale_pval < 0.05 else '*' if scale_pval < 0.1 else 'ns'
            
            # Zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            # Labels and title
            ax.set_xlabel('Quantile (τ)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Coefficient Estimate', fontsize=13, fontweight='bold')
            ax.set_title(f'MMQR Trajectory: {var}\n(Scale: δ={scale_coef:.4f}, p={scale_pval:.4f} {scale_sig})', 
                        fontsize=14, fontweight='bold')
            
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add significance markers
            for i, tau in enumerate(taus):
                pval = results[tau]['pvalues'][var]
                if pval < 0.1:
                    sig_marker = '***' if pval < 0.01 else '**' if pval < 0.05 else '*'
                    ax.text(tau, coefs[i], sig_marker, ha='center', va='bottom', 
                           fontsize=12, fontweight='bold', color='red')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # ==========================================
        # Download Results
        # ==========================================
        st.subheader("📥 Download Complete Results")
        
        # Comprehensive results table
        download_data = []
        
        # Location parameters
        for var in location['params'].index:
            download_data.append({
                'Variable': var,
                'Parameter_Type': 'Location',
                'Coefficient': location['params'][var],
                'Std_Error': location['std_errors'][var],
                'P_Value': location['pvalues'][var],
                'Quantile': reference_quantile,
                'h_tau': 'N/A'
            })
        
        # Scale parameters
        for var in scale['params'].index:
            download_data.append({
                'Variable': var,
                'Parameter_Type': 'Scale',
                'Coefficient': scale['params'][var],
                'Std_Error': scale['std_errors'][var],
                'P_Value': scale['pvalues'][var],
                'Quantile': 'N/A',
                'h_tau': 'N/A'
            })
        
        # MMQR coefficients
        for tau in quantiles:
            for var in results[tau]['coefficients'].index:
                download_data.append({
                    'Variable': var,
                    'Parameter_Type': 'MMQR',
                    'Coefficient': results[tau]['coefficients'][var],
                    'Std_Error': results[tau]['std_errors'][var],
                    'P_Value': results[tau]['pvalues'][var],
                    'Quantile': tau,
                    'h_tau': results[tau]['h_tau']
                })
        
        download_df = pd.DataFrame(download_data)
        csv_data = download_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Complete MMQR Results (CSV)",
            data=csv_data,
            file_name="MMQR_Complete_Results_Corrected.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Summary statistics
        st.subheader("📈 Model Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Observations", mmqr_output['n_obs'])
        with col2:
            st.metric("Reference Quantile", reference_quantile)
        with col3:
            sig_scale_count = sum(1 for p in scale['pvalues'] if p < 0.05)
            st.metric("Significant Scale Parameters", f"{sig_scale_count}/{len(scale['params'])}")
        
    except Exception as e:
        st.error(f"❌ MMQR estimation failed: {str(e)}")
        st.write("**Error details:**")
        st.exception(e)

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
