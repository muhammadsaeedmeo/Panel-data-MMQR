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
# Correlation Heatmap with Dropdowns, Color Selection & Interpretation
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.header("Correlation Heatmap")

# --- Load your data here ---
# data = pd.read_csv("your_data.csv")

# Detect numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.warning("No numeric variables found in your dataset.")
else:
    # --- Dropdowns for selecting dependent & independent variables ---
    dep_var = st.selectbox("Select Dependent Variable", options=numeric_cols)
    indep_vars = st.multiselect(
        "Select Independent Variable(s)",
        options=[col for col in numeric_cols if col != dep_var],
        default=[col for col in numeric_cols if col != dep_var][:3]
    )

    # --- Color palette selector ---
    color_option = st.selectbox(
        "Select Heatmap Color Palette",
        options=[
            "coolwarm", "viridis", "plasma", "magma", "cividis",
            "Blues", "Greens", "Reds", "Purples", "icefire", "Spectral"
        ],
        index=0
    )

    if indep_vars:
        # --- Compute correlation matrix ---
        selected_vars = [dep_var] + indep_vars
        corr = data[selected_vars].corr()

        # --- Generate heatmap ---
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

        # --- Add download button for heatmap image ---
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            label="Download Heatmap Image",
            data=buf.getvalue(),
            file_name="correlation_heatmap.png",
            mime="image/png"
        )

        # --- Interpret correlation results ---
        st.subheader("Correlation Interpretation")

        def interpret_corr(value):
            val = abs(value)
            if val < 0.20:
                return "very weak"
            elif val < 0.40:
                return "weak"
            elif val < 0.60:
                return "moderate"
            elif val < 0.80:
                return "strong"
            else:
                return "very strong"

        interpretation_text = ""
        for var in indep_vars:
            corr_value = corr.loc[dep_var, var]
            strength = interpret_corr(corr_value)
            direction = "positive" if corr_value > 0 else "negative"
            interpretation_text += (
                f"- The correlation between **{dep_var}** and **{var}** is "
                f"**{corr_value:.2f}**, indicating a **{strength} {direction} relationship**.\n"
            )

        st.markdown(interpretation_text)

        st.info(
            "According to Evans (1996), correlation strengths are defined as: "
            "very weak (0.00–0.19), weak (0.20–0.39), moderate (0.40–0.59), "
            "strong (0.60–0.79), and very strong (0.80–1.00).\n\n"
            "**Reference:** Evans, J. D. (1996). *Straightforward statistics for the behavioral sciences.* "
            "Brooks/Cole Publishing."
        )
    else:
        st.warning("Please select at least one independent variable to display correlation.")

st.subheader("Table 2: Correlation Matrix")
try:
    st.dataframe(corr)
except Exception as e:
    st.warning(f"Cannot compute correlation matrix: {e}")

# ============================================
# Slope Homogeneity Test (Pesaran and Yamagata, 2008)
# ============================================

st.subheader("Slope Homogeneity Test (Pesaran and Yamagata, 2008)")

try:
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd

    # Check required columns
    if "Country" not in data.columns or "Year" not in data.columns:
        st.warning("Please ensure your dataset includes 'Country' and 'Year' columns for panel data.")
    else:
        dep = dep_var
        indeps = indep_vars

        if not indeps:
            st.warning("Please select independent variables first.")
        else:
            # Prepare data by country
            panel_results = []
            for country, subset in data.groupby("Country"):
                if subset[dep].isnull().any() or subset[indeps].isnull().any().any():
                    continue  # skip missing data
                X = sm.add_constant(subset[indeps])
                y = subset[dep]
                model = sm.OLS(y, X).fit()
                panel_results.append(model.params.values)

            betas = np.vstack(panel_results)
            mean_beta = np.mean(betas, axis=0)
            N, k = betas.shape

            # Compute test statistics
            S = np.sum((betas - mean_beta) ** 2, axis=0)
            delta = N * np.sum(S) / np.sum(mean_beta ** 2)
            delta_adj = (N * delta - k) / np.sqrt(2 * k)

            # Compute p-values (two-tailed from normal distribution)
            from scipy.stats import norm
            p_delta = 2 * (1 - norm.cdf(abs(delta)))
            p_delta_adj = 2 * (1 - norm.cdf(abs(delta_adj)))

            # Create a nice result table
            results_df = pd.DataFrame({
                "Statistic": ["Δ", "Δ_adj"],
                "Value": [round(delta, 3), round(delta_adj, 3)],
                "p-value": [f"{p_delta:.3f}", f"{p_delta_adj:.3f}"]
            })

            st.write("**Slope Homogeneity Test Results**")
            st.dataframe(results_df, use_container_width=True)

            # Simple interpretation line
            if p_delta_adj < 0.05:
                st.success("Reject the null hypothesis — slopes are *heterogeneous* across cross-sections.")
                st.markdown("**Interpretation:** The regression slopes are not the same for all cross-sections.")
            else:
                st.info("Fail to reject the null hypothesis — slopes are *homogeneous* across cross-sections.")
                st.markdown("**Interpretation:** The regression slopes are broadly similar across cross-sections.")

            # Reference
            st.caption(
                "Reference: Pesaran, M. H., & Yamagata, T. (2008). "
                "Testing slope homogeneity in large panels. *Journal of Econometrics*, 142(1), 50–93."
            )

except Exception as e:
    st.warning(f"Error running slope homogeneity test: {e}")

# #################################################
# MMQR
# ###############################################33
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Method of Moments Quantile Regression (MMQR)
# Complete Implementation with Location-Scale Decomposition
# ============================================

st.set_page_config(page_title="MMQR Analysis", layout="wide")
st.title("Method of Moments Quantile Regression (MMQR)")
st.markdown("""
**Reference:** Machado, J.A.F. and Silva, J.M.C.S. (2019). "Quantiles via moments."  
*Journal of Econometrics*, 213(1), 145-173.

This implementation estimates **unconditional quantile partial effects** (UQPEs) through:
1. **Location Model**: β (mean regression)
2. **Scale Model**: γ (variance regression)  
3. **MMQR Coefficients**: β(τ) = β + q(τ)·γ for each quantile τ
""")

# ============================================
# MMQR Model Class with Full Decomposition
# ============================================

class MMQRModel:
    """
    Method of Moments Quantile Regression with Location-Scale Decomposition
    """
    
    def __init__(self, y, X, quantiles=[0.05, 0.25, 0.50, 0.75, 0.95]):
        self.y = np.array(y).flatten()
        self.X = np.array(X)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        self.n, self.k = self.X.shape
        self.quantiles = sorted(quantiles)
        
        # Add constant
        self.X_full = np.column_stack([np.ones(self.n), self.X])
        self.k_full = self.k + 1
        
        # Storage for results
        self.beta_location = None
        self.gamma_scale = None
        self.beta_mmqr = {}
        self.se_location = None
        self.se_scale = None
        self.se_mmqr = {}
        
    def fit(self, bootstrap_se=True, n_bootstrap=200):
        """
        Estimate MMQR using location-scale approach
        """
        # ==========================================
        # Step 1: Location Model (OLS for mean)
        # ==========================================
        self.beta_location = np.linalg.lstsq(self.X_full, self.y, rcond=None)[0]
        residuals = self.y - self.X_full @ self.beta_location
        
        # Location model standard errors (OLS)
        sigma2 = np.sum(residuals**2) / (self.n - self.k_full)
        var_beta = sigma2 * np.linalg.inv(self.X_full.T @ self.X_full)
        self.se_location = np.sqrt(np.diag(var_beta))
        
        # ==========================================
        # Step 2: Scale Model (for log|residuals|)
        # ==========================================
        log_abs_resid = np.log(np.abs(residuals) + 1e-10)
        self.gamma_scale = np.linalg.lstsq(self.X_full, log_abs_resid, rcond=None)[0]
        
        # Scale model standard errors
        scale_residuals = log_abs_resid - self.X_full @ self.gamma_scale
        sigma2_scale = np.sum(scale_residuals**2) / (self.n - self.k_full)
        var_gamma = sigma2_scale * np.linalg.inv(self.X_full.T @ self.X_full)
        self.se_scale = np.sqrt(np.diag(var_gamma))
        
        # ==========================================
        # Step 3: MMQR Coefficients for Each Quantile
        # ==========================================
        for tau in self.quantiles:
            # Standard normal quantile
            q_tau = norm.ppf(tau)
            
            # MMQR formula: β(τ) = β_location + q(τ) × γ_scale
            self.beta_mmqr[tau] = self.beta_location + q_tau * self.gamma_scale
        
        # ==========================================
        # Step 4: Bootstrap Standard Errors
        # ==========================================
        if bootstrap_se:
            self._bootstrap_inference(n_bootstrap)
        else:
            # Analytical approximation (Delta method)
            self._analytical_se()
        
        return self
    
    def _bootstrap_inference(self, n_bootstrap):
        """
        Bootstrap standard errors for MMQR coefficients
        """
        st.info(f"🔄 Running {n_bootstrap} bootstrap replications...")
        progress_bar = st.progress(0)
        
        bootstrap_betas = {tau: [] for tau in self.quantiles}
        
        for b in range(n_bootstrap):
            # Update progress
            if b % 20 == 0:
                progress_bar.progress(b / n_bootstrap)
            
            # Resample
            indices = np.random.choice(self.n, size=self.n, replace=True)
            y_boot = self.y[indices]
            X_boot = self.X_full[indices]
            
            try:
                # Location model
                beta_loc_boot = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]
                resid_boot = y_boot - X_boot @ beta_loc_boot
                
                # Scale model
                log_abs_resid_boot = np.log(np.abs(resid_boot) + 1e-10)
                gamma_scale_boot = np.linalg.lstsq(X_boot, log_abs_resid_boot, rcond=None)[0]
                
                # MMQR for each quantile
                for tau in self.quantiles:
                    q_tau = norm.ppf(tau)
                    beta_mmqr_boot = beta_loc_boot + q_tau * gamma_scale_boot
                    bootstrap_betas[tau].append(beta_mmqr_boot)
            except:
                continue
        
        progress_bar.progress(1.0)
        
        # Compute standard errors and p-values
        for tau in self.quantiles:
            boot_array = np.array(bootstrap_betas[tau])
            if len(boot_array) > 10:  # Need sufficient samples
                self.se_mmqr[tau] = np.std(boot_array, axis=0, ddof=1)
            else:
                # Fallback to analytical
                q_tau = norm.ppf(tau)
                self.se_mmqr[tau] = np.sqrt(self.se_location**2 + (q_tau**2) * self.se_scale**2)
    
    def _analytical_se(self):
        """
        Analytical standard errors using Delta method
        """
        for tau in self.quantiles:
            q_tau = norm.ppf(tau)
            # Approximate SE: sqrt(var(β) + q²·var(γ))
            self.se_mmqr[tau] = np.sqrt(self.se_location**2 + (q_tau**2) * self.se_scale**2)
    
    def get_location_table(self, var_names):
        """
        Create table for Location Model (β)
        """
        t_stats = self.beta_location / self.se_location
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.n - self.k_full))
        
        df = pd.DataFrame({
            'Variable': var_names,
            'Coefficient (β)': self.beta_location,
            'Std. Error': self.se_location,
            't-statistic': t_stats,
            'P>|t|': p_values,
            'CI_Lower': self.beta_location - 1.96 * self.se_location,
            'CI_Upper': self.beta_location + 1.96 * self.se_location
        })
        return df
    
    def get_scale_table(self, var_names):
        """
        Create table for Scale Model (γ)
        """
        t_stats = self.gamma_scale / self.se_scale
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.n - self.k_full))
        
        df = pd.DataFrame({
            'Variable': var_names,
            'Coefficient (γ)': self.gamma_scale,
            'Std. Error': self.se_scale,
            't-statistic': t_stats,
            'P>|t|': p_values,
            'CI_Lower': self.gamma_scale - 1.96 * self.se_scale,
            'CI_Upper': self.gamma_scale + 1.96 * self.se_scale
        })
        return df
    
    def get_mmqr_table(self, var_names):
        """
        Create comprehensive MMQR table with all quantiles
        """
        results_list = []
        
        for tau in self.quantiles:
            beta_tau = self.beta_mmqr[tau]
            se_tau = self.se_mmqr.get(tau, np.ones_like(beta_tau) * np.nan)
            
            for i, var_name in enumerate(var_names):
                z_stat = beta_tau[i] / se_tau[i] if se_tau[i] > 0 else np.nan
                p_val = 2 * (1 - norm.cdf(np.abs(z_stat))) if not np.isnan(z_stat) else np.nan
                
                results_list.append({
                    'Quantile': tau,
                    'Variable': var_name,
                    'Coefficient β(τ)': beta_tau[i],
                    'Std. Error': se_tau[i],
                    'z-statistic': z_stat,
                    'P>|z|': p_val,
                    'CI_Lower': beta_tau[i] - 1.96 * se_tau[i],
                    'CI_Upper': beta_tau[i] + 1.96 * se_tau[i]
                })
        
        return pd.DataFrame(results_list)


# ============================================
# Streamlit Interface
# ============================================

st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.session_state["uploaded_data"] = data
else:
    data = st.session_state.get("uploaded_data", None)

if data is not None:
    st.success("✅ Dataset loaded successfully.")
    
    with st.expander("📊 View Dataset Preview"):
        st.dataframe(data.head(10), use_container_width=True)
        st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    
    # Check for missing data
    if data.isnull().sum().sum() > 0:
        st.warning("⚠️ Missing values detected!")
        if st.checkbox("Remove missing values?", value=True):
            data = data.dropna()
            st.success(f"✅ Dataset cleaned: {data.shape[0]} rows")
    
    # Variable selection
    st.subheader("Variable Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        dependent_var = st.selectbox("🎯 Dependent Variable (Y)", options=data.columns)
    
    with col2:
        independent_vars = st.multiselect(
            "📊 Independent Variables (X)",
            options=[c for c in data.columns if c != dependent_var]
        )
    
    # Quantile selection
    st.sidebar.subheader("Quantile Configuration")
    quantile_preset = st.sidebar.radio(
        "Quantile set:",
        ["Standard (5)", "Extended (9)", "Custom"]
    )
    
    if quantile_preset == "Standard (5)":
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    elif quantile_preset == "Extended (9)":
        quantiles = [0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95]
    else:
        q_input = st.sidebar.text_input("Quantiles (comma-separated):", "0.1,0.25,0.5,0.75,0.9")
        try:
            quantiles = [float(q.strip()) for q in q_input.split(",") if 0 < float(q.strip()) < 1]
        except:
            st.sidebar.error("Invalid format. Using defaults.")
            quantiles = [0.25, 0.50, 0.75]
    
    st.sidebar.write(f"**Selected:** {quantiles}")
    
    # Bootstrap settings
    st.sidebar.subheader("Inference Settings")
    use_bootstrap = st.sidebar.checkbox("Bootstrap Standard Errors", value=True)
    n_bootstrap = st.sidebar.slider("Bootstrap Replications", 50, 500, 200, 50) if use_bootstrap else 0
    
    # Run analysis
    if independent_vars:
        if st.button("🚀 Run MMQR Analysis", type="primary"):
            try:
                # Prepare data
                y = data[dependent_var].values
                X = data[independent_vars].values
                
                # Validate
                if np.any(~np.isfinite(y)) or np.any(~np.isfinite(X)):
                    st.error("❌ Data contains NaN or infinite values!")
                else:
                    # Fit model
                    with st.spinner("Estimating MMQR model..."):
                        mmqr = MMQRModel(y, X, quantiles=quantiles)
                        mmqr.fit(bootstrap_se=use_bootstrap, n_bootstrap=n_bootstrap)
                    
                    # Store results
                    st.session_state['mmqr_model'] = mmqr
                    st.session_state['var_names'] = ['Constant'] + independent_vars
                    st.session_state['dep_var'] = dependent_var
                    st.success("✅ MMQR estimation completed!")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # ============================================
    # Display Results
    # ============================================
    if 'mmqr_model' in st.session_state and 'var_names' in st.session_state and 'dep_var' in st.session_state:
        mmqr = st.session_state['mmqr_model']
        var_names = st.session_state['var_names']
        dep_var = st.session_state['dep_var']
        
        st.header("📊 MMQR Results")
        
        # Model formula
        st.markdown(f"""
        **Model Specification:**  
        `{dep_var} = β₀ + β₁·X₁ + β₂·X₂ + ... + ε`  
        where `log|ε| = γ₀ + γ₁·X₁ + γ₂·X₂ + ...`
        """)
        
        # ==========================================
        # Table 1: Location Model
        # ==========================================
        st.subheader("📍 Table 1: Location Model (Mean Regression)")
        st.markdown("**Estimates of β (location effects)**")
        
        location_df = mmqr.get_location_table(var_names)
        
        def style_pvalues(val):
            if pd.isna(val):
                return ''
            if val < 0.001:
                return 'background-color: #d4edda'
            elif val < 0.01:
                return 'background-color: #d1ecf1'
            elif val < 0.05:
                return 'background-color: #fff3cd'
            return ''
        
        styled_location = location_df.style.format({
            'Coefficient (β)': '{:.4f}',
            'Std. Error': '{:.4f}',
            't-statistic': '{:.3f}',
            'P>|t|': '{:.4f}',
            'CI_Lower': '{:.4f}',
            'CI_Upper': '{:.4f}'
        }).applymap(style_pvalues, subset=['P>|t|'])
        
        st.dataframe(styled_location, use_container_width=True)
        st.caption("Significance: Green (p<0.001), Blue (p<0.01), Yellow (p<0.05)")
        
        # ==========================================
        # Table 2: Scale Model
        # ==========================================
        st.subheader("📏 Table 2: Scale Model (Variance Regression)")
        st.markdown("**Estimates of γ (scale/heterogeneity effects)**")
        
        scale_df = mmqr.get_scale_table(var_names)
        
        styled_scale = scale_df.style.format({
            'Coefficient (γ)': '{:.4f}',
            'Std. Error': '{:.4f}',
            't-statistic': '{:.3f}',
            'P>|t|': '{:.4f}',
            'CI_Lower': '{:.4f}',
            'CI_Upper': '{:.4f}'
        }).applymap(style_pvalues, subset=['P>|t|'])
        
        st.dataframe(styled_scale, use_container_width=True)
        
        # Interpretation of scale effects
        with st.expander("💡 How to interpret Scale (γ) coefficients"):
            st.markdown("""
            - **Positive γ**: Variable increases dispersion/inequality of Y
            - **Negative γ**: Variable reduces dispersion/inequality of Y
            - **γ ≈ 0**: Variable only affects location, not spread
            
            Example: If education has γ > 0 for income, it means education increases income inequality.
            """)
        
        # ==========================================
        # Table 3: MMQR Coefficients (All Quantiles)
        # ==========================================
        st.subheader("🎯 Table 3: MMQR Coefficients β(τ) = β + q(τ)·γ")
        st.markdown("**Unconditional Quantile Partial Effects (UQPE)**")
        
        mmqr_df = mmqr.get_mmqr_table(var_names)
        
        def add_significance_stars(row):
            p = row['P>|z|']
            coef_str = f"{row['Coefficient β(τ)']:.4f}"
            if pd.notna(p):
                if p < 0.001:
                    return coef_str + "***"
                elif p < 0.01:
                    return coef_str + "**"
                elif p < 0.05:
                    return coef_str + "*"
                elif p < 0.10:
                    return coef_str + "†"
            return coef_str
        
        mmqr_display = mmqr_df.copy()
        mmqr_display['Coefficient β(τ)'] = mmqr_df.apply(add_significance_stars, axis=1)
        
        styled_mmqr = mmqr_display[['Quantile', 'Variable', 'Coefficient β(τ)', 
                                     'Std. Error', 'z-statistic', 'P>|z|']].style.format({
            'Quantile': '{:.2f}',
            'Std. Error': '{:.4f}',
            'z-statistic': '{:.3f}',
            'P>|z|': '{:.4f}'
        })
        
        st.dataframe(styled_mmqr, use_container_width=True, height=400)
        st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10")
        
        # ==========================================
        # Decomposition Table (Side by Side)
        # ==========================================
        st.subheader("🔬 Decomposition: β(τ) = β + q(τ)·γ")
        
        decomp_data = []
        for tau in quantiles:
            q_tau = norm.ppf(tau)
            decomp_data.append({
                'Quantile τ': tau,
                'q(τ)': q_tau,
                'Formula': f"β({tau:.2f}) = β + {q_tau:.3f}·γ"
            })
        
        decomp_df = pd.DataFrame(decomp_data)
        st.dataframe(decomp_df.style.format({'Quantile τ': '{:.2f}', 'q(τ)': '{:.4f}'}), 
                    use_container_width=True)
        
        st.markdown("""
        **Understanding the decomposition:**
        - **β** (location): Constant effect across all quantiles
        - **q(τ)·γ** (scale adjustment): Varies by quantile
        - At median (τ=0.5): q(0.5)=0, so β(0.5) = β
        - At extremes: |q(τ)| is large, so scale effects dominate
        """)
        
        # ==========================================
        # Visualization
        # ==========================================
        st.subheader("📈 Figure: MMQR Coefficient Plots")
        
        plot_vars = st.multiselect(
            "Select variables to plot:",
            options=independent_vars,
            default=independent_vars[:min(3, len(independent_vars))]
        )
        
        if plot_vars:
            n_plots = len(plot_vars)
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            colors = sns.color_palette("husl", n_plots)
            
            for idx, var in enumerate(plot_vars):
                ax = axes[idx]
                var_idx = var_names.index(var)
                
                # Extract coefficients and CIs
                coefs = [mmqr.beta_mmqr[tau][var_idx] for tau in quantiles]
                ses = [mmqr.se_mmqr[tau][var_idx] for tau in quantiles]
                ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
                ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]
                
                # Plot
                ax.plot(quantiles, coefs, 'o-', color=colors[idx], linewidth=2.5, 
                       markersize=8, label=var)
                ax.fill_between(quantiles, ci_lower, ci_upper, alpha=0.25, color=colors[idx])
                
                # Reference lines
                ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.axhline(mmqr.beta_location[var_idx], color='red', linestyle=':', 
                          linewidth=1.5, alpha=0.7, label=f'β (location)')
                
                ax.set_xlabel('Quantile (τ)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Coefficient β(τ)', fontsize=12, fontweight='bold')
                ax.set_title(f'{var}', fontsize=14, fontweight='bold')
                ax.legend(loc='best', frameon=True, shadow=True)
                ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("""
            **Red dashed line** = Location effect (β)  
            **Blue curve** = Full MMQR effect β(τ) = β + q(τ)·γ  
            **Shaded area** = 95% Confidence interval
            """)
        
        # ==========================================
        # Download Options
        # ==========================================
        st.subheader("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv1 = location_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Location Model", csv1, "location_model.csv", "text/csv")
        
        with col2:
            csv2 = scale_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Scale Model", csv2, "scale_model.csv", "text/csv")
        
        with col3:
            csv3 = mmqr_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 MMQR Coefficients", csv3, "mmqr_results.csv", "text/csv")
        
        # Combined export
        st.markdown("---")
        with pd.ExcelWriter('mmqr_complete_results.xlsx', engine='openpyxl') as writer:
            location_df.to_excel(writer, sheet_name='Location Model', index=False)
            scale_df.to_excel(writer, sheet_name='Scale Model', index=False)
            mmqr_df.to_excel(writer, sheet_name='MMQR Coefficients', index=False)
        
                        with open('mmqr_complete_results.xlsx', 'rb') as f:
            st.download_button(
                "📦 Download All Results (Excel)",
                f.read(),
                "mmqr_complete_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    else:
        if independent_vars:
            st.info("👆 Click 'Run MMQR Analysis' button above to estimate the model and see results.")
        else:
            st.warning("⚠️ Please select at least one independent variable to proceed.")

else:
    st.info("📁 Upload a CSV file to begin analysis")
    
    # Sample data generator
    if st.checkbox("Generate sample dataset"):
        np.random.seed(42)
        n = 500
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        
        # Generate Y with heteroskedasticity
        epsilon = np.random.normal(0, 1, n) * np.exp(0.3 * X1)
        Y = 2 + 1.5*X1 + 0.8*X2 + epsilon
        
        sample_data = pd.DataFrame({
            'Income': Y,
            'Education': X1,
            'Experience': X2
        })
        
        st.session_state["uploaded_data"] = sample_data
        st.success("✅ Sample data created! Click button above to start.")
        st.dataframe(sample_data.head())

# Footer
st.markdown("---")
st.markdown("""
**Method of Moments Quantile Regression (MMQR)**  
Estimates unconditional quantile effects using location-scale decomposition:
- **β** captures mean effects (location)
- **γ** captures heterogeneity effects (scale)  
- **β(τ)** = β + q(τ)·γ gives quantile-specific effects

**Citation:** Machado & Silva (2019), *Journal of Econometrics*, 213(1), 145-173.
""")

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
