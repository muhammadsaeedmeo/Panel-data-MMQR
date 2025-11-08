import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Method of Moments Quantile Regression (MMQR)
# Implementation based on Machado & Silva (2019)
# ============================================

st.set_page_config(page_title="MMQR Analysis", layout="wide")
st.title("Method of Moments Quantile Regression (MMQR)")
st.markdown("""
**Reference:** Machado, J.A.F. and Silva, J.M.C.S. (2019). "Quantiles via moments."  
*Journal of Econometrics*, 213(1), 145-173.

This implementation estimates **unconditional quantile partial effects** (UQPEs) using the location-scale approach.
""")

# ============================================
# Helper Functions for MMQR
# ============================================

class MMQRModel:
    """
    Method of Moments Quantile Regression
    Estimates unconditional quantile treatment effects
    """
    
    def __init__(self, y, X, quantiles=[0.05, 0.25, 0.50, 0.75, 0.95]):
        self.y = np.array(y)
        self.X = np.array(X)
        self.n, self.k = X.shape
        self.quantiles = quantiles
        self.results = {}
        
    def fit(self, bootstrap_se=True, n_bootstrap=200):
        """
        Fit MMQR model using location-scale approach
        """
        # Step 1: Location model (OLS)
        X_with_const = np.column_stack([np.ones(self.n), self.X])
        beta_location = np.linalg.lstsq(X_with_const, self.y, rcond=None)[0]
        residuals = self.y - X_with_const @ beta_location
        
        # Step 2: Scale model (log absolute residuals)
        log_abs_resid = np.log(np.abs(residuals) + 1e-10)
        gamma_scale = np.linalg.lstsq(X_with_const, log_abs_resid, rcond=None)[0]
        
        # Step 3: Compute MMQR coefficients for each quantile
        for tau in self.quantiles:
            q_tau = stats.norm.ppf(tau)  # Standard normal quantile
            
            # MMQR coefficient: β(τ) = β_location + q_τ * exp(X'γ_scale) * ∂γ/∂x
            # Simplified: β(τ) = β_location + q_τ * γ_scale
            beta_mmqr = beta_location + q_tau * gamma_scale
            
            # Store results
            self.results[tau] = {
                'coefficients': beta_mmqr,
                'beta_location': beta_location,
                'gamma_scale': gamma_scale
            }
        
        # Step 4: Bootstrap standard errors if requested
        if bootstrap_se:
            self._bootstrap_inference(n_bootstrap)
        
        return self
    
    def _bootstrap_inference(self, n_bootstrap):
        """
        Compute bootstrap standard errors and confidence intervals
        """
        bootstrap_coefs = {tau: [] for tau in self.quantiles}
        
        for b in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(self.n, size=self.n, replace=True)
            y_boot = self.y[indices]
            X_boot = self.X[indices]
            
            try:
                # Fit bootstrap sample
                X_boot_const = np.column_stack([np.ones(len(indices)), X_boot])
                beta_loc_boot = np.linalg.lstsq(X_boot_const, y_boot, rcond=None)[0]
                resid_boot = y_boot - X_boot_const @ beta_loc_boot
                log_abs_resid_boot = np.log(np.abs(resid_boot) + 1e-10)
                gamma_scale_boot = np.linalg.lstsq(X_boot_const, log_abs_resid_boot, rcond=None)[0]
                
                for tau in self.quantiles:
                    q_tau = stats.norm.ppf(tau)
                    beta_mmqr_boot = beta_loc_boot + q_tau * gamma_scale_boot
                    bootstrap_coefs[tau].append(beta_mmqr_boot)
            except:
                continue
        
        # Compute standard errors and confidence intervals
        for tau in self.quantiles:
            boot_array = np.array(bootstrap_coefs[tau])
            if len(boot_array) > 0:
                self.results[tau]['std_errors'] = np.std(boot_array, axis=0)
                self.results[tau]['ci_lower'] = np.percentile(boot_array, 2.5, axis=0)
                self.results[tau]['ci_upper'] = np.percentile(boot_array, 97.5, axis=0)
                self.results[tau]['pvalues'] = 2 * (1 - stats.norm.cdf(
                    np.abs(self.results[tau]['coefficients'] / self.results[tau]['std_errors'])
                ))
    
    def summary_table(self, var_names):
        """
        Create summary table of results
        """
        results_list = []
        
        for tau in self.quantiles:
            res = self.results[tau]
            coefs = res['coefficients']
            
            for i, var_name in enumerate(var_names):
                row = {
                    'Quantile': f"τ={tau:.2f}",
                    'Variable': var_name,
                    'Coefficient': coefs[i],
                }
                
                if 'std_errors' in res:
                    row['Std.Error'] = res['std_errors'][i]
                    row['z-value'] = coefs[i] / res['std_errors'][i]
                    row['P>|z|'] = res['pvalues'][i]
                    row['CI_Lower'] = res['ci_lower'][i]
                    row['CI_Upper'] = res['ci_upper'][i]
                
                results_list.append(row)
        
        return pd.DataFrame(results_list)


# ============================================
# Streamlit App
# ============================================

st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="mmqr_upload")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.session_state["uploaded_data"] = data
else:
    data = st.session_state.get("uploaded_data", None)

if data is not None:
    st.success("✅ Dataset loaded successfully.")
    
    with st.expander("📊 View Dataset Preview"):
        st.dataframe(data.head(10))
        st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    
    # Data quality check
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        st.warning(f"⚠️ Missing values detected: {missing_data[missing_data > 0].to_dict()}")
        if st.checkbox("Drop rows with missing values?"):
            data = data.dropna()
            st.success(f"✅ Cleaned dataset: {data.shape[0]} rows remaining")
    
    # Variable selection
    col1, col2 = st.columns(2)
    
    with col1:
        dependent_var = st.selectbox(
            "🎯 Select Dependent Variable (Y)", 
            options=data.columns,
            help="The outcome variable you want to model"
        )
    
    with col2:
        independent_vars = st.multiselect(
            "📊 Select Independent Variables (X)",
            options=[c for c in data.columns if c != dependent_var],
            help="Covariates/predictors"
        )
    
    # Quantiles selection
    st.sidebar.subheader("Quantile Specification")
    quantile_preset = st.sidebar.radio(
        "Choose quantile set:",
        ["Standard (5)", "Extended (9)", "Custom"]
    )
    
    if quantile_preset == "Standard (5)":
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    elif quantile_preset == "Extended (9)":
        quantiles = [0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95]
    else:
        quantile_input = st.sidebar.text_input(
            "Enter quantiles (comma-separated, e.g., 0.1,0.5,0.9):",
            "0.05,0.25,0.50,0.75,0.95"
        )
        try:
            quantiles = [float(q.strip()) for q in quantile_input.split(",")]
            quantiles = [q for q in quantiles if 0 < q < 1]
        except:
            st.sidebar.error("Invalid input. Using default quantiles.")
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    
    st.sidebar.write(f"**Selected quantiles:** {quantiles}")
    
    # Bootstrap options
    st.sidebar.subheader("Inference Options")
    use_bootstrap = st.sidebar.checkbox("Use Bootstrap Standard Errors", value=True)
    n_bootstrap = st.sidebar.slider("Bootstrap Replications", 50, 500, 200, 50) if use_bootstrap else 0
    
    # Run analysis
    if independent_vars:
        if st.button("🚀 Run MMQR Analysis", type="primary"):
            with st.spinner("Estimating MMQR model... This may take a moment."):
                try:
                    # Prepare data
                    y = data[dependent_var].values
                    X = data[independent_vars].values
                    
                    # Check for infinite values
                    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(X)):
                        st.error("❌ Data contains infinite or NaN values. Please clean your data.")
                    else:
                        # Fit MMQR model
                        mmqr = MMQRModel(y, X, quantiles=quantiles)
                        mmqr.fit(bootstrap_se=use_bootstrap, n_bootstrap=n_bootstrap)
                        
                        # Store in session state
                        st.session_state['mmqr_model'] = mmqr
                        st.session_state['var_names'] = ['Intercept'] + independent_vars
                        st.success("✅ MMQR estimation completed!")
                        
                except Exception as e:
                    st.error(f"❌ Error during estimation: {str(e)}")
                    st.exception(e)
    
    # Display results if model is fitted
    if 'mmqr_model' in st.session_state:
        mmqr = st.session_state['mmqr_model']
        var_names = st.session_state['var_names']
        
        st.header("📈 MMQR Results")
        
        # ========================
        # Summary Table
        # ========================
        st.subheader("Table 1: MMQR Coefficient Estimates")
        
        summary_df = mmqr.summary_table(var_names)
        
        # Format table
        def format_results(df):
            formatted = df.copy()
            if 'Coefficient' in formatted.columns:
                formatted['Coefficient'] = formatted['Coefficient'].map('{:.4f}'.format)
            if 'Std.Error' in formatted.columns:
                formatted['Std.Error'] = formatted['Std.Error'].map('{:.4f}'.format)
            if 'z-value' in formatted.columns:
                formatted['z-value'] = formatted['z-value'].map('{:.3f}'.format)
            if 'P>|z|' in formatted.columns:
                def format_pval(p):
                    if p < 0.001:
                        return '<0.001***'
                    elif p < 0.01:
                        return f'{p:.4f}**'
                    elif p < 0.05:
                        return f'{p:.4f}*'
                    elif p < 0.10:
                        return f'{p:.4f}†'
                    else:
                        return f'{p:.4f}'
                formatted['P>|z|'] = formatted['P>|z|'].map(format_pval)
            return formatted
        
        st.dataframe(format_results(summary_df), use_container_width=True, hide_index=True)
        st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10")
        
        # Download button
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download MMQR Results (CSV)",
            data=csv,
            file_name="MMQR_Results.csv",
            mime="text/csv"
        )
        
        # ========================
        # Visualization
        # ========================
        st.subheader("Figure 1: Unconditional Quantile Partial Effects (UQPE)")
        
        # Select variables to plot
        plot_vars = st.multiselect(
            "Select variables to visualize:",
            options=independent_vars,
            default=independent_vars[:min(3, len(independent_vars))]
        )
        
        if plot_vars:
            fig, axes = plt.subplots(1, len(plot_vars), figsize=(5*len(plot_vars), 5))
            if len(plot_vars) == 1:
                axes = [axes]
            
            palette = sns.color_palette("husl", len(plot_vars))
            
            for idx, var in enumerate(plot_vars):
                ax = axes[idx]
                var_idx = var_names.index(var)
                
                coefs = [mmqr.results[tau]['coefficients'][var_idx] for tau in quantiles]
                
                if use_bootstrap:
                    lower = [mmqr.results[tau]['ci_lower'][var_idx] for tau in quantiles]
                    upper = [mmqr.results[tau]['ci_upper'][var_idx] for tau in quantiles]
                    ax.fill_between(quantiles, lower, upper, alpha=0.2, color=palette[idx])
                
                ax.plot(quantiles, coefs, marker='o', linewidth=2, 
                       markersize=6, color=palette[idx], label=var)
                ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
                
                ax.set_xlabel("Quantile (τ)", fontsize=12, fontweight='bold')
                ax.set_ylabel("UQPE Coefficient", fontsize=12, fontweight='bold')
                ax.set_title(f"{var}", fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlim(min(quantiles)-0.05, max(quantiles)+0.05)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("""
            **Interpretation:** The plot shows how the marginal effect of each variable varies across 
            the unconditional distribution of the outcome. Unlike conditional quantile regression, 
            these coefficients represent population-level effects at different points of the outcome distribution.
            """)
        
        # ========================
        # Interpretation Guide
        # ========================
        st.subheader("📝 Interpretation Guidelines")
        
        with st.expander("How to interpret MMQR results"):
            st.markdown("""
            ### Unconditional Quantile Partial Effects (UQPE)
            
            **Key Differences from Conditional Quantile Regression:**
            - **MMQR coefficients** estimate the effect of X on the **unconditional quantiles** of Y
            - They answer: "How does X affect individuals at different points of the Y distribution?"
            - **CQR coefficients** estimate effects on conditional quantiles (given X values)
            
            **Interpretation Example:**
            - If β(τ=0.90) = 0.5 for education → income:
              - A one-unit increase in education increases income by 0.5 units **for individuals 
                at the 90th percentile of the income distribution**
            
            **Location vs. Scale Effects:**
            - If coefficients are similar across quantiles → **location shift** (parallel shift)
            - If coefficients vary substantially → **scale effect** (changes inequality)
            - Increasing coefficients (0.1→0.5→0.9) suggest the variable increases inequality
            - Decreasing coefficients suggest the variable reduces inequality
            
            **Statistical Significance:**
            - Check p-values and confidence intervals
            - Effects may be significant at some quantiles but not others
            """)
        
        # ========================
        # Diagnostic Summary
        # ========================
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            median_effect = mmqr.results[0.50]['coefficients'][1:] if 0.50 in quantiles else None
            if median_effect is not None:
                st.metric("Median Effects (Q50)", 
                         f"{np.mean(np.abs(median_effect)):.4f}",
                         help="Average absolute effect at median")
        
        with col2:
            effect_range = []
            for var_idx in range(1, len(var_names)):
                coefs_var = [mmqr.results[tau]['coefficients'][var_idx] for tau in quantiles]
                effect_range.append(max(coefs_var) - min(coefs_var))
            st.metric("Avg Effect Heterogeneity", 
                     f"{np.mean(effect_range):.4f}",
                     help="Average range of effects across quantiles")
        
        with col3:
            st.metric("Number of Quantiles", len(quantiles))
        
        # ========================
        # Export Options
        # ========================
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results export
            export_df = summary_df.copy()
            export_csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download Full Results Table",
                data=export_csv,
                file_name="MMQR_Full_Results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Plot data export
            plot_data = []
            for tau in quantiles:
                for i, var in enumerate(var_names):
                    plot_data.append({
                        'Quantile': tau,
                        'Variable': var,
                        'Coefficient': mmqr.results[tau]['coefficients'][i],
                        'CI_Lower': mmqr.results[tau].get('ci_lower', [None]*len(var_names))[i],
                        'CI_Upper': mmqr.results[tau].get('ci_upper', [None]*len(var_names))[i]
                    })
            plot_df = pd.DataFrame(plot_data)
            plot_csv = plot_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Download Plot Data",
                data=plot_csv,
                file_name="MMQR_Plot_Data.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Configure your variables and click 'Run MMQR Analysis' to see results.")

else:
    st.info("📁 Please upload a CSV dataset to begin MMQR analysis.")
    
    # Sample data option
    if st.checkbox("Use sample dataset for testing"):
        np.random.seed(42)
        n = 500
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        epsilon = np.random.normal(0, 1 + 0.5*X1, n)  # Heteroskedastic errors
        Y = 2 + 1.5*X1 + 0.8*X2 + epsilon
        
        sample_data = pd.DataFrame({
            'Y': Y,
            'X1': X1,
            'X2': X2
        })
        
        st.session_state["uploaded_data"] = sample_data
        st.success("✅ Sample dataset loaded! Refresh the page to see it.")
        st.dataframe(sample_data.head())

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
**About MMQR:**  
Method of Moments Quantile Regression estimates unconditional quantile treatment effects using a location-scale 
decomposition. This allows for policy-relevant inference about effects on the marginal distribution of outcomes.

**Citation:**  
Machado, J.A.F. and Silva, J.M.C.S. (2019). Quantiles via moments. *Journal of Econometrics*, 213(1), 145-173.
""")
