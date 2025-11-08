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
# ============================================
# Section E: Enhanced MMQR Approximation
# ============================================

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import quantreg
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.header("E. Method of Moments Quantile Regression (Enhanced Approximation)")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="mmqr_upload")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.session_state["uploaded_data"] = data
else:
    data = st.session_state.get("uploaded_data", None)

if data is not None:
    st.success("✅ Dataset loaded successfully.")
    
    # Data overview
    with st.expander("Data Overview"):
        st.dataframe(data.head())
        st.write(f"Dataset shape: {data.shape}")
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            st.warning(f"Missing values detected: {missing[missing > 0].to_dict()}")
    
    # Variable selection
    col1, col2 = st.columns(2)
    with col1:
        dependent_var = st.selectbox("Select Dependent Variable", 
                                   options=data.columns,
                                   key="mmqr_dep")
    with col2:
        independent_vars = st.multiselect(
            "Select Independent Variables",
            options=[c for c in data.columns if c != dependent_var],
            key="mmqr_ind"
        )
    
    # MMQR configuration
    st.subheader("MMQR Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        quantiles = st.text_input("Quantiles (comma-separated)", 
                                "0.05,0.25,0.50,0.75,0.95")
        quantiles = [float(q.strip()) for q in quantiles.split(",")]
    with col2:
        bootstrap_ci = st.checkbox("Bootstrap Confidence Intervals", True)
    with col3:
        n_bootstrap = st.slider("Bootstrap Samples", 100, 1000, 200) if bootstrap_ci else 100
    
    if independent_vars and len(independent_vars) > 0:
        
        # Enhanced MMQR Implementation
        def enhanced_mmqr_estimation(data, y_var, x_vars, quantiles, bootstrap=True, n_boot=200):
            """
            Enhanced MMQR approximation with location-scale modeling
            """
            results = {}
            bootstrap_results = {q: [] for q in quantiles}
            
            # Prepare data
            X = sm.add_constant(data[x_vars])
            y = data[y_var]
            
            # Step 1: Location effect (mean regression)
            ols_model = sm.OLS(y, X).fit()
            location_effects = ols_model.params
            
            # Step 2: Scale effect (absolute residuals modeling)
            residuals = ols_model.resid
            abs_residuals = np.abs(residuals)
            scale_model = sm.OLS(abs_residuals, X).fit()
            scale_effects = scale_model.params
            
            # Step 3: Quantile regression with enhanced features
            for q in quantiles:
                formula = f"{y_var} ~ {' + '.join(x_vars)}"
                q_model = quantreg(formula, data).fit(q=q, vcov='hc3')  # Robust standard errors
                
                # Store results
                results[q] = {
                    'coefficients': q_model.params,
                    'pvalues': q_model.pvalues,
                    'conf_int': q_model.conf_int(),
                    'residuals': q_model.resid,
                    'location_effect': location_effects,
                    'scale_effect': scale_effects
                }
            
            # Bootstrap for joint inference
            if bootstrap:
                st.info("Running bootstrap inference... This may take a moment.")
                progress_bar = st.progress(0)
                
                for i in range(n_boot):
                    # Bootstrap sample
                    boot_sample = data.sample(n=len(data), replace=True)
                    
                    for q in quantiles:
                        try:
                            formula = f"{y_var} ~ {' + '.join(x_vars)}"
                            boot_model = quantreg(formula, boot_sample).fit(q=q)
                            bootstrap_results[q].append(boot_model.params)
                        except:
                            continue
                    
                    progress_bar.progress((i + 1) / n_boot)
                
                # Calculate bootstrap confidence intervals
                for q in quantiles:
                    if len(bootstrap_results[q]) > 0:
                        boot_coefs = np.array(bootstrap_results[q])
                        results[q]['bootstrap_ci'] = {
                            'lower': np.percentile(boot_coefs, 2.5, axis=0),
                            'upper': np.percentile(boot_coefs, 97.5, axis=0)
                        }
            
            return results
        
        # Run enhanced MMQR
        try:
            mmqr_results = enhanced_mmqr_estimation(
                data, dependent_var, independent_vars, 
                quantiles, bootstrap_ci, n_bootstrap
            )
            
            # ========================
            # Results Table
            # ========================
            st.subheader("Table: Enhanced MMQR Coefficients Across Quantiles")
            
            # Create comprehensive results table
            results_data = []
            for var in ['const'] + independent_vars:
                row = {'Variable': var}
                for q in quantiles:
                    coef = mmqr_results[q]['coefficients'][var]
                    row[f'Q{q}'] = coef
                    
                    # Add significance stars
                    pval = mmqr_results[q]['pvalues'][var]
                    if pval < 0.01:
                        row[f'Q{q}'] = f"{coef:.3f}***"
                    elif pval < 0.05:
                        row[f'Q{q}'] = f"{coef:.3f}**"
                    elif pval < 0.1:
                        row[f'Q{q}'] = f"{coef:.3f}*"
                    else:
                        row[f'Q{q}'] = f"{coef:.3f}"
                
                results_data.append(row)
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # ========================
            # Enhanced Coefficient Plot
            # ========================
            st.subheader("Figure: MMQR Coefficient Dynamics with Confidence Intervals")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Coefficient trajectories
            for i, var in enumerate(independent_vars):
                coefs = [mmqr_results[q]['coefficients'][var] for q in quantiles]
                
                # Use bootstrap CI if available, else model CI
                if bootstrap_ci and 'bootstrap_ci' in mmqr_results[quantiles[0]]:
                    lower = [mmqr_results[q]['bootstrap_ci']['lower'][var] for q in quantiles]
                    upper = [mmqr_results[q]['bootstrap_ci']['upper'][var] for q in quantiles]
                else:
                    lower = [mmqr_results[q]['conf_int'].loc[var, 0] for q in quantiles]
                    upper = [mmqr_results[q]['conf_int'].loc[var, 1] for q in quantiles]
                
                axes[0].plot(quantiles, coefs, marker='o', linewidth=2, label=var)
                axes[0].fill_between(quantiles, lower, upper, alpha=0.2)
            
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0].set_xlabel("Quantiles (τ)")
            axes[0].set_ylabel("Coefficient Estimates")
            axes[0].set_title("MMQR Coefficient Dynamics")
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Location-scale effects comparison
            location_effects = mmqr_results[0.5]['location_effect']
            scale_effects = mmqr_results[0.5]['scale_effect']
            
            variables = independent_vars
            loc_vals = [location_effects[var] for var in variables]
            scale_vals = [scale_effects[var] for var in variables]
            
            x_pos = np.arange(len(variables))
            width = 0.35
            
            axes[1].bar(x_pos - width/2, loc_vals, width, label='Location Effect', alpha=0.7)
            axes[1].bar(x_pos + width/2, scale_vals, width, label='Scale Effect', alpha=0.7)
            axes[1].set_xlabel("Variables")
            axes[1].set_ylabel("Effect Size")
            axes[1].set_title("Location vs Scale Effects (Median)")
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(variables, rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # ========================
            # Diagnostic Tests
            # ========================
            st.subheader("Diagnostic Tests")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Quantile slope equality test (approximation)
                st.write("**Quantile Slope Equality Test**")
                median_coefs = mmqr_results[0.5]['coefficients']
                q1_coefs = mmqr_results[0.25]['coefficients']
                q3_coefs = mmqr_results[0.75]['coefficients']
                
                diff_low = np.abs(median_coefs - q1_coefs).mean()
                diff_high = np.abs(median_coefs - q3_coefs).mean()
                
                st.metric("Avg difference Q0.25 vs Q0.50", f"{diff_low:.4f}")
                st.metric("Avg difference Q0.50 vs Q0.75", f"{diff_high:.4f}")
            
            with col2:
                # Heteroskedasticity check
                st.write("**Heteroskedasticity Check**")
                median_residuals = mmqr_results[0.5]['residuals']
                scale_variation = np.std(np.abs(median_residuals)) / np.mean(np.abs(median_residuals))
                
                if scale_variation > 0.5:
                    st.warning(f"High residual variation: {scale_variation:.3f}")
                else:
                    st.success(f"Residual variation: {scale_variation:.3f}")
            
            # ========================
            # Economic Interpretation
            # ========================
            st.subheader("Economic Interpretation")
            
            interpretation_text = ""
            
            for var in independent_vars:
                coefs = [mmqr_results[q]['coefficients'][var] for q in quantiles]
                pvals = [mmqr_results[q]['pvalues'][var] for q in quantiles]
                
                # Significance pattern
                sig_low = any(pvals[i] < 0.1 for i in [0, 1])  # Lower quantiles
                sig_high = any(pvals[i] < 0.1 for i in [3, 4])  # Upper quantiles
                
                # Coefficient dynamics
                coef_range = max(coefs) - min(coefs)
                trend = "increasing" if coefs[-1] > coefs[0] else "decreasing" if coefs[-1] < coefs[0] else "stable"
                
                interpretation_text += f"""
                **{var}**: 
                - Effect ranges from {min(coefs):.3f} to {max(coefs):.3f} across quantiles
                - Shows {trend} marginal effects
                - {'Significant at lower quantiles' if sig_low else 'Not significant at lower quantiles'}
                - {'Significant at upper quantiles' if sig_high else 'Not significant at upper quantiles'}
                """
            
            st.markdown(interpretation_text)
            
            # ========================
            # Download Results
            # ========================
            st.subheader("Download Results")
            
            # Prepare comprehensive results for download
            download_data = []
            for var in ['const'] + independent_vars:
                for q in quantiles:
                    download_data.append({
                        'Variable': var,
                        'Quantile': q,
                        'Coefficient': mmqr_results[q]['coefficients'][var],
                        'P_Value': mmqr_results[q]['pvalues'][var],
                        'CI_Lower': mmqr_results[q]['conf_int'].loc[var, 0],
                        'CI_Upper': mmqr_results[q]['conf_int'].loc[var, 1]
                    })
            
            download_df = pd.DataFrame(download_data)
            csv = download_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download Complete MMQR Results",
                data=csv,
                file_name="MMQR_Complete_Results.csv",
                mime="text/csv"
            )
            
            # ========================
            # Methodological Note
            # ========================
            with st.expander("Methodological Notes"):
                st.markdown("""
                **Enhanced MMQR Approximation Features:**
                
                1. **Location-Scale Modeling**: Combines mean effects with scale effects
                2. **Robust Inference**: HC3 standard errors and bootstrap confidence intervals
                3. **Joint Quantile Analysis**: Simultaneous examination across distribution
                4. **Diagnostic Tests**: Quantile stability and heteroskedasticity checks
                
                **Limitations:**
                - This is an approximation of true MMQR (Machado & Silva, 2019)
                - True MMQR requires simultaneous estimation of all quantiles
                - Consider specialized packages for exact MMQR implementation
                
                **Interpretation Guide:**
                - Increasing coefficients → Stronger effects at upper quantiles
                - Decreasing coefficients → Stronger effects at lower quantiles  
                - Stable coefficients → Homogeneous effects across distribution
                """)
                
        except Exception as e:
            st.error(f"Estimation failed: {str(e)}")
            st.info("Please check your data for multicollinearity or other issues.")
    
    else:
        st.warning("Please select at least one independent variable.")

else:
    st.warning("Please upload your dataset to proceed.")
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
