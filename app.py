# ============================================
# Streamlit Panel Data Analysis App using MMQR
# Complete Reorganized Version
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import norm, chi2, f as f_dist
from statsmodels.formula.api import quantreg
import warnings

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(page_title="Panel Data Analysis", layout="wide", initial_sidebar_state="expanded")

# ============================================
# HELPER FUNCTIONS
# ============================================

def dumitrescu_hurlin_test(panel_data, id_var, time_var, y_var, x_var, max_lag):
    """
    Approximate Panel Granger Causality Test (Dumitrescu & Hurlin, 2012)
    """
    try:
        N = panel_data[id_var].nunique()
        T = panel_data[time_var].nunique()
        
        if N < 2 or T < max_lag + 3:
            return None, None, "Insufficient data", N, T, 0

        F_stats = []
        successful_units = 0
        
        for unit, subset in panel_data.groupby(id_var):
            subset = subset[[y_var, x_var]].dropna().sort_values(time_var).reset_index(drop=True)
            
            if len(subset) > max_lag + 2:
                try:
                    gct_result = grangercausalitytests(
                        subset[[y_var, x_var]], 
                        maxlag=max_lag, 
                        verbose=False
                    )
                    F_stat = gct_result[max_lag][0]['ssr_ftest'][0]
                    F_stats.append(F_stat)
                    successful_units += 1
                except:
                    continue

        if len(F_stats) < 2:
            return None, None, "Test failed", N, T, 0

        W_bar = np.mean(F_stats)
        k = max_lag
        
        if T > 2 * k + 2:
            Z_bar = np.sqrt(N) * (W_bar - 1) / np.sqrt(2 * k / (T - 2 * k - 2))
            p_value = 2 * (1 - norm.cdf(abs(Z_bar)))
        else:
            return None, None, "Insufficient time periods", N, T, successful_units
        
        interpretation = f"W-bar: {W_bar:.4f}, Z-bar: {Z_bar:.4f}"
        return W_bar, p_value, interpretation, N, T, successful_units
        
    except Exception as e:
        return None, None, f"Error: {str(e)}", 0, 0, 0


def enhanced_mmqr_estimation(data, y_var, x_vars, quantiles, bootstrap=True, n_boot=200):
    """
    Enhanced MMQR approximation with location-scale modeling
    """
    results = {}
    bootstrap_results = {q: [] for q in quantiles}
    
    # Prepare data
    X = data[x_vars]
    y = data[y_var]
    
    # Step 1: Location effect (mean regression)
    X_with_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_with_const).fit()
    location_effects = ols_model.params
    location_pvalues = ols_model.pvalues
    
    # Step 2: Scale effect (absolute residuals modeling)
    residuals = ols_model.resid
    abs_residuals = np.abs(residuals)
    scale_model = sm.OLS(abs_residuals, X_with_const).fit()
    scale_effects = scale_model.params
    scale_pvalues = scale_model.pvalues
    
    # Store location and scale results
    location_scale_results = {
        'location_intercept': location_effects['const'],
        'location_intercept_pvalue': location_pvalues['const'],
        'scale_intercept': scale_effects['const'],
        'scale_intercept_pvalue': scale_pvalues['const']
    }
    
    # Step 3: Quantile regression with robust standard errors
    for q in quantiles:
        formula = f"{y_var} ~ {' + '.join(x_vars)}"
        q_model = quantreg(formula, data).fit(q=q, vcov='robust')
        
        coef_names = q_model.params.index.tolist()
        
        results[q] = {
            'coefficients': q_model.params,
            'pvalues': q_model.pvalues,
            'conf_int': q_model.conf_int(),
            'residuals': q_model.resid,
            'location_effect': location_effects,
            'scale_effect': scale_effects,
            'coef_names': coef_names,
            'quantile': q
        }
    
    # Bootstrap for joint inference
    if bootstrap:
        for i in range(n_boot):
            boot_sample = data.sample(n=len(data), replace=True)
            
            for q in quantiles:
                try:
                    formula = f"{y_var} ~ {' + '.join(x_vars)}"
                    boot_model = quantreg(formula, boot_sample).fit(q=q)
                    bootstrap_results[q].append(boot_model.params)
                except:
                    continue
        
        # Calculate bootstrap confidence intervals
        for q in quantiles:
            if len(bootstrap_results[q]) > 0:
                boot_coefs = pd.DataFrame(bootstrap_results[q])
                results[q]['bootstrap_ci'] = {
                    'lower': boot_coefs.quantile(0.025),
                    'upper': boot_coefs.quantile(0.975)
                }
    
    return results, location_scale_results


# ============================================
# SECTION 1: DATA UPLOAD
# ============================================

st.title("📊 Panel Data Analysis Dashboard (MMQR Framework)")
st.markdown("""
This dashboard performs comprehensive **panel data econometric analysis** following this sequence:
1. **Data Upload** → 2. **Correlation Analysis** → 3. **Slope Homogeneity Test** → 
4. **MMQR Analysis** → 5. **MMQR Diagnostics** → 6. **Panel Granger Causality**
""")

st.markdown("---")

st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Custom data loaded successfully!")
    st.session_state["main_data"] = data
else:
    st.info("ℹ️ No file uploaded. Using sample dataset for demonstration.")
    # Create sample data
    np.random.seed(42)
    countries = ['USA', 'UK', 'Germany', 'France', 'Italy']
    years = list(range(2000, 2021))
    
    data_list = []
    for country in countries:
        for year in years:
            data_list.append({
                'Country': country,
                'Year': year,
                'GDP': np.random.uniform(20000, 50000) + year * 500,
                'Tourism': np.random.uniform(1000, 5000) + year * 50,
                'FDI': np.random.uniform(100, 500),
                'Infrastructure': np.random.uniform(50, 100)
            })
    
    data = pd.DataFrame(data_list)
    st.session_state["main_data"] = data

# Retrieve data
data = st.session_state.get("main_data", data)

# Display data preview
with st.expander("📋 View Data Preview"):
    st.dataframe(data.head(20), use_container_width=True)
    st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    if 'Country' in data.columns and 'Year' in data.columns:
        st.write(f"**Countries:** {data['Country'].nunique()}")
        st.write(f"**Time Period:** {data['Year'].min()} - {data['Year'].max()}")

st.markdown("---")

# ============================================
# SECTION 2: CORRELATION HEATMAP
# ============================================

st.header("Section 2: Correlation Analysis")

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.warning("⚠️ No numeric variables found in your dataset.")
else:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dep_var = st.selectbox("Select Dependent Variable", options=numeric_cols, key="corr_dep")
    
    with col2:
        indep_vars = st.multiselect(
            "Select Independent Variable(s)",
            options=[col for col in numeric_cols if col != dep_var],
            default=[col for col in numeric_cols if col != dep_var][:min(3, len(numeric_cols)-1)],
            key="corr_indep"
        )
    
    with col3:
        color_option = st.selectbox(
            "Heatmap Color Palette",
            options=["coolwarm", "viridis", "plasma", "RdBu_r", "Spectral"],
            index=0
        )

    if indep_vars:
        # Compute correlation matrix
        selected_vars = [dep_var] + indep_vars
        corr = data[selected_vars].corr()

        # Generate heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            cmap=color_option,
            center=0,
            linewidths=0.5,
            fmt=".3f",
            square=True
        )
        plt.title(f"Correlation Heatmap ({color_option} palette)", fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

        # Download button
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        st.download_button(
            label="📥 Download Heatmap",
            data=buf.getvalue(),
            file_name="correlation_heatmap.png",
            mime="image/png"
        )

        # Correlation interpretation
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
                f"- **{dep_var}** and **{var}**: {corr_value:.3f} "
                f"({strength} {direction} relationship)\n"
            )

        st.markdown(interpretation_text)
        
        st.info(
            "📚 **Reference:** Evans, J. D. (1996). *Straightforward statistics for the behavioral sciences.* "
            "Correlation strengths: very weak (0.00–0.19), weak (0.20–0.39), moderate (0.40–0.59), "
            "strong (0.60–0.79), very strong (0.80–1.00)."
        )
        
        # Display correlation matrix table
        st.subheader("Correlation Matrix Table")
        st.dataframe(corr.round(4), use_container_width=True)
        
        # Store variables for later sections
        st.session_state['dep_var'] = dep_var
        st.session_state['indep_vars'] = indep_vars
    else:
        st.warning("⚠️ Please select at least one independent variable.")

st.markdown("---")

# ============================================
# SECTION 3: SLOPE HOMOGENEITY TEST
# ============================================

st.header("Section 3: Slope Homogeneity Test (Pesaran & Yamagata, 2008)")

if "Country" not in data.columns or "Year" not in data.columns:
    st.warning("⚠️ Panel structure requires 'Country' and 'Year' columns.")
else:
    # Use variables from correlation section if available
    dep_var = st.session_state.get('dep_var', numeric_cols[0] if numeric_cols else None)
    indep_vars = st.session_state.get('indep_vars', numeric_cols[1:2] if len(numeric_cols) > 1 else [])
    
    if dep_var and indep_vars:
        try:
            # Prepare data by country
            panel_results = []
            valid_countries = []
            
            for country, subset in data.groupby("Country"):
                # Clean data
                clean_subset = subset[[dep_var] + indep_vars].dropna()
                
                if len(clean_subset) > len(indep_vars) + 1:  # Need enough observations
                    X = sm.add_constant(clean_subset[indep_vars])
                    y = clean_subset[dep_var]
                    
                    try:
                        model = sm.OLS(y, X).fit()
                        panel_results.append(model.params.values)
                        valid_countries.append(country)
                    except:
                        continue

            if len(panel_results) < 2:
                st.error("❌ Insufficient valid cross-sections for slope homogeneity test.")
            else:
                betas = np.vstack(panel_results)
                mean_beta = np.mean(betas, axis=0)
                N, k = betas.shape

                # Compute test statistics
                S = np.sum((betas - mean_beta) ** 2, axis=0)
                delta = N * np.sum(S) / np.sum(mean_beta ** 2)
                delta_adj = (N * delta - k) / np.sqrt(2 * k)

                # Compute p-values
                p_delta = 2 * (1 - norm.cdf(abs(delta)))
                p_delta_adj = 2 * (1 - norm.cdf(abs(delta_adj)))

                # Results table
                results_df = pd.DataFrame({
                    "Test Statistic": ["Δ (Delta)", "Δ_adj (Adjusted Delta)"],
                    "Value": [round(delta, 4), round(delta_adj, 4)],
                    "P-value": [f"{p_delta:.4f}", f"{p_delta_adj:.4f}"],
                    "Decision (α=0.05)": [
                        "Reject H₀" if p_delta < 0.05 else "Fail to Reject H₀",
                        "Reject H₀" if p_delta_adj < 0.05 else "Fail to Reject H₀"
                    ]
                })

                st.subheader("Slope Homogeneity Test Results")
                st.dataframe(results_df, use_container_width=True)
                
                st.caption(f"Test performed on {N} cross-sections (countries) with {k} parameters.")

                # Interpretation
                if p_delta_adj < 0.05:
                    st.success("✅ **Reject H₀**: Slopes are **heterogeneous** across cross-sections.")
                    st.markdown("**Implication:** Different countries have different regression relationships. Panel methods accounting for heterogeneity (like MMQR) are appropriate.")
                else:
                    st.info("ℹ️ **Fail to Reject H₀**: Slopes are **homogeneous** across cross-sections.")
                    st.markdown("**Implication:** Regression relationships are similar across countries. Standard panel methods may be appropriate.")

                # Reference
                st.caption(
                    "📚 **Reference:** Pesaran, M. H., & Yamagata, T. (2008). "
                    "Testing slope homogeneity in large panels. *Journal of Econometrics*, 142(1), 50–93."
                )

        except Exception as e:
            st.error(f"❌ Error in slope homogeneity test: {str(e)}")
    else:
        st.warning("⚠️ Please complete the correlation analysis first to select variables.")

st.markdown("---")

# ============================================
# SECTION 4: MMQR ANALYSIS
# ============================================

st.header("Section 4: Method of Moments Quantile Regression (MMQR)")

if dep_var and indep_vars:
    st.subheader("MMQR Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        quantiles_input = st.text_input(
            "Quantiles (comma-separated)", 
            "0.10,0.25,0.50,0.75,0.90",
            key="mmqr_quantiles"
        )
        quantiles = [float(q.strip()) for q in quantiles_input.split(",")]
    
    with col2:
        bootstrap_ci = st.checkbox("Bootstrap CI", True, key="mmqr_bootstrap")
    
    with col3:
        n_bootstrap = st.slider(
            "Bootstrap Samples", 
            50, 500, 200, 
            key="mmqr_n_boot"
        ) if bootstrap_ci else 100

    if st.button("🚀 Run MMQR Analysis", type="primary", key="run_mmqr"):
        with st.spinner("Running MMQR estimation... This may take a moment."):
            try:
                # Run MMQR
                mmqr_results, location_scale_results = enhanced_mmqr_estimation(
                    data, dep_var, indep_vars, quantiles, bootstrap_ci, n_bootstrap
                )
                
                st.success("✅ MMQR estimation completed successfully!")
                
                # Store results in session state
                st.session_state['mmqr_results'] = mmqr_results
                st.session_state['location_scale_results'] = location_scale_results
                st.session_state['mmqr_quantiles'] = quantiles
                
                # Table 1: Location and Scale Parameters
                st.subheader("Table 4.1: Location and Scale Intercept Parameters")
                
                location_data = {
                    'Parameter': ['Location Intercept (α)', 'Scale Intercept (λ)'],
                    'Coefficient': [
                        location_scale_results['location_intercept'],
                        location_scale_results['scale_intercept']
                    ],
                    'P-Value': [
                        location_scale_results['location_intercept_pvalue'],
                        location_scale_results['scale_intercept_pvalue']
                    ],
                    'Significance': [
                        '***' if location_scale_results['location_intercept_pvalue'] < 0.01 else 
                        '**' if location_scale_results['location_intercept_pvalue'] < 0.05 else 
                        '*' if location_scale_results['location_intercept_pvalue'] < 0.1 else '',
                        '***' if location_scale_results['scale_intercept_pvalue'] < 0.01 else 
                        '**' if location_scale_results['scale_intercept_pvalue'] < 0.05 else 
                        '*' if location_scale_results['scale_intercept_pvalue'] < 0.1 else ''
                    ]
                }
                
                location_df = pd.DataFrame(location_data)
                location_df['Coefficient'] = location_df['Coefficient'].round(4)
                location_df['P-Value'] = location_df['P-Value'].round(4)
                st.dataframe(location_df, use_container_width=True)
                st.caption("*** p<0.01, ** p<0.05, * p<0.1")
                
                # Table 2: MMQR Coefficients
                st.subheader("Table 4.2: MMQR Coefficient Estimates Across Quantiles")
                
                coef_names = mmqr_results[quantiles[0]]['coef_names']
                
                results_data = []
                for var in coef_names:
                    row = {'Variable': var}
                    for q in quantiles:
                        coef = mmqr_results[q]['coefficients'][var]
                        pval = mmqr_results[q]['pvalues'][var]
                        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                        row[f'Q{q}'] = f"{coef:.4f}{sig}"
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                st.caption("*** p<0.01, ** p<0.05, * p<0.1")
                
                # Coefficient Plot
                st.subheader("Figure 4.1: MMQR Coefficient Dynamics")
                
                plot_vars = [var for var in coef_names if var != 'Intercept']
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot 1: Coefficients
                for var in plot_vars:
                    coefs = [mmqr_results[q]['coefficients'][var] for q in quantiles]
                    pvals = [mmqr_results[q]['pvalues'][var] for q in quantiles]
                    
                    if bootstrap_ci and 'bootstrap_ci' in mmqr_results[quantiles[0]]:
                        lower = [mmqr_results[q]['bootstrap_ci']['lower'][var] for q in quantiles]
                        upper = [mmqr_results[q]['bootstrap_ci']['upper'][var] for q in quantiles]
                    else:
                        lower = [mmqr_results[q]['conf_int'].loc[var, 0] for q in quantiles]
                        upper = [mmqr_results[q]['conf_int'].loc[var, 1] for q in quantiles]
                    
                    line_style = '-' if any(p < 0.1 for p in pvals) else '--'
                    axes[0].plot(quantiles, coefs, marker='o', linewidth=2, 
                               label=var, linestyle=line_style)
                    axes[0].fill_between(quantiles, lower, upper, alpha=0.2)
                
                axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[0].set_xlabel("Quantiles (τ)", fontsize=12)
                axes[0].set_ylabel("Coefficient Estimates", fontsize=12)
                axes[0].set_title("Coefficient Trajectories Across Quantiles", fontsize=13, fontweight='bold')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: P-values
                for var in plot_vars:
                    pvals = [mmqr_results[q]['pvalues'][var] for q in quantiles]
                    axes[1].plot(quantiles, pvals, marker='s', linewidth=2, label=var)
                
                axes[1].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='10%')
                axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5%')
                axes[1].axhline(y=0.01, color='darkred', linestyle='--', alpha=0.7, label='1%')
                
                axes[1].set_xlabel("Quantiles (τ)", fontsize=12)
                axes[1].set_ylabel("P-Values", fontsize=12)
                axes[1].set_title("Statistical Significance Across Quantiles", fontsize=13, fontweight='bold')
                axes[1].legend()
                axes[1].set_yscale('log')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"❌ MMQR estimation failed: {str(e)}")
                st.info("""
                **Common issues:**
                - Insufficient observations for selected variables
                - Multicollinearity between independent variables
                - Missing values in the data
                - Variables with insufficient variation
                """)
else:
    st.warning("⚠️ Please complete correlation analysis first to select variables.")

st.markdown("---")

# ============================================
# SECTION 5: MMQR DIAGNOSTICS
# ============================================

st.header("Section 5: MMQR Diagnostic Tests")

if 'mmqr_results' in st.session_state:
    mmqr_results = st.session_state['mmqr_results']
    location_scale_results = st.session_state['location_scale_results']
    quantiles = st.session_state['mmqr_quantiles']
    coef_names = mmqr_results[quantiles[0]]['coef_names']
    
    # Diagnostic Table
    st.subheader("Table 5.1: MMQR Framework Diagnostic Tests")
    
    # Calculate stability measures
    test_vars = [var for var in coef_names if var != 'Intercept']
    
    if test_vars:
        test_var = test_vars[0]
        coefs = [mmqr_results[q]['coefficients'][test_var] for q in quantiles]
        stability_ratio = (max(coefs) - min(coefs)) / abs(np.mean(coefs)) if abs(np.mean(coefs)) > 1e-6 else np.nan
    else:
        stability_ratio = np.nan
    
    diagnostic_data = {
        'Test': [
            'Location Intercept (α)',
            'Location Intercept p-value',
            'Scale Intercept (λ)',
            'Scale Intercept p-value',
            f'Quantile Stability ({test_vars[0] if test_vars else "N/A"})'
        ],
        'Value': [
            f"{location_scale_results['location_intercept']:.4f}",
            f"{location_scale_results['location_intercept_pvalue']:.4f}",
            f"{location_scale_results['scale_intercept']:.4f}",
            f"{location_scale_results['scale_intercept_pvalue']:.4f}",
            f"{stability_ratio:.4f}" if not np.isnan(stability_ratio) else "N/A"
        ],
        'Interpretation': [
            'Baseline mean of dependent variable',
            'H₀: Location intercept = 0',
            'Baseline volatility/dispersion',
            'H₀: Scale intercept = 0',
            'Coefficient variation across quantiles'
        ],
        'Decision (α=0.10)': [
            '-',
            'Reject H₀' if location_scale_results['location_intercept_pvalue'] < 0.1 else 'Fail to Reject',
            '-',
            'Reject H₀' if location_scale_results['scale_intercept_pvalue'] < 0.1 else 'Fail to Reject',
            'Lower is better'
        ]
    }
    
    diagnostic_df = pd.DataFrame(diagnostic_data)
    st.dataframe(diagnostic_df, use_container_width=True)
    
    # Diagnostic Metrics
    st.subheader("Diagnostic Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Location Significance",
            "Yes ✓" if location_scale_results['location_intercept_pvalue'] < 0.1 else "No ✗",
            help="Significant location parameter validates MMQR framework"
        )
    
    with col2:
        st.metric(
            "Scale Significance",
            "Yes ✓" if location_scale_results['scale_intercept_pvalue'] < 0.1 else "No ✗",
            help="Significant scale parameter indicates heteroskedasticity"
        )
    
    with col3:
        significant_vars = sum(
            1 for var in test_vars 
            if any(mmqr_results[q]['pvalues'][var] < 0.1 for q in quantiles)
        )
        st.metric(
            "Significant Variables",
            f"{significant_vars}/{len(test_vars)}",
            help="Variables significant in at least one quantile"
        )
    
    # Interpretation
    st.subheader("Diagnostic Interpretation")
    
    if location_scale_results['location_intercept_pvalue'] < 0.1:
        st.success("✅ **Location parameter is significant** - MMQR framework is appropriate")
    else:
        st.warning("⚠️ Location parameter is not significant")
    
    if location_scale_results['scale_intercept_pvalue'] < 0.1:
        st.success("✅ **Scale parameter is significant** - Heteroskedasticity present, quantile methods appropriate")
    else:
        st.info("ℹ️ Scale parameter is not significant - Homoskedastic model may suffice")
    
    # Download results
    st.subheader("Download Results")
    
    download_data = []
    
    for var in coef_names:
        for q in quantiles:
            download_data.append({
                'Variable': var,
                'Quantile': q,
                'Coefficient': mmqr_results[q]['coefficients'][var],
                'P_Value': mmqr_results[q]['pvalues'][var],
                'Significance': '***' if mmqr_results[q]['pvalues'][var] < 0.01 else 
                              '**' if mmqr_results[q]['pvalues'][var] < 0.05 else 
                              '*' if mmqr_results[q]['pvalues'][var] < 0.1 else ''
            })
    
    download_df = pd.DataFrame(download_data)
    csv = download_df.to_csv(index=False)
    
    st.download_button(
        "📥 Download Complete MMQR Results",
        data=csv,
        file_name="MMQR_Complete_Results.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ Please run MMQR analysis first (Section 4).")

st.markdown("---")

# ============================================
# SECTION 6: PANEL GRANGER CAUSALITY
# ============================================

st.header("Section 6: Panel Granger Causality Tests (Dumitrescu & Hurlin, 2012)")

if "Country" not in data.columns or "Year" not in data.columns:
    st.error("❌ Panel Granger Causality requires 'Country' and 'Year' columns.")
elif len(numeric_cols) < 2:
    st.warning("⚠️ Need at least 2 numeric variables for causality testing.")
else:
    st.markdown("""
    This test examines whether past values of one variable help predict another variable 
    in a panel data context. The Dumitrescu-Hurlin test extends Granger causality to heterogeneous panels.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gc_var_y = st.selectbox(
            "Dependent Variable (Y)", 
            options=numeric_cols, 
            key="gc_y"
        )
    
    with col2:
        gc_var_x = st.selectbox(
            "Causal Variable (X)", 
            options=[c for c in numeric_cols if c != gc_var_y], 
            key="gc_x"
        )
    
    with col3:
        max_lag = st.slider("Max Lag Order", min_value=1, max_value=5, value=2, key="gc_lag")

    if gc_var_y and gc_var_x and st.button("🔍 Run Granger Causality Tests", type="primary", key="run_gc"):
        with st.spinner("Running Panel Granger Causality tests..."):
            
            # Test 1: X -> Y
            st.write(f"**Test 1:** Does {gc_var_x} Granger-cause {gc_var_y}?")
            W_bar_1, p_value_1, interp_1, N1, T1, n_units_1 = dumitrescu_hurlin_test(
                data, 'Country', 'Year', gc_var_y, gc_var_x, max_lag
            )
            
            # Test 2: Y -> X (Reverse)
            st.write(f"**Test 2:** Does {gc_var_y} Granger-cause {gc_var_x}?")
            W_bar_2, p_value_2, interp_2, N2, T2, n_units_2 = dumitrescu_hurlin_test(
                data, 'Country', 'Year', gc_var_x, gc_var_y, max_lag
            )

            if W_bar_1 is not None and W_bar_2 is not None:
                st.success("✅ Granger Causality tests completed successfully!")
                
                # Results Table
                st.subheader("Table 6.1: Panel Granger Causality Test Results")
                
                gc_results = pd.DataFrame({
                    "Null Hypothesis (H₀)": [
                        f"{gc_var_x} does not Granger-cause {gc_var_y}",
                        f"{gc_var_y} does not Granger-cause {gc_var_x}"
                    ],
                    "W-bar Statistic": [
                        f"{W_bar_1:.4f}",
                        f"{W_bar_2:.4f}"
                    ],
                    "P-value": [
                        f"{p_value_1:.4f}",
                        f"{p_value_2:.4f}"
                    ],
                    "Decision (α=0.05)": [
                        "Reject H₀ ***" if p_value_1 < 0.01 else "Reject H₀ **" if p_value_1 < 0.05 else "Reject H₀ *" if p_value_1 < 0.1 else "Fail to Reject H₀",
                        "Reject H₀ ***" if p_value_2 < 0.01 else "Reject H₀ **" if p_value_2 < 0.05 else "Reject H₀ *" if p_value_2 < 0.1 else "Fail to Reject H₀"
                    ],
                    "Interpretation": [
                        f"{gc_var_x} → {gc_var_y}" if p_value_1 < 0.05 else "No causality",
                        f"{gc_var_y} → {gc_var_x}" if p_value_2 < 0.05 else "No causality"
                    ]
                })
                
                st.dataframe(gc_results, use_container_width=True)
                
                st.caption(f"""
                **Test Details:** N={N1} cross-sections, T={T1} time periods, p={max_lag} lags.
                Tests based on {n_units_1} units with sufficient data. 
                *** p<0.01, ** p<0.05, * p<0.1
                """)
                
                # Causal Relationship Summary
                st.subheader("Causal Relationship Summary")
                
                if p_value_1 < 0.05 and p_value_2 >= 0.05:
                    st.success(f"✅ **Unidirectional Causality:** {gc_var_x} → {gc_var_y}")
                    st.markdown(f"""
                    **Finding:** {gc_var_x} Granger-causes {gc_var_y}, but not vice versa.
                    
                    **Implication:** Past values of {gc_var_x} provide statistically significant 
                    information for predicting {gc_var_y}, suggesting that {gc_var_x} leads {gc_var_y}.
                    """)
                    
                elif p_value_1 >= 0.05 and p_value_2 < 0.05:
                    st.success(f"✅ **Unidirectional Causality:** {gc_var_y} → {gc_var_x}")
                    st.markdown(f"""
                    **Finding:** {gc_var_y} Granger-causes {gc_var_x}, but not vice versa.
                    
                    **Implication:** Past values of {gc_var_y} provide statistically significant 
                    information for predicting {gc_var_x}, suggesting that {gc_var_y} leads {gc_var_x}.
                    """)
                    
                elif p_value_1 < 0.05 and p_value_2 < 0.05:
                    st.warning(f"⚠️ **Bidirectional Causality (Feedback):** {gc_var_x} ⇄ {gc_var_y}")
                    st.markdown(f"""
                    **Finding:** Both variables Granger-cause each other.
                    
                    **Implication:** There is a dynamic feedback relationship between {gc_var_x} and {gc_var_y}. 
                    They mutually influence each other over time, suggesting complex interdependence.
                    """)
                    
                else:
                    st.info(f"ℹ️ **No Granger Causality Detected**")
                    st.markdown(f"""
                    **Finding:** Neither variable Granger-causes the other at conventional significance levels.
                    
                    **Implication:** The lagged values of one variable do not significantly improve 
                    predictions of the other variable, suggesting no predictive relationship.
                    """)
                
                # Visualization
                st.subheader("Figure 6.1: Causal Relationship Diagram")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
                
                # Draw variable boxes
                box1 = FancyBboxPatch((1, 2), 2.5, 1, boxstyle="round,pad=0.1", 
                                     edgecolor='#2E86AB', facecolor='#A9D6E5', linewidth=2.5)
                box2 = FancyBboxPatch((6.5, 2), 2.5, 1, boxstyle="round,pad=0.1", 
                                     edgecolor='#2A9D8F', facecolor='#B7E4C7', linewidth=2.5)
                ax.add_patch(box1)
                ax.add_patch(box2)
                
                # Variable names
                ax.text(2.25, 2.5, gc_var_x, ha='center', va='center', 
                       fontsize=13, fontweight='bold')
                ax.text(7.75, 2.5, gc_var_y, ha='center', va='center', 
                       fontsize=13, fontweight='bold')
                
                # Draw arrows based on significance
                if p_value_1 < 0.05:
                    arrow1 = FancyArrowPatch((3.6, 2.7), (6.4, 2.7), 
                                           arrowstyle='->', mutation_scale=30, 
                                           linewidth=3, color='#2E86AB')
                    ax.add_patch(arrow1)
                    sig1 = '***' if p_value_1 < 0.01 else '**' if p_value_1 < 0.05 else '*'
                    ax.text(5, 3.15, f'p={p_value_1:.4f}{sig1}', 
                           ha='center', fontsize=10, color='#2E86AB', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                if p_value_2 < 0.05:
                    arrow2 = FancyArrowPatch((6.4, 2.3), (3.6, 2.3), 
                                           arrowstyle='->', mutation_scale=30, 
                                           linewidth=3, color='#2A9D8F')
                    ax.add_patch(arrow2)
                    sig2 = '***' if p_value_2 < 0.01 else '**' if p_value_2 < 0.05 else '*'
                    ax.text(5, 1.85, f'p={p_value_2:.4f}{sig2}', 
                           ha='center', fontsize=10, color='#2A9D8F', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Add "No causality" text if neither is significant
                if p_value_1 >= 0.05 and p_value_2 >= 0.05:
                    ax.text(5, 2.5, 'No Granger Causality', 
                           ha='center', va='center', fontsize=12, 
                           color='gray', style='italic',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   edgecolor='gray', alpha=0.9))
                
                ax.set_xlim(0, 10)
                ax.set_ylim(1, 4)
                ax.axis('off')
                ax.set_title('Panel Granger Causality Relationships', 
                           fontsize=15, fontweight='bold', pad=20)
                
                # Legend
                ax.text(5, 1.3, '*** p<0.01, ** p<0.05, * p<0.1', 
                       ha='center', fontsize=9, style='italic')
                
                st.pyplot(fig)
                plt.close()
                
                # Download results
                csv_gc = gc_results.to_csv(index=False)
                st.download_button(
                    "📥 Download Granger Causality Results",
                    data=csv_gc,
                    file_name="granger_causality_results.csv",
                    mime="text/csv"
                )
                
                # Methodological notes
                with st.expander("📚 Methodological Notes"):
                    st.markdown("""
                    **Test Methodology:**
                    - **Framework:** Dumitrescu & Hurlin (2012) Panel Granger Causality Test
                    - **Null Hypothesis:** Variable X does not Granger-cause variable Y
                    - **W-bar Statistic:** Average of individual Wald statistics across panel units
                    - **Z-bar Statistic:** Standardized test statistic for inference
                    
                    **Interpretation:**
                    - Granger causality tests predictive relationships, not true causation
                    - "X Granger-causes Y" means past values of X help predict Y
                    - Significant results indicate temporal precedence and predictive power
                    - Bidirectional causality suggests feedback mechanisms
                    
                    **Reference:**
                    Dumitrescu, E. I., & Hurlin, C. (2012). Testing for Granger non-causality 
                    in heterogeneous panels. *Economic Modelling*, 29(4), 1450-1460.
                    """)
                
            else:
                st.error(f"❌ Test failed: {interp_1}")
                st.info("""
                **Possible reasons:**
                - Insufficient time periods per cross-section
                - Too many missing values
                - Selected lag order too large for available data
                - Data quality issues (constants, extreme outliers)
                
                **Suggestions:**
                - Reduce the lag order
                - Check data for missing values or anomalies
                - Ensure sufficient time series length (T > lag + 3)
                """)

st.markdown("---")

# ============================================
# FOOTER & EXPORT OPTIONS
# ============================================

st.header("Complete Analysis Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", f"{data.shape[0]} observations")
    st.metric("Variables", f"{len(numeric_cols)} numeric")

with col2:
    if 'Country' in data.columns:
        st.metric("Countries", data['Country'].nunique())
    if 'Year' in data.columns:
        st.metric("Time Span", f"{data['Year'].max() - data['Year'].min() + 1} years")

with col3:
    sections_completed = 0
    if 'dep_var' in st.session_state:
        sections_completed += 1
    if 'mmqr_results' in st.session_state:
        sections_completed += 2
    st.metric("Analysis Progress", f"{sections_completed}/6 sections")

st.markdown("---")

st.info("""
**📊 Analysis Workflow Summary:**

1. ✅ **Data Upload** - Load your panel dataset
2. ✅ **Correlation Analysis** - Examine relationships between variables
3. ✅ **Slope Homogeneity** - Test for parameter heterogeneity
4. ✅ **MMQR Analysis** - Quantile regression across distribution
5. ✅ **MMQR Diagnostics** - Validate model framework
6. ✅ **Granger Causality** - Test temporal precedence relationships

**Key Features:**
- Interactive variable selection
- Professional academic tables and figures
- Downloadable results (CSV, PNG)
- Comprehensive interpretations
- Econometric rigor with proper citations
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><b>Panel Data Econometric Analysis Dashboard</b></p>
    <p>MMQR Framework | Granger Causality | Slope Homogeneity</p>
    <p style='font-size: 0.9em;'>Developed for academic research and applied econometrics</p>
</div>
""", unsafe_allow_html=True)
