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
from scipy import stats
from scipy.stats import norm
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

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
    st.info("No file uploaded. Using sample dataset (sample_data.csv).")
    # Create sample data for demonstration
    np.random.seed(42)
    countries = ['Country_' + str(i) for i in range(1, 11)]
    years = list(range(2000, 2020))
    
    sample_data = []
    for country in countries:
        for year in years:
            gdp = np.random.normal(100, 20)
            tourism = gdp * 0.3 + np.random.normal(0, 5)
            sample_data.append({
                'Country': country,
                'Year': year,
                'GDP': gdp,
                'Tourism': tourism,
                'Investment': np.random.normal(50, 10),
                'Trade': np.random.normal(60, 15)
            })
    
    df = pd.DataFrame(sample_data)
    st.session_state["uploaded_data"] = df

# Display data overview
st.header("A. Data Overview")
st.dataframe(df.head())
st.write(f"Dataset shape: {df.shape}")

# Check for required columns
if 'Country' not in df.columns or 'Year' not in df.columns:
    st.error("❌ Required columns 'Country' and/or 'Year' not found in dataset!")
    st.stop()

# ============================================
# Section B: Correlation Analysis
# ============================================

st.header("B. Correlation Analysis")

# Detect numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.warning("No numeric variables found in your dataset.")
else:
    # Variable selection
    col1, col2 = st.columns(2)
    with col1:
        dep_var = st.selectbox("Select Dependent Variable", options=numeric_cols)
    with col2:
        indep_vars = st.multiselect(
            "Select Independent Variable(s)",
            options=[col for col in numeric_cols if col != dep_var],
            default=[col for col in numeric_cols if col != dep_var][:3] if len(numeric_cols) > 1 else []
        )

    # Color palette selector
    color_option = st.selectbox(
        "Select Heatmap Color Palette",
        options=["coolwarm", "viridis", "plasma", "magma", "cividis", "Blues", "Greens", "Reds"],
        index=0
    )

    if indep_vars:
        # Compute correlation matrix
        selected_vars = [dep_var] + indep_vars
        corr = df[selected_vars].corr()

        # Generate heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap=color_option, center=0, linewidths=0.5, fmt=".2f", ax=ax)
        plt.title(f"Correlation Heatmap")
        st.pyplot(fig)

        # Download button for heatmap
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            label="Download Heatmap Image",
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
                f"- The correlation between **{dep_var}** and **{var}** is "
                f"**{corr_value:.2f}**, indicating a **{strength} {direction} relationship**.\n"
            )

        st.markdown(interpretation_text)

        # Display correlation table
        st.subheader("Table 1: Correlation Matrix")
        st.dataframe(corr)

        st.info(
            "According to Evans (1996), correlation strengths are defined as: "
            "very weak (0.00–0.19), weak (0.20–0.39), moderate (0.40–0.59), "
            "strong (0.60–0.79), and very strong (0.80–1.00)."
        )
    else:
        st.warning("Please select at least one independent variable to display correlation.")

# ============================================
# Section C: Slope Homogeneity Test
# ============================================

st.header("C. Slope Homogeneity Test (Pesaran and Yamagata, 2008)")

try:
    # Check if we have the required variables
    if 'dep_var' not in locals() or 'indep_vars' not in locals() or not indep_vars:
        st.warning("Please complete the correlation analysis section first to select variables.")
    else:
        # Prepare data by country
        panel_results = []
        countries_with_data = []
        
        for country, subset in df.groupby("Country"):
            if len(subset) < 2:  # Need at least 2 observations per country
                continue
            if subset[dep_var].isnull().any() or subset[indep_vars].isnull().any().any():
                continue
                
            X = sm.add_constant(subset[indep_vars])
            y = subset[dep_var]
            
            # Check if we have enough data points
            if len(y) > len(indep_vars) + 1:  # More observations than parameters
                try:
                    model = sm.OLS(y, X).fit()
                    panel_results.append(model.params.values)
                    countries_with_data.append(country)
                except:
                    continue

        if len(panel_results) < 2:
            st.warning("Insufficient data for slope homogeneity test. Need at least 2 countries with sufficient observations.")
        else:
            betas = np.vstack(panel_results)
            mean_beta = np.mean(betas, axis=0)
            N, k = betas.shape

            # Compute test statistics
            S = np.sum((betas - mean_beta) ** 2, axis=0)
            delta = N * np.sum(S) / np.sum(mean_beta ** 2) if np.sum(mean_beta ** 2) != 0 else 0
            delta_adj = (N * delta - k) / np.sqrt(2 * k) if k > 0 else 0

            # Compute p-values
            p_delta = 2 * (1 - norm.cdf(abs(delta))) if delta != 0 else 1.0
            p_delta_adj = 2 * (1 - norm.cdf(abs(delta_adj))) if delta_adj != 0 else 1.0

            # Create results table
            results_df = pd.DataFrame({
                "Statistic": ["Δ", "Δ_adj"],
                "Value": [round(delta, 3), round(delta_adj, 3)],
                "p-value": [f"{p_delta:.3f}", f"{p_delta_adj:.3f}"]
            })

            st.write("**Slope Homogeneity Test Results**")
            st.dataframe(results_df, use_container_width=True)

            # Interpretation
            if p_delta_adj < 0.05:
                st.success("Reject the null hypothesis — slopes are *heterogeneous* across cross-sections.")
                st.markdown("**Interpretation:** The regression slopes are not the same for all cross-sections.")
            else:
                st.info("Fail to reject the null hypothesis — slopes are *homogeneous* across cross-sections.")
                st.markdown("**Interpretation:** The regression slopes are broadly similar across cross-sections.")

            st.caption(
                "Reference: Pesaran, M. H., & Yamagata, T. (2008). "
                "Testing slope homogeneity in large panels. Journal of Econometrics, 142(1), 50–93."
            )

except Exception as e:
    st.error(f"Error in slope homogeneity test: {str(e)}")

# ============================================
# Section D: MMQR Analysis
# ============================================

# ============================================
# Section D: MMQR Analysis (Revised)
# ============================================

st.header("D. Method of Moments Quantile Regression (MMQR)")

if 'dep_var' not in locals() or 'indep_vars' not in locals() or not indep_vars:
    st.warning("Please complete the correlation analysis section first to select variables.")
else:
    # MMQR configuration
    st.subheader("MMQR Configuration")
    col1, col2 = st.columns(2)
    with col1:
        quantiles = st.text_input("Quantiles (comma-separated)", "0.05,0.25,0.50,0.75,0.95")
        quantiles = [float(q.strip()) for q in quantiles.split(",")]
    with col2:
        bootstrap_ci = st.checkbox("Bootstrap Confidence Intervals", True)
        n_bootstrap = st.slider("Bootstrap Samples", 100, 1000, 200) if bootstrap_ci else 100

    # Revised MMQR Implementation with separate location and scale for each variable
    def revised_mmqr_estimation(data, y_var, x_vars, quantiles, bootstrap=True, n_boot=200):
        """
        Revised MMQR with location and scale effects for each variable
        """
        results = {}
        bootstrap_results = {q: [] for q in quantiles}
        
        # Prepare data
        X = data[x_vars]
        y = data[y_var]
        
        # Step 1: Location effects (mean regression) for each variable
        X_with_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_with_const).fit()
        location_effects = ols_model.params
        location_pvalues = ols_model.pvalues
        
        # Step 2: Scale effects (absolute residuals modeling) for each variable
        residuals = ols_model.resid
        abs_residuals = np.abs(residuals)
        scale_model = sm.OLS(abs_residuals, X_with_const).fit()
        scale_effects = scale_model.params
        scale_pvalues = scale_model.pvalues
        
        # Store location and scale results for each variable
        location_scale_results = {}
        for var in ['const'] + x_vars:
            location_scale_results[var] = {
                'location_coef': location_effects[var],
                'location_pvalue': location_pvalues[var],
                'scale_coef': scale_effects[var],
                'scale_pvalue': scale_pvalues[var]
            }
        
        # Step 3: Quantile regression for each quantile
        for q in quantiles:
            formula = f"{y_var} ~ {' + '.join(x_vars)}"
            q_model = quantreg(formula, data).fit(q=q, vcov='robust')
            
            coef_names = q_model.params.index.tolist()
            
            results[q] = {
                'coefficients': q_model.params,
                'pvalues': q_model.pvalues,
                'conf_int': q_model.conf_int(),
                'coef_names': coef_names,
                'quantile': q
            }
        
        # Bootstrap for confidence intervals
        if bootstrap:
            st.info("Running bootstrap inference... This may take a moment.")
            progress_bar = st.progress(0)
            
            for i in range(n_boot):
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
                    boot_coefs = pd.DataFrame(bootstrap_results[q])
                    results[q]['bootstrap_ci'] = {
                        'lower': boot_coefs.quantile(0.025),
                        'upper': boot_coefs.quantile(0.975)
                    }
        
        return results, location_scale_results

    # Run revised MMQR
    try:
        mmqr_results, location_scale_results = revised_mmqr_estimation(
            df, dep_var, indep_vars, quantiles, bootstrap_ci, n_bootstrap
        )
        
        # ========================
        # Table 1: Location and Scale Parameters for Each Variable
        # ========================
        st.subheader("Table 2: Location and Scale Parameters")
        
        location_scale_data = []
        for var in ['const'] + indep_vars:
            var_name = 'Intercept' if var == 'const' else var
            location_coef = location_scale_results[var]['location_coef']
            location_pval = location_scale_results[var]['location_pvalue']
            scale_coef = location_scale_results[var]['scale_coef']
            scale_pval = location_scale_results[var]['scale_pvalue']
            
            location_scale_data.append({
                'Variable': var_name,
                'Location Coef.': f"{location_coef:.3f}",
                'Location P-value': f"{location_pval:.3f}",
                'Location Sig.': '***' if location_pval < 0.01 else '**' if location_pval < 0.05 else '*' if location_pval < 0.1 else '',
                'Scale Coef.': f"{scale_coef:.3f}",
                'Scale P-value': f"{scale_pval:.3f}",
                'Scale Sig.': '***' if scale_pval < 0.01 else '**' if scale_pval < 0.05 else '*' if scale_pval < 0.1 else ''
            })
        
        location_scale_df = pd.DataFrame(location_scale_data)
        st.dataframe(location_scale_df, use_container_width=True)
        
        # ========================
        # Table 2: MMQR Coefficients with Probability Values (Academic Format)
        # ========================
        st.subheader("Table 3: MMQR Estimation Results")
        
        # Create comprehensive results table in academic format
        academic_results = []
        coef_names = mmqr_results[quantiles[0]]['coef_names']
        
        for var in coef_names:
            var_name = 'Intercept' if var == 'Intercept' else var
            row = {'Variable': var_name}
            
            for q in quantiles:
                coef = mmqr_results[q]['coefficients'][var]
                pval = mmqr_results[q]['pvalues'][var]
                
                # Academic format: coefficient with stars and p-value in parentheses
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                row[f'τ = {q}'] = f"{coef:.3f}{stars}\n({pval:.3f})"
            
            academic_results.append(row)
        
        academic_df = pd.DataFrame(academic_results)
        st.dataframe(academic_df, use_container_width=True)
        
        # ========================
        # Individual Coefficient Plots for Each Variable
        # ========================
        st.subheader("Figure 1: MMQR Coefficient Dynamics by Variable")
        
        # Create separate plots for each independent variable (excluding intercept)
        plot_vars = [var for var in coef_names if var != 'Intercept']
        
        for var in plot_vars:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            coefs = [mmqr_results[q]['coefficients'][var] for q in quantiles]
            pvals = [mmqr_results[q]['pvalues'][var] for q in quantiles]
            
            # Use bootstrap CI if available, else model CI
            if bootstrap_ci and 'bootstrap_ci' in mmqr_results[quantiles[0]]:
                lower = [mmqr_results[q]['bootstrap_ci']['lower'][var] for q in quantiles]
                upper = [mmqr_results[q]['bootstrap_ci']['upper'][var] for q in quantiles]
            else:
                lower = [mmqr_results[q]['conf_int'].loc[var, 0] for q in quantiles]
                upper = [mmqr_results[q]['conf_int'].loc[var, 1] for q in quantiles]
            
            # Plot main coefficient line
            line = ax.plot(quantiles, coefs, marker='o', linewidth=2.5, 
                          label=f'{var} Coefficient', color='#2E86AB')
            
            # Add confidence interval
            ax.fill_between(quantiles, lower, upper, alpha=0.3, color='#2E86AB')
            
            # Add significance markers
            significant_points = [(q, coef) for q, coef, pval in zip(quantiles, coefs, pvals) if pval < 0.1]
            if significant_points:
                sig_quantiles, sig_coefs = zip(*significant_points)
                ax.scatter(sig_quantiles, sig_coefs, color='red', s=100, zorder=5, 
                          label='Significant (p < 0.1)')
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            ax.set_xlabel("Quantiles (τ)", fontsize=12)
            ax.set_ylabel("Coefficient Estimate", fontsize=12)
            ax.set_title(f"MMQR Coefficient Dynamics: {var}", fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add p-value annotations
            for i, (q, pval) in enumerate(zip(quantiles, pvals)):
                sig_text = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                if sig_text:
                    ax.annotate(sig_text, (q, coefs[i]), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontweight='bold', color='red')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add some space between plots
            st.markdown("---")

        # ========================
        # Probability Summary Table
        # ========================
        st.subheader("Table 4: Probability Value Summary")
        
        prob_summary_data = []
        for var in plot_vars:
            pvals = [mmqr_results[q]['pvalues'][var] for q in quantiles]
            min_pval = min(pvals)
            max_pval = max(pvals)
            mean_pval = np.mean(pvals)
            significant_quantiles = [f'τ={q}' for q, p in zip(quantiles, pvals) if p < 0.1]
            
            prob_summary_data.append({
                'Variable': var,
                'Min P-value': f"{min_pval:.3f}",
                'Max P-value': f"{max_pval:.3f}",
                'Mean P-value': f"{mean_pval:.3f}",
                'Significant at': ', '.join(significant_quantiles) if significant_quantiles else 'None',
                'Overall Significance': 'Yes' if any(p < 0.1 for p in pvals) else 'No'
            })
        
        prob_summary_df = pd.DataFrame(prob_summary_data)
        st.dataframe(prob_summary_df, use_container_width=True)

        # ========================
        # Download Section with Multiple Formats
        # ========================
        st.subheader("Download Results")
        
        # Format 1: Complete results for Excel
        download_data = []
        
        # Add location and scale parameters
        for var in ['const'] + indep_vars:
            var_name = 'Intercept' if var == 'const' else var
            download_data.append({
                'Variable': var_name,
                'Type': 'Location',
                'Coefficient': round(location_scale_results[var]['location_coef'], 3),
                'P_Value': round(location_scale_results[var]['location_pvalue'], 3),
                'Quantile': 'All',
                'Significance': '***' if location_scale_results[var]['location_pvalue'] < 0.01 else 
                              '**' if location_scale_results[var]['location_pvalue'] < 0.05 else 
                              '*' if location_scale_results[var]['location_pvalue'] < 0.1 else ''
            })
            
            download_data.append({
                'Variable': var_name,
                'Type': 'Scale',
                'Coefficient': round(location_scale_results[var]['scale_coef'], 3),
                'P_Value': round(location_scale_results[var]['scale_pvalue'], 3),
                'Quantile': 'All',
                'Significance': '***' if location_scale_results[var]['scale_pvalue'] < 0.01 else 
                              '**' if location_scale_results[var]['scale_pvalue'] < 0.05 else 
                              '*' if location_scale_results[var]['scale_pvalue'] < 0.1 else ''
            })
        
        # Add quantile results
        for var in coef_names:
            var_name = 'Intercept' if var == 'Intercept' else var
            for q in quantiles:
                download_data.append({
                    'Variable': var_name,
                    'Type': f'Quantile_τ={q}',
                    'Coefficient': round(mmqr_results[q]['coefficients'][var], 3),
                    'P_Value': round(mmqr_results[q]['pvalues'][var], 3),
                    'Quantile': q,
                    'Significance': '***' if mmqr_results[q]['pvalues'][var] < 0.01 else 
                                  '**' if mmqr_results[q]['pvalues'][var] < 0.05 else 
                                  '*' if mmqr_results[q]['pvalues'][var] < 0.1 else ''
                })
        
        download_df = pd.DataFrame(download_data)
        
        # Format 2: Academic table format
        academic_download = []
        for var in coef_names:
            var_name = 'Intercept' if var == 'Intercept' else var
            row = {'Variable': var_name}
            for q in quantiles:
                coef = mmqr_results[q]['coefficients'][var]
                pval = mmqr_results[q]['pvalues'][var]
                row[f'Q{q}_Coef'] = round(coef, 3)
                row[f'Q{q}_Pval'] = round(pval, 3)
            academic_download.append(row)
        
        academic_download_df = pd.DataFrame(academic_download)
        
        # Create download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Complete results
            csv_complete = download_df.to_csv(index=False)
            st.download_button(
                "📥 Download Complete Results (CSV)",
                data=csv_complete,
                file_name="MMQR_Complete_Results.csv",
                mime="text/csv",
                help="Complete results including location, scale, and quantile parameters"
            )
        
        with col2:
            # Academic format
            csv_academic = academic_download_df.to_csv(index=False)
            st.download_button(
                "📊 Download Academic Format (CSV)",
                data=csv_academic,
                file_name="MMQR_Academic_Format.csv",
                mime="text/csv",
                help="Coefficients and p-values in academic table format"
            )
        
        # Copy-paste friendly table
        st.subheader("Copy-Paste Friendly Table")
        st.info("Select and copy the table below for easy pasting into Excel or Word:")
        
        # Create a simplified table for easy copying
        copy_table_data = []
        for var in coef_names:
            var_name = 'Intercept' if var == 'Intercept' else var
            row = {'Variable': var_name}
            for q in quantiles:
                coef = mmqr_results[q]['coefficients'][var]
                pval = mmqr_results[q]['pvalues'][var]
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                row[f'τ={q}'] = f"{coef:.3f}{stars}"
            copy_table_data.append(row)
        
        copy_table_df = pd.DataFrame(copy_table_data)
        st.dataframe(copy_table_df, use_container_width=True)
        
        # ========================
        # Interpretation Section
        # ========================
        st.subheader("Economic Interpretation")
        
        # Location and scale interpretation
        st.write("**Location and Scale Parameters:**")
        for var in ['const'] + indep_vars:
            var_name = 'Intercept' if var == 'const' else var
            loc_coef = location_scale_results[var]['location_coef']
            loc_pval = location_scale_results[var]['location_pvalue']
            scale_coef = location_scale_results[var]['scale_coef']
            scale_pval = location_scale_results[var]['scale_pvalue']
            
            st.write(f"**{var_name}**:")
            st.write(f"  - Location effect: {loc_coef:.3f} ({'significant' if loc_pval < 0.1 else 'not significant'})")
            st.write(f"  - Scale effect: {scale_coef:.3f} ({'significant' if scale_pval < 0.1 else 'not significant'})")
        
        # Quantile dynamics interpretation
        st.write("**Quantile-Specific Effects:**")
        for var in plot_vars:
            coefs = [mmqr_results[q]['coefficients'][var] for q in quantiles]
            pvals = [mmqr_results[q]['pvalues'][var] for q in quantiles]
            
            # Calculate trend
            trend = "increasing" if coefs[-1] > coefs[0] else "decreasing" if coefs[-1] < coefs[0] else "stable"
            
            # Significance pattern
            sig_count = sum(1 for p in pvals if p < 0.1)
            
            st.write(f"**{var}**:")
            st.write(f"  - Marginal effect trend: {trend}")
            st.write(f"  - Coefficient range: {min(coefs):.3f} to {max(coefs):.3f}")
            st.write(f"  - Significant at {sig_count} out of {len(quantiles)} quantiles")
            st.write(f"  - Overall significance: {'Yes' if sig_count > 0 else 'No'}")

    except Exception as e:
        st.error(f"MMQR estimation failed: {str(e)}")
        st.info("""
        Common issues to check:
        - Ensure variables have sufficient variation
        - Check for multicollinearity between independent variables
        - Verify there are no constant or near-constant variables
        - Ensure sufficient observations for the number of variables
        """)
# ============================================
# Section F: Panel Granger Causality Test
# ============================================

st.header("F. Panel Granger Causality Test")

try:
    # Check for required variables
    if 'GDP' not in df.columns or 'Tourism' not in df.columns:
        st.warning("GDP and/or Tourism columns not found. Using available numeric variables.")
        available_vars = [col for col in ['GDP', 'Tourism'] if col in df.columns]
        if len(available_vars) < 2:
            numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_vars) >= 2:
                var1, var2 = st.selectbox("Select first variable", numeric_vars), st.selectbox("Select second variable", 
                                                                                           [v for v in numeric_vars if v != var1])
            else:
                st.error("Need at least 2 numeric variables for Granger causality test")
                st.stop()
        else:
            var1, var2 = available_vars[0], available_vars[1]
    else:
        var1, var2 = 'GDP', 'Tourism'

    # Lag selection
    max_lag = st.slider("Select maximum lag for Granger test", 1, 5, 2)

    # Prepare panel data for Granger test
    def prepare_panel_data(df, var1, var2):
        """Prepare panel data for Granger causality testing"""
        countries = df['Country'].unique()
        results = {}
        
        for country in countries:
            country_data = df[df['Country'] == country].sort_values('Year')
            if len(country_data) > max_lag + 1:  # Need enough observations
                # Check for stationarity (basic check - constant mean and variance)
                data_subset = country_data[[var1, var2]].dropna()
                if len(data_subset) > max_lag + 1:
                    results[country] = data_subset
        return results

    panel_data = prepare_panel_data(df, var1, var2)
    
    if len(panel_data) < 2:
        st.warning(f"Insufficient data for Granger causality test. Need at least 2 countries with sufficient time series data.")
    else:
        st.info(f"Testing Granger causality between {var1} and {var2} using {len(panel_data)} countries")
        
        # Perform Granger tests
        direction1_pvals = []
        direction2_pvals = []
        
        for country, data in panel_data.items():
            try:
                # Test: var1 does not Granger cause var2
                test1 = grangercausalitytests(data[[var2, var1]], maxlag=max_lag, verbose=False)
                pval1 = test1[max_lag][0]['ssr_chi2test'][1]
                direction1_pvals.append(pval1)
                
                # Test: var2 does not Granger cause var1  
                test2 = grangercausalitytests(data[[var1, var2]], maxlag=max_lag, verbose=False)
                pval2 = test2[max_lag][0]['ssr_chi2test'][1]
                direction2_pvals.append(pval2)
            except:
                continue

        if direction1_pvals and direction2_pvals:
            # Calculate combined p-values using Fisher's method
            chi2_1 = -2 * np.sum(np.log(direction1_pvals))
            chi2_2 = -2 * np.sum(np.log(direction2_pvals))
            
            combined_p1 = 1 - stats.chi2.cdf(chi2_1, 2 * len(direction1_pvals))
            combined_p2 = 1 - stats.chi2.cdf(chi2_2, 2 * len(direction2_pvals))
            
            # Create results table
            granger_results = pd.DataFrame({
                "Null Hypothesis": [
                    f"{var1} does not Granger cause {var2}",
                    f"{var2} does not Granger cause {var1}"
                ],
                "Countries Tested": [len(direction1_pvals), len(direction2_pvals)],
                "Combined p-value": [f"{combined_p1:.4f}", f"{combined_p2:.4f}"],
                "Decision": [
                    "Reject H0" if combined_p1 < 0.05 else "Fail to reject H0",
                    "Reject H0" if combined_p2 < 0.05 else "Fail to reject H0"
                ]
            })
            
            st.dataframe(granger_results)
            
            # Interpretation
            st.subheader("Interpretation")
            if combined_p1 < 0.05 and combined_p2 < 0.05:
                st.success("**Bidirectional causality**: Both variables Granger-cause each other")
            elif combined_p1 < 0.05:
                st.success(f"**Unidirectional causality**: {var1} Granger-causes {var2}")
            elif combined_p2 < 0.05:
                st.success(f"**Unidirectional causality**: {var2} Granger-causes {var1}")
            else:
                st.info("**No Granger causality**: No causal relationship detected in either direction")
                
        else:
            st.error("Could not perform Granger causality tests on any country")

except Exception as e:
    st.error(f"Error in Granger causality test: {str(e)}")

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
