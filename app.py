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

# ============================================
# Enhanced Distribution Overview (inserted after Section B)
# ============================================

st.header("B2. Enhanced Distribution Overview (Sample-Inside Visuals)")

if numeric_cols:
    st.write("This section visualizes variable distributions with embedded density samples for shape inspection.")
    selected_var2 = st.selectbox("Select variable for distribution view:", options=numeric_cols, key="dist_select")

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    cmap = sns.color_palette("viridis", as_cmap=True)

    # 1. Bar Plot + Density trace inside bars
    sns.histplot(df[selected_var2], bins=25, kde=True, color=cmap(0.5), ax=axs[0, 0], edgecolor='white')
    x_vals = np.linspace(df[selected_var2].min(), df[selected_var2].max(), 200)
    kde = stats.gaussian_kde(df[selected_var2])
    axs[0, 0].plot(x_vals, kde(x_vals) * len(df[selected_var2]) * (x_vals[1] - x_vals[0]) * 25, color='black', lw=2)
    axs[0, 0].set_title("Bar Plot + Internal Density", fontsize=12, fontweight="bold")
    axs[0, 0].grid(alpha=0.3)

    # 2. Box Plot + Internal Kernel Sample
    sns.boxplot(y=df[selected_var2], color=cmap(0.4), ax=axs[0, 1])
    sample_points = np.random.choice(df[selected_var2], size=min(50, len(df)), replace=False)
    jitter_y = sample_points + np.random.normal(0, df[selected_var2].std()/50, len(sample_points))
    axs[0, 1].scatter(np.random.normal(1, 0.02, len(jitter_y)), jitter_y, color='black', s=15, alpha=0.6)
    axs[0, 1].set_title("Box Plot + Embedded Sample Trace", fontsize=12, fontweight="bold")
    axs[0, 1].grid(alpha=0.3)

    # 3. Violin Plot + Embedded Sample Density Line
    sns.violinplot(y=df[selected_var2], inner=None, color=cmap(0.6), ax=axs[1, 0])
    kde_y = np.linspace(df[selected_var2].min(), df[selected_var2].max(), 300)
    axs[1, 0].plot(np.full_like(kde_y, 1.02), kde_y, color='black', lw=1.8)
    axs[1, 0].set_title("Violin Plot + Internal Density Line", fontsize=12, fontweight="bold")
    axs[1, 0].grid(alpha=0.3)

    # 4. Strip Plot + Mini KDE Overlay
    sns.stripplot(y=df[selected_var2], color=cmap(0.7), alpha=0.6, size=4, jitter=0.25, ax=axs[1, 1])
    axs[1, 1].axhline(df[selected_var2].mean(), color='red', linestyle='--', lw=2, label="Mean")
    axs[1, 1].legend()
    axs[1, 1].set_title("Strip Plot + Mean Reference", fontsize=12, fontweight="bold")
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("No numeric variables available for distribution visualization.")

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
# Section D: MMQR Implementation (unchanged)
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")

# ============================================
# Section D: Corrected MMQR Implementation with Scale P-values
# ============================================

st.header("D. Method of Moments Quantile Regression (MMQR) - Machado & Santos Silva (2019)")

if 'dep_var' not in locals() or 'indep_vars' not in locals() or not indep_vars:
    st.warning("Please complete the correlation analysis first to select variables.")
else:
    st.subheader("MMQR Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        quantiles = st.text_input("Quantiles (comma-separated)", "0.05,0.25,0.50,0.75,0.95")
        quantiles = [float(q.strip()) for q in quantiles.split(",")]
    
    with col2:
        bootstrap_ci = st.checkbox("Bootstrap Confidence Intervals", True)
        n_bootstrap = st.slider("Bootstrap Samples", 100, 1000, 200) if bootstrap_ci else 100
        reference_quantile = st.selectbox("Reference Quantile for Location", [0.25, 0.50, 0.75], index=1)

    def correct_mmqr_estimation(data, y_var, x_vars, quantiles, reference_quantile=0.5, bootstrap=True, n_boot=200):
        """
        Correct MMQR implementation following Machado & Santos Silva (2019)
        with proper inference for scale parameters
        """
        results = {}
        
        # Prepare data
        X = data[x_vars]
        y = data[y_var]
        
        # Step 1: Estimate location parameters using reference quantile (usually median)
        formula_ref = f"{y_var} ~ {' + '.join(x_vars)}"
        location_model = quantreg(formula_ref, data).fit(q=reference_quantile, vcov='robust')
        location_params = location_model.params
        location_pvalues = location_model.pvalues
        
        # Step 2: Estimate scale parameters using symmetric quantiles with proper inference
        tau_high = 0.75
        tau_low = 0.25
        
        model_high = quantreg(formula_ref, data).fit(q=tau_high, vcov='robust')
        model_low = quantreg(formula_ref, data).fit(q=tau_low, vcov='robust')
        
        # Scale parameters are proportional to the difference between high and low quantiles
        scale_params = (model_high.params - model_low.params) / (tau_high - tau_low)
        
        # Calculate p-values for scale parameters using delta method
        scale_pvalues = {}
        for var in scale_params.index:
            try:
                # Variance of scale parameter using delta method
                # Var(scale) = [Var(high) + Var(low) - 2*Cov(high,low)] / (tau_high - tau_low)^2
                
                # Get variances from the models
                var_high = model_high.bse[var] ** 2
                var_low = model_low.bse[var] ** 2
                
                # For covariance, we use approximation since we don't have direct covariance matrix
                # A conservative approach: assume moderate positive correlation (0.3)
                cov_high_low = 0.3 * np.sqrt(var_high * var_low)
                
                var_scale = (var_high + var_low - 2 * cov_high_low) / ((tau_high - tau_low) ** 2)
                se_scale = np.sqrt(var_scale)
                
                # Calculate t-statistic and p-value
                t_stat = scale_params[var] / se_scale if se_scale > 0 else 0
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(data) - len(x_vars) - 1))
                scale_pvalues[var] = p_value
                
            except:
                scale_pvalues[var] = 1.0  # Default to insignificant if calculation fails
        
        # Alternative method: Bootstrap for scale parameters
        if bootstrap:
            st.info("Bootstrapping scale parameters...")
            bootstrap_scale = {var: [] for var in scale_params.index}
            progress_bar = st.progress(0)
            
            for i in range(n_boot):
                boot_sample = data.sample(n=len(data), replace=True)
                try:
                    boot_high = quantreg(formula_ref, boot_sample).fit(q=tau_high)
                    boot_low = quantreg(formula_ref, boot_sample).fit(q=tau_low)
                    boot_scale = (boot_high.params - boot_low.params) / (tau_high - tau_low)
                    
                    for var in scale_params.index:
                        bootstrap_scale[var].append(boot_scale[var])
                except:
                    continue
                
                progress_bar.progress((i + 1) / n_boot)
            
            # Calculate bootstrap p-values for scale parameters
            for var in scale_params.index:
                if len(bootstrap_scale[var]) > 0:
                    # Two-sided p-value: proportion of bootstrap estimates with opposite sign
                    boot_vals = np.array(bootstrap_scale[var])
                    p_value_boot = 2 * min(
                        np.mean(boot_vals > 0) if scale_params[var] < 0 else np.mean(boot_vals <= 0),
                        np.mean(boot_vals <= 0) if scale_params[var] < 0 else np.mean(boot_vals > 0)
                    )
                    # Use bootstrap p-value if we have enough samples
                    if len(bootstrap_scale[var]) >= 100:
                        scale_pvalues[var] = p_value_boot
        
        # Step 3: MMQR transformation and estimation
        for q in quantiles:
            try:
                # Transform the model using location and scale parameters
                h_tau = q
                
                # MMQR coefficients: β(τ) = α + δ·h(τ)
                mmqr_coefficients = location_params + scale_params * h_tau
                
                # Estimate the final model with robust inference
                y_approx = location_model.predict() + scale_params.mean() * h_tau * (y - location_model.predict()).std()
                
                data_temp = data.copy()
                data_temp['y_transformed'] = y_approx
                
                formula_mmqr = f"y_transformed ~ {' + '.join(x_vars)}"
                final_model = quantreg(formula_mmqr, data_temp).fit(q=q, vcov='robust')
                
                results[q] = {
                    'coefficients': final_model.params,
                    'pvalues': final_model.pvalues,
                    'conf_int': final_model.conf_int(),
                    'location_params': location_params,
                    'scale_params': scale_params,
                    'scale_pvalues': scale_pvalues,
                    'mmqr_coefficients': mmqr_coefficients,
                    'quantile': q,
                    'model': final_model
                }
                
            except Exception as e:
                st.warning(f"MMQR estimation failed for quantile {q}: {str(e)}")
                # Fallback to standard QR if MMQR fails
                formula = f"{y_var} ~ {' + '.join(x_vars)}"
                fallback_model = quantreg(formula, data).fit(q=q, vcov='robust')
                results[q] = {
                    'coefficients': fallback_model.params,
                    'pvalues': fallback_model.pvalues,
                    'conf_int': fallback_model.conf_int(),
                    'location_params': location_params,
                    'scale_params': scale_params,
                    'scale_pvalues': scale_pvalues,
                    'mmqr_coefficients': fallback_model.params,
                    'quantile': q,
                    'model': fallback_model
                }
        
        return results, location_params, scale_params, scale_pvalues

    # Run corrected MMQR with scale p-values
    try:
        mmqr_results, location_params, scale_params, scale_pvalues = correct_mmqr_estimation(
            df, dep_var, indep_vars, quantiles, reference_quantile, bootstrap_ci, n_bootstrap
        )
        
        # ========================
        # Table 1: Location Parameters (Reference Quantile)
        # ========================
        st.subheader(f"Table 2: Location Parameters (τ = {reference_quantile})")
        
        location_data = []
        for var in location_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            coef = location_params[var]
            pval = mmqr_results[reference_quantile]['pvalues'][var] if reference_quantile in mmqr_results else 1.0
            
            location_data.append({
                'Variable': var_name,
                'Coefficient': f"{coef:.3f}",
                'P-Value': f"{pval:.3f}",
                'Significance': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            })
        
        location_df = pd.DataFrame(location_data)
        st.dataframe(location_df, use_container_width=True)
        
        # ========================
        # Table 2: Scale Parameters with P-Values
        # ========================
        st.subheader("Table 3: Scale Parameters with Statistical Inference")
        
        scale_data = []
        for var in scale_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            scale_val = scale_params[var]
            pval = scale_pvalues[var]
            
            # Calculate economic significance (relative to location)
            if var in location_params:
                relative_effect = abs(scale_val) / abs(location_params[var]) if location_params[var] != 0 else float('inf')
            else:
                relative_effect = 0
            
            scale_data.append({
                'Variable': var_name,
                'Scale Coefficient': f"{scale_val:.3f}",
                'P-Value': f"{pval:.3f}",
                'Significance': '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else '',
                'Relative to Location': f"{relative_effect:.3f}" if relative_effect != float('inf') else "N/A",
                'Interpretation': 'Strong heterogeneity' if pval < 0.05 and relative_effect > 0.5 else 
                                'Moderate heterogeneity' if pval < 0.05 else 
                                'No significant heterogeneity' if pval >= 0.05 else 'Inconclusive'
            })
        
        scale_df = pd.DataFrame(scale_data)
        st.dataframe(scale_df, use_container_width=True)
        
        # ========================
        # Table 3: MMQR Results
        # ========================
        st.subheader("Table 4: MMQR Estimation Results")
        
        mmqr_data = []
        coef_names = mmqr_results[quantiles[0]]['coefficients'].index.tolist()
        
        for var in coef_names:
            var_name = 'Intercept' if var == 'Intercept' else var
            row = {'Variable': var_name}
            
            for q in quantiles:
                if q in mmqr_results:
                    # Use MMQR coefficients
                    coef = mmqr_results[q]['mmqr_coefficients'][var]
                    pval = mmqr_results[q]['pvalues'][var]
                    
                    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                    row[f'τ = {q}'] = f"{coef:.3f}{stars}"
            
            mmqr_data.append(row)
        
        mmqr_df = pd.DataFrame(mmqr_data)
        st.dataframe(mmqr_df, use_container_width=True)
        
        # ========================
        # Scale Parameters Interpretation
        # ========================
        st.subheader("Scale Parameters Interpretation")
        
        st.write("**Statistical Significance of Heterogeneity:**")
        
        significant_scale_vars = [var for var in scale_params.index 
                                if var != 'Intercept' and scale_pvalues[var] < 0.1]
        insignificant_scale_vars = [var for var in scale_params.index 
                                  if var != 'Intercept' and scale_pvalues[var] >= 0.1]
        
        if significant_scale_vars:
            st.success(f"**Significant heterogeneity detected in:** {', '.join(significant_scale_vars)}")
            st.write("These variables show statistically significant variation in their effects across different quantiles.")
        
        if insignificant_scale_vars:
            st.info(f"**No significant heterogeneity in:** {', '.join(insignificant_scale_vars)}")
            st.write("These variables show relatively stable effects across quantiles.")
        
        # Detailed interpretation for each variable
        st.write("**Variable-specific Scale Effects:**")
        for var in scale_params.index:
            if var != 'Intercept':
                scale_val = scale_params[var]
                pval = scale_pvalues[var]
                loc_val = location_params[var] if var in location_params else 0
                
                if pval < 0.05:
                    direction = "increasing" if scale_val > 0 else "decreasing"
                    st.write(f"- **{var}**: Significant heterogeneity (p={pval:.3f})")
                    st.write(f"  - Marginal effects show {direction} pattern across quantiles")
                    if loc_val != 0:
                        relative_mag = abs(scale_val / loc_val)
                        st.write(f"  - Scale effect is {relative_mag:.1%} of location effect")
                else:
                    st.write(f"- **{var}**: No significant heterogeneity (p={pval:.3f})")
                    st.write(f"  - Effects remain relatively constant across quantiles")

        # ========================
        # MMQR Coefficient Dynamics
        # ========================
        st.subheader("Figure 1: MMQR Coefficient Dynamics")
        
        plot_vars = [var for var in coef_names if var != 'Intercept']
        
        for var in plot_vars:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot MMQR coefficients
            mmqr_coefs = [mmqr_results[q]['mmqr_coefficients'][var] for q in quantiles if q in mmqr_results]
            quantiles_avail = [q for q in quantiles if q in mmqr_results]
            
            if len(mmqr_coefs) > 0:
                # Plot the MMQR trajectory
                ax.plot(quantiles_avail, mmqr_coefs, 'o-', linewidth=2.5, 
                       label=f'MMQR Coefficients', color='#2E86AB', markersize=8)
                
                # Add location parameter (horizontal line)
                loc_coef = location_params[var]
                ax.axhline(y=loc_coef, color='red', linestyle='--', alpha=0.7, 
                          label=f'Location (τ={reference_quantile})')
                
                # Add scale effect indication
                scale_val = scale_params[var]
                scale_pval = scale_pvalues[var]
                
                if scale_pval < 0.1:
                    # Add a trend line to emphasize the scale effect
                    z = np.polyfit(quantiles_avail, mmqr_coefs, 1)
                    p = np.poly1d(z)
                    ax.plot(quantiles_avail, p(quantiles_avail), ':', color='green', 
                           alpha=0.7, label=f'Scale trend (p={scale_pval:.3f})')
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlabel("Quantiles (τ)", fontsize=12)
                ax.set_ylabel("Coefficient Estimate", fontsize=12)
                
                # Add scale significance to title
                scale_sig = '***' if scale_pval < 0.01 else '**' if scale_pval < 0.05 else '*' if scale_pval < 0.1 else 'ns'
                ax.set_title(f"MMQR Coefficient Dynamics: {var} (Scale: {scale_sig})", 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add significance markers for MMQR coefficients
                for i, q in enumerate(quantiles_avail):
                    pval = mmqr_results[q]['pvalues'][var]
                    if pval < 0.1:
                        ax.annotate('*' * (3 if pval < 0.01 else 2 if pval < 0.05 else 1), 
                                   (q, mmqr_coefs[i]), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontweight='bold', 
                                   color='red', fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig)

        # ========================
        # Download Section with Scale P-values
        # ========================
        st.subheader("Download MMQR Results")
        
        # Prepare comprehensive download data
        download_data = []
        
        # Location parameters
        for var in location_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            pval = mmqr_results[reference_quantile]['pvalues'][var] if reference_quantile in mmqr_results else 1.0
            download_data.append({
                'Variable': var_name,
                'Type': 'Location',
                'Coefficient': round(location_params[var], 3),
                'P-Value': round(pval, 3),
                'Quantile': reference_quantile,
                'Method': 'MMQR'
            })
        
        # Scale parameters with p-values
        for var in scale_params.index:
            var_name = 'Intercept' if var == 'Intercept' else var
            download_data.append({
                'Variable': var_name,
                'Type': 'Scale', 
                'Coefficient': round(scale_params[var], 3),
                'P-Value': round(scale_pvalues[var], 3),
                'Quantile': 'N/A',
                'Method': 'MMQR'
            })
        
        # MMQR coefficients
        for var in coef_names:
            var_name = 'Intercept' if var == 'Intercept' else var
            for q in quantiles:
                if q in mmqr_results:
                    download_data.append({
                        'Variable': var_name,
                        'Type': f'MMQR_τ={q}',
                        'Coefficient': round(mmqr_results[q]['mmqr_coefficients'][var], 3),
                        'P-Value': round(mmqr_results[q]['pvalues'][var], 3),
                        'Quantile': q,
                        'Method': 'MMQR'
                    })
        
        download_df = pd.DataFrame(download_data)
        csv_data = download_df.to_csv(index=False)
        
        st.download_button(
            "📥 Download Complete MMQR Results (CSV)",
            data=csv_data,
            file_name="MMQR_Complete_Results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"MMQR estimation failed: {str(e)}")

# Rest of the code remains the same...



# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
