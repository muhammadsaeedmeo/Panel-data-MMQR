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
# Enhanced MMQR Visualization Section
# ============================================

st.subheader("Figure 1: MMQR Coefficient Dynamics")

# Visualization Configuration
st.markdown("### 🎨 Visualization Settings")

col1, col2, col3 = st.columns(3)
with col1:
    # Color selection
    line_color = st.color_picker("Line Color", "#2E86AB")
    confidence_color = st.color_picker("Confidence Interval Color", "#2E86AB", key="ci_color")
    
with col2:
    # Style options
    line_style = st.selectbox("Line Style", ["solid", "dashed", "dashdot", "dotted"])
    marker_style = st.selectbox("Marker Style", ["circle", "square", "diamond", "triangle-up"])
    
with col3:
    # Layout options
    grid_style = st.selectbox("Grid Style", ["light", "medium", "dark", "none"])
    fig_size = st.selectbox("Figure Size", ["medium", "large", "extra-large"])

# Size mapping
size_map = {"medium": (10, 6), "large": (12, 8), "extra-large": (14, 10)}

plot_vars = [var for var in coef_names if var != 'Intercept']

# Create individual plots for each variable with enhanced styling
for var in plot_vars:
    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=size_map[fig_size])
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot data
    mmqr_coefs = [mmqr_results[q]['mmqr_coefficients'][var] for q in quantiles if q in mmqr_results]
    quantiles_avail = [q for q in quantiles if q in mmqr_results]
    
    if len(mmqr_coefs) > 0:
        # Main coefficient line with enhanced styling
        line = ax.plot(quantiles_avail, mmqr_coefs, 
                      marker=marker_style,
                      markersize=10,
                      linewidth=3,
                      linestyle=line_style,
                      color=line_color,
                      label='MMQR Coefficients',
                      alpha=0.9,
                      markerfacecolor='white',
                      markeredgecolor=line_color,
                      markeredgewidth=2)
        
        # Add location parameter (horizontal line)
        loc_coef = location_params[var]
        loc_line = ax.axhline(y=loc_coef, 
                             color='#FF6B6B', 
                             linestyle='--', 
                             linewidth=2.5,
                             alpha=0.8,
                             label=f'Location (τ={reference_quantile})')
        
        # Add confidence intervals with enhanced styling
        if bootstrap_ci:
            try:
                lower = [mmqr_results[q]['conf_int'].loc[var, 0] for q in quantiles_avail]
                upper = [mmqr_results[q]['conf_int'].loc[var, 1] for q in quantiles_avail]
                ax.fill_between(quantiles_avail, lower, upper, 
                              alpha=0.2, 
                              color=confidence_color,
                              label='95% Confidence Interval')
            except:
                pass
        
        # Zero line for reference
        zero_line = ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Customize grid
        if grid_style != "none":
            alpha_map = {"light": 0.2, "medium": 0.4, "dark": 0.6}
            ax.grid(True, alpha=alpha_map[grid_style], linestyle='-', linewidth=0.5)
        
        # Customize spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Labels and title (clean, minimal)
        ax.set_xlabel("Quantiles (τ)", fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel("Coefficient Estimate", fontsize=14, fontweight='bold', labelpad=10)
        
        # Get scale significance for subtitle
        scale_pval = scale_pvalues[var]
        scale_sig = '***' if scale_pval < 0.01 else '**' if scale_pval < 0.05 else '*' if scale_pval < 0.1 else 'ns'
        
        # Main title
        ax.set_title(f"MMQR Analysis: {var}", 
                   fontsize=16, 
                   fontweight='bold', 
                   pad=20,
                   color='#2C3E50')
        
        # Add statistical information as text below the plot
        stats_text = f"Location: {loc_coef:.3f} | Scale p-value: {scale_pval:.3f} ({scale_sig})"
        ax.text(0.5, -0.15, stats_text, 
               transform=ax.transAxes, 
               fontsize=11,
               ha='center',
               va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        
        # Enhanced legend
        legend = ax.legend(bbox_to_anchor=(1.05, 1), 
                          loc='upper left',
                          borderaxespad=0.,
                          frameon=True,
                          fancybox=True,
                          shadow=True,
                          framealpha=0.9)
        
        # Customize tick parameters
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='x', which='major', pad=5)
        ax.tick_params(axis='y', which='major', pad=5)
        
        # Set background color
        ax.set_facecolor('#F8F9FA')
        fig.patch.set_facecolor('white')
        
        # Add subtle border to figure
        for spine in ax.spines.values():
            spine.set_edgecolor('#BDC3C7')
        
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # ========================
        # Plot Interpretation Box
        # ========================
        
        # Create interpretation based on the pattern
        coef_range = max(mmqr_coefs) - min(mmqr_coefs)
        trend = "increasing" if mmqr_coefs[-1] > mmqr_coefs[0] else "decreasing" if mmqr_coefs[-1] < mmqr_coefs[0] else "stable"
        
        # Count significant quantiles
        sig_quantiles = sum(1 for q in quantiles_avail if mmqr_results[q]['pvalues'][var] < 0.1)
        
        # Interpretation text
        interpretation = f"""
        **Interpretation for {var}:**
        
        - **Overall Trend**: {trend} marginal effects across quantiles
        - **Effect Range**: {min(mmqr_coefs):.3f} to {max(mmqr_coefs):.3f}
        - **Statistical Significance**: {sig_quantiles} out of {len(quantiles_avail)} quantiles significant
        - **Heterogeneity**: {'Present' if scale_pval < 0.1 else 'Not detected'}
        - **Economic Significance**: {'Strong' if coef_range > abs(loc_coef) * 0.5 else 'Moderate' if coef_range > abs(loc_coef) * 0.2 else 'Weak'}
        """
        
        # Color code the interpretation box
        if scale_pval < 0.05 and coef_range > abs(loc_coef) * 0.3:
            st.success(interpretation)
        elif scale_pval < 0.1:
            st.info(interpretation)
        else:
            st.warning(interpretation)
        
        # Add some space between plots
        st.markdown("---")

# ========================
# Combined Plot Option
# ========================

st.subheader("Figure 2: Combined MMQR Analysis")

# Option to show all variables in one plot
if st.checkbox("Show combined plot (all variables)"):
    
    # Color palette for multiple lines
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E92CC', '#6A8EAE']
    
    fig_combined, ax_combined = plt.subplots(figsize=(14, 9))
    
    # Plot each variable
    for i, var in enumerate(plot_vars):
        if i < len(colors):  # Ensure we don't exceed color list
            mmqr_coefs = [mmqr_results[q]['mmqr_coefficients'][var] for q in quantiles if q in mmqr_results]
            quantiles_avail = [q for q in quantiles if q in mmqr_results]
            
            if len(mmqr_coefs) > 0:
                ax_combined.plot(quantiles_avail, mmqr_coefs,
                               marker=marker_style,
                               markersize=8,
                               linewidth=2.5,
                               color=colors[i],
                               label=var,
                               alpha=0.8)
    
    # Styling for combined plot
    ax_combined.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_combined.set_xlabel("Quantiles (τ)", fontsize=13, fontweight='bold')
    ax_combined.set_ylabel("Coefficient Estimate", fontsize=13, fontweight='bold')
    ax_combined.set_title("Combined MMQR Coefficient Dynamics", fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced legend
    ax_combined.legend(bbox_to_anchor=(1.05, 1), 
                      loc='upper left',
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      framealpha=0.9)
    
    # Grid and spines
    if grid_style != "none":
        alpha_map = {"light": 0.2, "medium": 0.4, "dark": 0.6}
        ax_combined.grid(True, alpha=alpha_map[grid_style])
    
    ax_combined.spines['top'].set_visible(False)
    ax_combined.spines['right'].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig_combined)
    
    st.info("**Combined Plot Interpretation:** Compare the quantile dynamics across all variables. Lines show how marginal effects evolve across different quantiles of the distribution.")

# ========================
# Download Enhanced Plots
# ========================

st.subheader("Download Enhanced Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Download individual plots
    if st.button("📥 Download Individual Plots as PNG"):
        for i, var in enumerate(plot_vars):
            # Recreate each plot for download (higher quality)
            fig_dl, ax_dl = plt.subplots(figsize=(10, 7))
            
            mmqr_coefs = [mmqr_results[q]['mmqr_coefficients'][var] for q in quantiles if q in mmqr_results]
            quantiles_avail = [q for q in quantiles if q in mmqr_results]
            
            if len(mmqr_coefs) > 0:
                ax_dl.plot(quantiles_avail, mmqr_coefs, 
                          marker='o', markersize=8, linewidth=3, color=line_color)
                ax_dl.axhline(y=location_params[var], color='red', linestyle='--', linewidth=2)
                ax_dl.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax_dl.set_xlabel("Quantiles (τ)", fontsize=12)
                ax_dl.set_ylabel("Coefficient Estimate", fontsize=12)
                ax_dl.set_title(f"MMQR: {var}", fontsize=14)
                ax_dl.grid(True, alpha=0.3)
                
                # Save individual plot
                buf = BytesIO()
                fig_dl.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button(
                    f"Download {var} Plot",
                    data=buf.getvalue(),
                    file_name=f"MMQR_{var}.png",
                    mime="image/png",
                    key=f"dl_{var}"
                )
                
            plt.close(fig_dl)

with col2:
    # Download settings summary
    settings_summary = f"""
    MMQR Visualization Settings:
    - Line Color: {line_color}
    - Line Style: {line_style}
    - Marker Style: {marker_style}
    - Grid Style: {grid_style}
    - Figure Size: {fig_size}
    - Variables: {', '.join(plot_vars)}
    """
    
    st.download_button(
        "📋 Download Plot Settings",
        data=settings_summary,
        file_name="MMQR_Plot_Settings.txt",
        mime="text/plain"
    )

# ========================
# Additional Visualization: Coefficient Range
# ========================

st.subheader("Figure 3: Coefficient Range Across Quantiles")

# Create a horizontal bar chart showing the range of coefficients
fig_range, ax_range = plt.subplots(figsize=(12, 6))

ranges = []
min_vals = []
max_vals = []

for var in plot_vars:
    mmqr_coefs = [mmqr_results[q]['mmqr_coefficients'][var] for q in quantiles if q in mmqr_results]
    if mmqr_coefs:
        ranges.append(max(mmqr_coefs) - min(mmqr_coefs))
        min_vals.append(min(mmqr_coefs))
        max_vals.append(max(mmqr_coefs))

y_pos = np.arange(len(plot_vars))

# Create horizontal bars
bars = ax_range.barh(y_pos, ranges, left=min_vals, alpha=0.7, color=line_color)
ax_range.set_yticks(y_pos)
ax_range.set_yticklabels(plot_vars)
ax_range.set_xlabel("Coefficient Range", fontsize=12)
ax_range.set_title("Range of MMQR Coefficients Across Quantiles", fontsize=14, fontweight='bold')

# Add value annotations
for i, (min_val, max_val) in enumerate(zip(min_vals, max_vals)):
    ax_range.text(max_val + (max_val - min_val) * 0.01, i, f"{max_val:.3f}", 
                 va='center', fontsize=9, fontweight='bold')
    ax_range.text(min_val - (max_val - min_val) * 0.01, i, f"{min_val:.3f}", 
                 va='center', ha='right', fontsize=9, fontweight='bold')

ax_range.grid(True, alpha=0.3, axis='x')
ax_range.spines['top'].set_visible(False)
ax_range.spines['right'].set_visible(False)

plt.tight_layout()
st.pyplot(fig_range)

st.info("**Range Plot Interpretation:** Shows the minimum and maximum coefficient values for each variable across all quantiles. Wider bars indicate greater heterogeneity in effects.")

# Rest of the code remains the same...
# ============================================
# Section E: Quantile Cointegration Test
# ============================================

st.header("E. Quantile Cointegration Test")

try:
    # Check if we have the required variables
    if 'dep_var' not in locals() or 'indep_vars' not in locals() or not indep_vars:
        st.warning("Please complete the MMQR analysis first to select variables.")
    else:
        st.subheader("Quantile Cointegration Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            coint_quantiles = st.text_input("Cointegration Quantiles (comma-separated)", 
                                          "0.05,0.25,0.50,0.75,0.95", key="coint_quantiles")
            coint_quantiles = [float(q.strip()) for q in coint_quantiles.split(",")]
        
        with col2:
            max_lags_coint = st.slider("Maximum Lags for Cointegration Test", 1, 5, 2, key="coint_lags")
            significance_level = st.selectbox("Significance Level", [0.01, 0.05, 0.10], index=1)
        
        # Quantile Cointegration Test Implementation
        def quantile_cointegration_test(data, y_var, x_vars, quantiles, max_lags):
            """
            Quantile Cointegration Test based on Xiao (2009) and related approaches
            """
            results = {}
            residual_tests = {}
            
            # Ensure panel data structure
            if 'Country' in data.columns and 'Year' in data.columns:
                # Use first country for demonstration (or pool data)
                country_data = data[data['Country'] == data['Country'].iloc[0]].sort_values('Year')
                test_data = country_data[[y_var] + x_vars].dropna()
            else:
                test_data = data[[y_var] + x_vars].dropna()
            
            if len(test_data) < max_lags + 5:
                st.warning("Insufficient data for cointegration test. Need more time periods.")
                return results, residual_tests
            
            for q in quantiles:
                try:
                    # Step 1: Quantile regression for long-run relationship
                    formula = f"{y_var} ~ {' + '.join(x_vars)}"
                    q_model = quantreg(formula, test_data).fit(q=q)
                    
                    # Step 2: Get residuals from quantile regression
                    residuals = q_model.resid
                    
                    # Step 3: Test residuals for stationarity (ADF test)
                    from statsmodels.tsa.stattools import adfuller
                    
                    # ADF test on residuals
                    adf_result = adfuller(residuals, maxlag=max_lags, autolag='AIC')
                    
                    # Store results
                    results[q] = {
                        'quantile': q,
                        'adf_statistic': adf_result[0],
                        'p_value': adf_result[1],
                        'critical_values': adf_result[4],
                        'residuals': residuals,
                        'coefficients': q_model.params,
                        'model': q_model
                    }
                    
                    # Additional residual diagnostics
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_test = acorr_ljungbox(residuals, lags=[max_lags], return_df=True)
                    
                    residual_tests[q] = {
                        'ljung_box_stat': lb_test.iloc[0]['lb_stat'],
                        'ljung_box_pval': lb_test.iloc[0]['lb_pvalue'],
                        'residual_mean': np.mean(residuals),
                        'residual_std': np.std(residuals)
                    }
                    
                except Exception as e:
                    st.warning(f"Quantile cointegration test failed for quantile {q}: {str(e)}")
                    continue
            
            return results, residual_tests
        
        # Run quantile cointegration test
        st.info("Running Quantile Cointegration Tests...")
        coint_results, residual_tests = quantile_cointegration_test(
            df, dep_var, indep_vars, coint_quantiles, max_lags_coint
        )
        
        if coint_results:
            # ========================
            # Table 1: Cointegration Test Results
            # ========================
            st.subheader("Table 5: Quantile Cointegration Test Results")
            
            coint_data = []
            for q in coint_quantiles:
                if q in coint_results:
                    result = coint_results[q]
                    critical_vals = result['critical_values']
                    
                    # Get critical values safely
                    critical_1p = critical_vals.get('1%', 'N/A')
                    critical_5p = critical_vals.get('5%', 'N/A')
                    critical_10p = critical_vals.get('10%', 'N/A')
                    
                    # Format critical values
                    critical_1p_str = f"{critical_1p:.3f}" if critical_1p != 'N/A' else 'N/A'
                    critical_5p_str = f"{critical_5p:.3f}" if critical_5p != 'N/A' else 'N/A'
                    critical_10p_str = f"{critical_10p:.3f}" if critical_10p != 'N/A' else 'N/A'
                    
                    # Determine cointegration decision
                    is_cointegrated = result['p_value'] < significance_level
                    
                    coint_data.append({
                        'Quantile (τ)': f"{q:.2f}",
                        'ADF Statistic': f"{result['adf_statistic']:.3f}",
                        'P-Value': f"{result['p_value']:.3f}",
                        '1% Critical': critical_1p_str,
                        '5% Critical': critical_5p_str,
                        '10% Critical': critical_10p_str,
                        'Cointegrated': 'Yes' if is_cointegrated else 'No'
                    })
            
            coint_df = pd.DataFrame(coint_data)
            st.dataframe(coint_df, use_container_width=True)
            
            # ========================
            # Table 2: Long-run Coefficients
            # ========================
            st.subheader("Table 6: Long-run Coefficients from Quantile Cointegration")
            
            coef_data = []
            coef_names = coint_results[coint_quantiles[0]]['coefficients'].index.tolist()
            
            for var in coef_names:
                var_name = 'Intercept' if var == 'Intercept' else var
                row = {'Variable': var_name}
                
                for q in coint_quantiles:
                    if q in coint_results:
                        coef = coint_results[q]['coefficients'][var]
                        row[f'τ = {q}'] = f"{coef:.3f}"
                
                coef_data.append(row)
            
            coef_df = pd.DataFrame(coef_data)
            st.dataframe(coef_df, use_container_width=True)
            
            # ========================
            # Visualization: Cointegration Results
            # ========================
            st.subheader("Figure 2: Quantile Cointegration Analysis")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: ADF Statistics across quantiles
            quantiles_plot = [q for q in coint_quantiles if q in coint_results]
            adf_stats = [coint_results[q]['adf_statistic'] for q in quantiles_plot]
            critical_1p = [coint_results[q]['critical_values'].get('1%', np.nan) for q in quantiles_plot]
            critical_5p = [coint_results[q]['critical_values'].get('5%', np.nan) for q in quantiles_plot]
            
            axes[0,0].plot(quantiles_plot, adf_stats, 'o-', linewidth=2, label='ADF Statistic', color='blue')
            axes[0,0].plot(quantiles_plot, critical_1p, '--', linewidth=2, label='1% Critical', color='red')
            axes[0,0].plot(quantiles_plot, critical_5p, '--', linewidth=2, label='5% Critical', color='orange')
            
            axes[0,0].set_xlabel('Quantiles')
            axes[0,0].set_ylabel('ADF Statistics')
            axes[0,0].set_title('ADF Statistics vs Critical Values across Quantiles')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: P-values across quantiles
            p_values = [coint_results[q]['p_value'] for q in quantiles_plot]
            axes[0,1].plot(quantiles_plot, p_values, 's-', linewidth=2, color='green')
            axes[0,1].axhline(y=significance_level, color='red', linestyle='--', 
                            label=f'{significance_level} Significance Level')
            axes[0,1].set_xlabel('Quantiles')
            axes[0,1].set_ylabel('P-Values')
            axes[0,1].set_title('P-Values across Quantiles')
            axes[0,1].set_yscale('log')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Residuals from median quantile
            if 0.5 in coint_results:
                residuals = coint_results[0.5]['residuals']
                axes[1,0].plot(residuals.index, residuals.values, linewidth=1.5, color='purple')
                axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1,0].set_xlabel('Time/Index')
                axes[1,0].set_ylabel('Residuals')
                axes[1,0].set_title('Residuals from Median Quantile Cointegration')
                axes[1,0].grid(True, alpha=0.3)
            
            # Plot 4: Cointegration persistence (coefficient variation)
            coef_variation = []
            for var in coef_names:
                if var != 'Intercept':
                    coefs = [coint_results[q]['coefficients'][var] for q in quantiles_plot if q in coint_results]
                    if coefs and len(coefs) > 1:
                        variation = np.std(coefs) / abs(np.mean(coefs)) if np.mean(coefs) != 0 else np.std(coefs)
                        coef_variation.append((var, variation))
            
            if coef_variation:
                variables, variations = zip(*coef_variation)
                axes[1,1].bar(variables, variations, color='steelblue', alpha=0.7)
                axes[1,1].set_xlabel('Variables')
                axes[1,1].set_ylabel('Coefficient of Variation')
                axes[1,1].set_title('Long-run Coefficient Persistence across Quantiles')
                axes[1,1].tick_params(axis='x', rotation=45)
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # ========================
            # Residual Diagnostics
            # ========================
            st.subheader("Table 7: Residual Diagnostics")
            
            residual_data = []
            for q in coint_quantiles:
                if q in residual_tests:
                    test = residual_tests[q]
                    residual_data.append({
                        'Quantile': f"{q:.2f}",
                        'Ljung-Box Stat': f"{test['ljung_box_stat']:.3f}",
                        'Ljung-Box P-value': f"{test['ljung_box_pval']:.3f}",
                        'Residual Mean': f"{test['residual_mean']:.3f}",
                        'Residual Std': f"{test['residual_std']:.3f}",
                        'AutoCorrelation': 'Present' if test['ljung_box_pval'] < 0.05 else 'Absent'
                    })
            
            if residual_data:
                residual_df = pd.DataFrame(residual_data)
                st.dataframe(residual_df, use_container_width=True)
            
            # ========================
            # Interpretation
            # ========================
            st.subheader("Economic Interpretation")
            
            # Count cointegrated quantiles
            cointegrated_count = sum(1 for q in coint_quantiles 
                                   if q in coint_results and coint_results[q]['p_value'] < significance_level)
            total_tested = len([q for q in coint_quantiles if q in coint_results])
            
            st.write(f"**Cointegration Summary:**")
            st.write(f"- {cointegrated_count} out of {total_tested} quantiles show evidence of cointegration")
            st.write(f"- Significance level: {significance_level}")
            
            if cointegrated_count > total_tested * 0.5:
                st.success("**Strong evidence of quantile cointegration** - Long-run relationship exists across most quantiles")
            elif cointegrated_count > 0:
                st.info("**Partial quantile cointegration** - Long-run relationship exists at specific quantiles only")
            else:
                st.warning("**No evidence of quantile cointegration** - No stable long-run relationship detected")
            
            # Variable-specific interpretation
            st.write("**Long-run Relationship Analysis:**")
            for var in coef_names:
                if var != 'Intercept':
                    var_name = var
                    coefs = [coint_results[q]['coefficients'][var] for q in coint_quantiles if q in coint_results]
                    if coefs:
                        min_coef = min(coefs)
                        max_coef = max(coefs)
                        mean_coef = np.mean(coefs)
                        
                        st.write(f"- **{var_name}**: Long-run coefficient ranges from {min_coef:.3f} to {max_coef:.3f}")
                        if min_coef * max_coef > 0:  # Same sign
                            direction = "positive" if mean_coef > 0 else "negative"
                            st.write(f"  - Consistent {direction} relationship across quantiles")
                        else:
                            st.write(f"  - Relationship sign varies across quantiles")
            
            # ========================
            # Download Section
            # ========================
            st.subheader("Download Quantile Cointegration Results")
            
            # Prepare comprehensive download data
            download_coint_data = []
            
            for q in coint_quantiles:
                if q in coint_results:
                    result = coint_results[q]
                    residual_test = residual_tests.get(q, {})
                    
                    # Cointegration test results
                    download_coint_data.append({
                        'Quantile': q,
                        'Test_Type': 'Cointegration',
                        'ADF_Statistic': result['adf_statistic'],
                        'P_Value': result['p_value'],
                        'Critical_1%': result['critical_values'].get('1%', np.nan),
                        'Critical_5%': result['critical_values'].get('5%', np.nan),
                        'Critical_10%': result['critical_values'].get('10%', np.nan),
                        'Cointegrated': 'Yes' if result['p_value'] < significance_level else 'No',
                        'Ljung_Box_Stat': residual_test.get('ljung_box_stat', np.nan),
                        'Ljung_Box_Pval': residual_test.get('ljung_box_pval', np.nan),
                        'Residual_Mean': residual_test.get('residual_mean', np.nan),
                        'Residual_Std': residual_test.get('residual_std', np.nan)
                    })
                    
                    # Coefficients for this quantile
                    for var_name, coef_value in result['coefficients'].items():
                        download_coint_data.append({
                            'Quantile': q,
                            'Test_Type': 'Coefficient',
                            'Variable': var_name,
                            'Coefficient_Value': coef_value,
                            'ADF_Statistic': np.nan,
                            'P_Value': np.nan
                        })
            
            download_coint_df = pd.DataFrame(download_coint_data)
            csv_coint = download_coint_df.to_csv(index=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Cointegration Results",
                    data=csv_coint,
                    file_name="Quantile_Cointegration_Results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Copy-paste friendly summary
                st.info("**Copy-Paste Summary:**")
                summary_text = f"Quantile Cointegration Test Results (Significance: {significance_level})\n"
                summary_text += "Quantile | ADF Stat | P-value | Cointegrated\n"
                summary_text += "--------|----------|---------|-------------\n"
                
                for q in coint_quantiles:
                    if q in coint_results:
                        result = coint_results[q]
                        coint_status = "Yes" if result['p_value'] < significance_level else "No"
                        summary_text += f"{q:.2f} | {result['adf_statistic']:.3f} | {result['p_value']:.3f} | {coint_status}\n"
                
                st.text_area("Summary for copying:", summary_text, height=200)
            
            # ========================
            # Methodological Notes
            # ========================
            with st.expander("Methodological Notes on Quantile Cointegration"):
                st.markdown("""
                **Quantile Cointegration Approach:**
                
                1. **Method**: Based on Xiao (2009) quantile cointegration framework
                2. **Procedure**:
                   - Estimate quantile regression for long-run relationship at each quantile
                   - Test residuals for stationarity using ADF test
                   - Cointegration exists if residuals are stationary
                
                3. **Interpretation**:
                   - Significant ADF statistic indicates cointegration at that quantile
                   - Variation in coefficients shows quantile-specific long-run effects
                   - Consistent cointegration across quantiles suggests robust relationship
                
                4. **Limitations**:
                   - Assumes linear cointegration relationship within quantiles
                   - Requires sufficient time series length
                   - Panel data adaptation may require more sophisticated tests
                
                **Reference**: Xiao, Z. (2009). Quantile cointegrating regression. 
                Journal of Econometrics, 150(2), 248-260.
                """)
        
        else:
            st.error("No quantile cointegration results could be computed. Check data requirements.")

except Exception as e:
    st.error(f"Error in quantile cointegration test: {str(e)}")
    st.info("""
    Common issues:
    - Insufficient time series length
    - Non-stationary variables (consider differencing)
    - Missing values in the data
    - Multicollinearity between variables
    """)
# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("**Panel Data Analysis Dashboard** | Built with Streamlit")
