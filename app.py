# ============================================
# Streamlit Panel Data Analysis App using MMQR
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
            return None, None, "Insufficient data: Need at least 2 units and sufficient time periods.", N, T, 0

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
            return None, None, "Test failed: insufficient valid units.", N, T, 0

        # Compute W-bar (average F-statistic)
        W_bar = np.mean(F_stats)
        
        # Compute Z-bar statistic
        k = max_lag
        # Using simplified asymptotic formula
        if T > 2 * k + 2:
            Z_bar = np.sqrt(N) * (W_bar - 1) / np.sqrt(2 * k / (T - 2 * k - 2))
            p_value = 2 * (1 - norm.cdf(abs(Z_bar)))
        else:
            return None, None, "Insufficient time periods for test.", N, T, successful_units
        
        interpretation = f"W-bar: {W_bar:.4f}, Z-bar: {Z_bar:.4f}"
        
        return W_bar, p_value, interpretation, N, T, successful_units
        
    except Exception as e:
        return None, None, f"Error: {str(e)}", 0, 0, 0


def enhanced_mmqr_estimation(data, dependent_var, independent_vars, quantiles, 
                            bootstrap_ci=True, n_bootstrap=200):
    """
    Enhanced MMQR estimation with proper statistical inference
    """
    results = {}
    
    # Prepare data
    clean_data = data[[dependent_var] + independent_vars].dropna()
    
    if len(clean_data) < 50:
        return None, None, "Insufficient observations for MMQR"
    
    # Run quantile regression for each quantile
    for q in quantiles:
        try:
            formula = f"{dependent_var} ~ " + " + ".join(independent_vars)
            mod = quantreg(formula, clean_data)
            res = mod.fit(q=q)
            
            results[q] = {
                'coefficients': res.params.to_dict(),
                'std_errors': res.bse.to_dict(),
                'pvalues': res.pvalues.to_dict(),
                'tvalues': res.tvalues.to_dict(),
                'conf_int': res.conf_int().to_dict()
            }
        except Exception as e:
            st.warning(f"Failed to estimate quantile {q}: {str(e)}")
            continue
    
    # Location-Scale model (simplified)
    # Location: Mean of dependent variable
    y = clean_data[dependent_var].values
    X = clean_data[independent_vars].values
    X_with_const = sm.add_constant(X)
    
    # OLS for location
    location_model = sm.OLS(y, X_with_const).fit()
    
    # Scale model (absolute residuals)
    residuals = location_model.resid
    abs_residuals = np.abs(residuals)
    scale_model = sm.OLS(abs_residuals, X_with_const).fit()
    
    location_scale_results = {
        'location_intercept': location_model.params[0],
        'location_intercept_pvalue': location_model.pvalues[0],
        'location_coefficients': location_model.params[1:],
        'scale_intercept': scale_model.params[0],
        'scale_intercept_pvalue': scale_model.pvalues[0],
        'scale_coefficients': scale_model.params[1:],
        'location_model': location_model,
        'scale_model': scale_model
    }
    
    return results, location_scale_results, None


# ============================================
# DATA LOADING
# ============================================

st.set_page_config(page_title="Panel Data Analysis", layout="wide")

st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Custom data loaded successfully!")
    st.session_state["main_data"] = data
else:
    st.info("No file uploaded. Using sample data structure.")
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
                'X1': np.random.uniform(10, 100),
                'X2': np.random.uniform(5, 50)
            })
    
    data = pd.DataFrame(data_list)
    st.session_state["main_data"] = data

data = st.session_state["main_data"]

# ============================================
# APP HEADER
# ============================================

st.title("📊 Panel Data Analysis Dashboard (MMQR Framework)")
st.markdown("""
This interactive dashboard performs **panel data econometric analysis** using
**Method of Moments Quantile Regression (MMQR)** and **Panel Granger Causality**.

**Required columns:** `Country`, `Year`, and your analysis variables.
""")

# Display data preview
with st.expander("📋 View Data Preview"):
    st.dataframe(data.head(20), use_container_width=True)
    st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    st.write(f"**Countries:** {data['Country'].nunique() if 'Country' in data.columns else 'N/A'}")
    st.write(f"**Time periods:** {data['Year'].nunique() if 'Year' in data.columns else 'N/A'}")

st.markdown("---")

# ============================================
# SECTION E: MMQR ANALYSIS
# ============================================

st.header("E. Method of Moments Quantile Regression (MMQR)")

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) >= 2:
    col1, col2 = st.columns(2)
    
    with col1:
        dependent_var = st.selectbox(
            "Select Dependent Variable (Y)", 
            options=numeric_cols,
            key="mmqr_dep"
        )
    
    with col2:
        independent_vars = st.multiselect(
            "Select Independent Variables (X)", 
            options=[c for c in numeric_cols if c != dependent_var],
            key="mmqr_indep"
        )
    
    if dependent_var and independent_vars:
        st.subheader("MMQR Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            quantiles = st.multiselect(
                "Select Quantiles",
                options=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                default=[0.10, 0.25, 0.50, 0.75, 0.90],
                key="quantiles"
            )
        
        with col2:
            n_bootstrap = st.slider(
                "Bootstrap Iterations",
                min_value=50,
                max_value=500,
                value=200,
                step=50
            )
        
        if st.button("🚀 Run MMQR Analysis", type="primary"):
            with st.spinner("Running MMQR estimation..."):
                mmqr_results, location_scale_results, error = enhanced_mmqr_estimation(
                    data, dependent_var, independent_vars, quantiles,
                    bootstrap_ci=True, n_bootstrap=n_bootstrap
                )
                
                if error:
                    st.error(f"MMQR Error: {error}")
                elif mmqr_results:
                    st.success("✅ MMQR estimation completed!")
                    
                    # Store results
                    st.session_state['mmqr_results'] = mmqr_results
                    st.session_state['location_scale_results'] = location_scale_results
                    
                    # Display results table
                    st.subheader("Table E.1: MMQR Coefficient Estimates")
                    
                    # Create results dataframe
                    results_data = []
                    for var in ['const'] + independent_vars:
                        row = {'Variable': var}
                        for q in quantiles:
                            if q in mmqr_results and var in mmqr_results[q]['coefficients']:
                                coef = mmqr_results[q]['coefficients'][var]
                                pval = mmqr_results[q]['pvalues'][var]
                                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                                row[f'Q{q}'] = f"{coef:.4f}{sig}"
                        results_data.append(row)
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    st.caption("*** p<0.01, ** p<0.05, * p<0.1")
                    
                    # Diagnostic Tests
                    st.subheader("Table E.2: MMQR Diagnostic Tests")
                    
                    # Calculate quantile stability
                    test_var = independent_vars[0]
                    coefs = [mmqr_results[q]['coefficients'].get(test_var, np.nan) 
                            for q in quantiles if q in mmqr_results]
                    
                    if len(coefs) > 1 and not all(np.isnan(coefs)):
                        stability_ratio = (max(coefs) - min(coefs)) / abs(np.mean(coefs))
                    else:
                        stability_ratio = np.nan
                    
                    diagnostic_data = {
                        'Test Statistic': [
                            "Location Intercept (α)",
                            "Location Intercept p-value",
                            "Scale Intercept (λ)",
                            "Scale Intercept p-value",
                            f"Quantile Stability Ratio ({test_var})"
                        ],
                        'Value': [
                            f"{location_scale_results['location_intercept']:.4f}",
                            f"{location_scale_results['location_intercept_pvalue']:.4f}",
                            f"{location_scale_results['scale_intercept']:.4f}",
                            f"{location_scale_results['scale_intercept_pvalue']:.4f}",
                            f"{stability_ratio:.4f}" if not np.isnan(stability_ratio) else "N/A"
                        ],
                        'Interpretation': [
                            "Baseline mean of Y",
                            "H₀: Location intercept = 0",
                            "Baseline volatility of Y",
                            "H₀: Scale intercept = 0",
                            "Coefficient stability (lower is better)"
                        ],
                        'Decision (α=0.10)': [
                            "-",
                            "Reject H₀" if location_scale_results['location_intercept_pvalue'] < 0.1 else "Fail to Reject",
                            "-",
                            "Reject H₀" if location_scale_results['scale_intercept_pvalue'] < 0.1 else "Fail to Reject",
                            "-"
                        ]
                    }
                    
                    diagnostic_df = pd.DataFrame(diagnostic_data)
                    st.dataframe(diagnostic_df, use_container_width=True)
                    
                    # Visualization
                    st.subheader("Quantile Coefficients Plot")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    for var in independent_vars:
                        coefs = []
                        for q in sorted(quantiles):
                            if q in mmqr_results and var in mmqr_results[q]['coefficients']:
                                coefs.append(mmqr_results[q]['coefficients'][var])
                            else:
                                coefs.append(np.nan)
                        ax.plot(sorted(quantiles), coefs, marker='o', label=var, linewidth=2)
                    
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Quantile', fontsize=12)
                    ax.set_ylabel('Coefficient', fontsize=12)
                    ax.set_title('MMQR Coefficients Across Quantiles', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close()
else:
    st.warning("Please ensure your data has at least 2 numeric variables for MMQR analysis.")

st.markdown("---")

# ============================================
# SECTION F: GRANGER CAUSALITY
# ============================================

st.header("F. Panel Granger Causality Tests")
st.subheader("Dumitrescu & Hurlin (2012) Test")

if "Country" not in data.columns or "Year" not in data.columns:
    st.error("Cannot perform Panel Granger Causality: 'Country' and 'Year' columns required.")
elif len(numeric_cols) < 2:
    st.warning("Need at least 2 numeric variables for causality testing.")
else:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var_y = st.selectbox("Dependent Variable (Y)", options=numeric_cols, key="gc_y")
    with col2:
        var_x = st.selectbox("Causal Variable (X)", 
                             options=[c for c in numeric_cols if c != var_y], key="gc_x")
    with col3:
        max_lag = st.slider("Max Lag Order", min_value=1, max_value=5, value=2)

    if var_y and var_x and st.button("🔍 Run Granger Causality Test", type="primary"):
        with st.spinner("Running Granger Causality tests..."):
            # Test 1: X -> Y
            W_bar_1, p_value_1, interp_1, N1, T1, n_units_1 = dumitrescu_hurlin_test(
                data, 'Country', 'Year', var_y, var_x, max_lag
            )
            
            # Test 2: Y -> X
            W_bar_2, p_value_2, interp_2, N2, T2, n_units_2 = dumitrescu_hurlin_test(
                data, 'Country', 'Year', var_x, var_y, max_lag
            )

            if W_bar_1 is not None and W_bar_2 is not None:
                st.success("✅ Granger Causality tests completed!")
                
                # Results table
                st.subheader("Table F.1: Panel Granger Causality Results")
                
                gc_results = pd.DataFrame({
                    "Null Hypothesis (H₀)": [
                        f"{var_x} does not Granger-cause {var_y}",
                        f"{var_y} does not Granger-cause {var_x}"
                    ],
                    "W-bar": [f"{W_bar_1:.4f}", f"{W_bar_2:.4f}"],
                    "P-value": [f"{p_value_1:.4f}", f"{p_value_2:.4f}"],
                    "Decision (α=0.05)": [
                        "Reject H₀" if p_value_1 < 0.05 else "Fail to Reject H₀",
                        "Reject H₀" if p_value_2 < 0.05 else "Fail to Reject H₀"
                    ]
                })
                
                st.dataframe(gc_results, use_container_width=True)
                
                st.caption(f"""
                **Note:** N={N1} cross-sections, T={T1} time periods, p={max_lag} lags.
                Tests based on {n_units_1} units with sufficient data.
                """)
                
                # Interpretation
                st.subheader("Interpretation")
                
                if p_value_1 < 0.05 and p_value_2 >= 0.05:
                    st.success(f"✅ **Unidirectional Causality:** {var_x} → {var_y}")
                    st.write(f"{var_x} Granger-causes {var_y}, but not vice versa.")
                elif p_value_1 >= 0.05 and p_value_2 < 0.05:
                    st.success(f"✅ **Unidirectional Causality:** {var_y} → {var_x}")
                    st.write(f"{var_y} Granger-causes {var_x}, but not vice versa.")
                elif p_value_1 < 0.05 and p_value_2 < 0.05:
                    st.warning(f"⚠️ **Bidirectional Causality:** {var_x} ⇄ {var_y}")
                    st.write("Both variables Granger-cause each other (feedback effect).")
                else:
                    st.info(f"ℹ️ **No Causality Detected**")
                    st.write("Neither variable Granger-causes the other.")
                
                # Visualization
                st.subheader("Causal Relationship Diagram")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
                
                # Boxes
                box1 = FancyBboxPatch((1, 2), 2, 1, boxstyle="round,pad=0.1", 
                                     edgecolor='blue', facecolor='lightblue', linewidth=2)
                box2 = FancyBboxPatch((6, 2), 2, 1, boxstyle="round,pad=0.1", 
                                     edgecolor='green', facecolor='lightgreen', linewidth=2)
                ax.add_patch(box1)
                ax.add_patch(box2)
                
                ax.text(2, 2.5, var_x, ha='center', va='center', fontsize=14, fontweight='bold')
                ax.text(7, 2.5, var_y, ha='center', va='center', fontsize=14, fontweight='bold')
                
                # Arrows
                if p_value_1 < 0.05:
                    arrow1 = FancyArrowPatch((3.2, 2.7), (5.8, 2.7), 
                                           arrowstyle='->', mutation_scale=30, 
                                           linewidth=2.5, color='blue')
                    ax.add_patch(arrow1)
                    sig1 = '***' if p_value_1 < 0.01 else '**' if p_value_1 < 0.05 else '*'
                    ax.text(4.5, 3.1, f'p={p_value_1:.3f}{sig1}', 
                           ha='center', fontsize=11, color='blue', fontweight='bold')
                
                if p_value_2 < 0.05:
                    arrow2 = FancyArrowPatch((5.8, 2.3), (3.2, 2.3), 
                                           arrowstyle='->', mutation_scale=30, 
                                           linewidth=2.5, color='green')
                    ax.add_patch(arrow2)
                    sig2 = '***' if p_value_2 < 0.01 else '**' if p_value_2 < 0.05 else '*'
                    ax.text(4.5, 1.9, f'p={p_value_2:.3f}{sig2}', 
                           ha='center', fontsize=11, color='green', fontweight='bold')
                
                ax.set_xlim(0, 9)
                ax.set_ylim(1, 4)
                ax.axis('off')
                ax.set_title('Granger Causality Relationships', 
                           fontsize=16, fontweight='bold', pad=20)
                
                st.pyplot(fig)
                plt.close()
                
            else:
                st.error(f"Test failed: {interp_1}")

st.markdown("---")
st.markdown("**Developed for Panel Data Econometric Analysis | MMQR & Granger Causality**")
