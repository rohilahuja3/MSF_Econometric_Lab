# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 17:55:18 2025

@author: nagra
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ============================================================================
# QUESTION 1 & 2: Data Loading and Excess Returns
# ============================================================================

try:
    # --- 1. Load the data ---
    # berkshire_df = pd.read_csv("C:/Users/rohil/iCloudDrive/UMN _ MSF/Curriculum/2) Fall A 2025/2) Econometrics/MSF_Econometric_Lab/Week5/berkshire.txt", delimiter="\t", header=None)
    berkshire_df = pd.read_csv('C:/Users/rohil/iCloudDrive/UMN _ MSF/Curriculum/2) Fall A 2025/2) Econometrics/MSF_Econometric_Lab/Week5/berkshire.txt', sep='\s+', header=None, names=['Date', 'Berkshire_Return'])
    # berkshire_df = pd.read_csv('berkshire.txt', sep='\s+', header=None, names=['Date', 'Berkshire_Return'])
    ff_df = pd.read_csv('C:/Users/rohil/iCloudDrive/UMN _ MSF/Curriculum/2) Fall A 2025/2) Econometrics/MSF_Econometric_Lab/Week5/FF3factors.txt', sep='\s+', header=None, names=['Year', 'Month', 'Mkt_RF', 'SMB', 'HML', 'RF'])
    print(berkshire_df)
    
    # --- 2. Prepare common key for merging ---
    berkshire_df['YYYYMM'] = berkshire_df['Date'].astype(str).str[:6]
    ff_df['YYYYMM'] = ff_df['Year'].astype(str) + ff_df['Month'].astype(str).str.zfill(2)
    
    # Merge datasets
    merged_df = pd.merge(berkshire_df, ff_df, on='YYYYMM', how='inner')
    
    # CRITICAL FIX: Convert FF factors from percentage to decimal
    merged_df['Mkt_RF'] = merged_df['Mkt_RF'] / 100
    merged_df['SMB'] = merged_df['SMB'] / 100
    merged_df['HML'] = merged_df['HML'] / 100
    merged_df['RF'] = merged_df['RF'] / 100
    
    # Calculate excess return for Question 2
    merged_df['Excess_Return'] = merged_df['Berkshire_Return'] - merged_df['RF']
    excess_returns = merged_df['Excess_Return']

    # --- 3. Calculate sample statistics ---
    r_bar = excess_returns.mean()
    sigma = excess_returns.std(ddof=1)
    n = len(excess_returns)
    se = sigma / np.sqrt(n)
    df = n - 1

    # --- 4. T-test for H₀: E(r) = 0 ---
    t_stat_0 = r_bar / se
    p_value_0 = stats.t.sf(np.abs(t_stat_0), df) * 2
    ci_95 = stats.t.interval(0.95, df, loc=r_bar, scale=se)
    is_significant_0 = p_value_0 < 0.01
    
    # --- 5. Visualization for H₀: E(r) = 0 ---
    alpha_0 = 0.01
    x_0 = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 500)
    plt.figure(figsize=(10, 6))
    plt.plot(x_0, stats.t.pdf(x_0, df), 'b-', label=f't-distribution (df={df})')
    crit_val_lower_0 = stats.t.ppf(alpha_0 / 2, df)
    crit_val_upper_0 = stats.t.ppf(1 - alpha_0 / 2, df)
    x_fill_lower_0 = np.linspace(stats.t.ppf(0.0001, df), crit_val_lower_0, 100)
    x_fill_upper_0 = np.linspace(crit_val_upper_0, stats.t.ppf(0.9999, df), 100)
    plt.fill_between(x_fill_lower_0, stats.t.pdf(x_fill_lower_0, df), color='red', alpha=0.5, label=f'{alpha_0*100}% Rejection Region')
    plt.fill_between(x_fill_upper_0, stats.t.pdf(x_fill_upper_0, df), color='red', alpha=0.5)
    plt.axvline(crit_val_lower_0, color='orange', linestyle='--', label=f'Critical Value ({crit_val_lower_0:.2f})')
    plt.axvline(crit_val_upper_0, color='orange', linestyle='--')
    plt.axvline(t_stat_0, color='green', linestyle='-', linewidth=2, label=f'Calculated t-statistic ({t_stat_0:.2f})')
    plt.title('t-test for H₀: E(r) = 0%')
    plt.xlabel('t-value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 6. T-test for H₀: E(r) = 1% ---
    null_hypothesis_1 = 0.01
    t_stat_1 = (r_bar - null_hypothesis_1) / se
    p_value_1 = stats.t.sf(np.abs(t_stat_1), df) * 2
    is_significant_1 = p_value_1 < 0.01

    # --- 7. Visualization for H₀: E(r) = 1% ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_0, stats.t.pdf(x_0, df), 'b-', label=f't-distribution (df={df})')
    plt.fill_between(x_fill_lower_0, stats.t.pdf(x_fill_lower_0, df), color='red', alpha=0.5, label=f'{alpha_0*100}% Rejection Region')
    plt.fill_between(x_fill_upper_0, stats.t.pdf(x_fill_upper_0, df), color='red', alpha=0.5)
    plt.axvline(crit_val_lower_0, color='orange', linestyle='--', label=f'Critical Value')
    plt.axvline(crit_val_upper_0, color='orange', linestyle='--')
    plt.axvline(t_stat_1, color='green', linestyle='-', linewidth=2, label=f'Calculated t-statistic ({t_stat_1:.2f})')
    plt.title(f't-test for H₀: E(r) = {null_hypothesis_1:.0%}')
    plt.xlabel('t-value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 8. Print Summary ---
    print("="*70)
    print("QUESTION 2: T-Test Analysis Summary")
    print("="*70)
    print("\nSample Statistics:")
    print(f"  Sample Mean (r_bar):     {r_bar:.6f} ({r_bar:.4%})")
    print(f"  Standard Deviation (σ):  {sigma:.6f}")
    print(f"  Sample Size (n):         {n}")
    print(f"  Standard Error (SE):     {se:.6f}")
    print(f"  Degrees of Freedom (df): {df}")

    print("\nTest 1: H₀: E(r) = 0%")
    print(f"  t-Statistic:             {t_stat_0:.4f}")
    print(f"  P-Value:                 {p_value_0:.6f}")
    print(f"  Significant at 1% level? {is_significant_0}")
    
    print("\nTest 2: H₀: E(r) = 1%")
    print(f"  t-Statistic:             {t_stat_1:.4f}")
    print(f"  P-Value:                 {p_value_1:.6f}")
    print(f"  Significant at 1% level? {is_significant_1}")

    print("\nConfidence Interval:")
    print(f"  95% CI for Mean:         ({ci_95[0]:.6f}, {ci_95[1]:.6f})")
    
except FileNotFoundError:
    print("ERROR: Make sure 'berkshire.txt' and 'FF3factors.txt' are in the same directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# ============================================================================
# QUESTION 3: Factor Model Regressions (Matrix Estimation)
# ============================================================================

print("\n" + "="*70)
print("QUESTION 3: Factor Model Regression Results")
print("="*70)

# Calculate Berkshire's excess return (dependent variable)
merged_df['BRK_Excess_Return'] = merged_df['Berkshire_Return'] - merged_df['RF']
y = merged_df['BRK_Excess_Return']
y_vec = y.values

# --- Model 1: CAPM ---
print("\nRunning CAPM (1-Factor Model)...")
X1 = merged_df[['Mkt_RF']]
X1_with_const = np.c_[np.ones(len(X1)), X1.values]
n1, k1 = X1_with_const.shape

# Matrix Estimation
XtX1 = X1_with_const.T @ X1_with_const
XtX1_inv = np.linalg.inv(XtX1)
Xty1 = X1_with_const.T @ y_vec
coeffs1 = XtX1_inv @ Xty1
residuals1 = y_vec - (X1_with_const @ coeffs1)

# Standard Errors
resid_var1 = (residuals1.T @ residuals1) / (n1 - k1)
var_beta1 = resid_var1 * XtX1_inv
se_std1 = np.sqrt(np.diag(var_beta1))
t_stats_std1 = coeffs1 / se_std1
p_vals_std1 = stats.t.sf(np.abs(t_stats_std1), df=n1 - k1) * 2

# Robust Standard Errors
S1 = np.diag(residuals1**2)
meat1 = X1_with_const.T @ S1 @ X1_with_const
var_beta_rob1 = XtX1_inv @ meat1 @ XtX1_inv
se_rob1 = np.sqrt(np.diag(var_beta_rob1))
t_stats_rob1 = coeffs1 / se_rob1
p_vals_rob1 = stats.t.sf(np.abs(t_stats_rob1), df=n1 - k1) * 2

# --- Model 2: 2-Factor Model ---
print("Running 2-Factor Model...")
X2 = merged_df[['Mkt_RF', 'SMB']]
X2_with_const = np.c_[np.ones(len(X2)), X2.values]
n2, k2 = X2_with_const.shape

# Matrix Estimation
XtX2 = X2_with_const.T @ X2_with_const
XtX2_inv = np.linalg.inv(XtX2)
Xty2 = X2_with_const.T @ y_vec
coeffs2 = XtX2_inv @ Xty2
residuals2 = y_vec - (X2_with_const @ coeffs2)

# Standard Errors
resid_var2 = (residuals2.T @ residuals2) / (n2 - k2)
var_beta2 = resid_var2 * XtX2_inv
se_std2 = np.sqrt(np.diag(var_beta2))
t_stats_std2 = coeffs2 / se_std2
p_vals_std2 = stats.t.sf(np.abs(t_stats_std2), df=n2 - k2) * 2

# Robust Standard Errors
S2 = np.diag(residuals2**2)
meat2 = X2_with_const.T @ S2 @ X2_with_const
var_beta_rob2 = XtX2_inv @ meat2 @ XtX2_inv
se_rob2 = np.sqrt(np.diag(var_beta_rob2))
t_stats_rob2 = coeffs2 / se_rob2
p_vals_rob2 = stats.t.sf(np.abs(t_stats_rob2), df=n2 - k2) * 2

# --- Model 3: Fama-French 3-Factor Model ---
print("Running Fama-French 3-Factor Model...")
X3 = merged_df[['Mkt_RF', 'SMB', 'HML']]
X3_with_const = np.c_[np.ones(len(X3)), X3.values]
n3, k3 = X3_with_const.shape

# Matrix Estimation
XtX3 = X3_with_const.T @ X3_with_const
XtX3_inv = np.linalg.inv(XtX3)
Xty3 = X3_with_const.T @ y_vec
coeffs3 = XtX3_inv @ Xty3
residuals3 = y_vec - (X3_with_const @ coeffs3)

# Standard Errors
resid_var3 = (residuals3.T @ residuals3) / (n3 - k3)
var_beta3 = resid_var3 * XtX3_inv
se_std3 = np.sqrt(np.diag(var_beta3))
t_stats_std3 = coeffs3 / se_std3
p_vals_std3 = stats.t.sf(np.abs(t_stats_std3), df=n3 - k3) * 2

# Robust Standard Errors
S3 = np.diag(residuals3**2)
meat3 = X3_with_const.T @ S3 @ X3_with_const
var_beta_rob3 = XtX3_inv @ meat3 @ XtX3_inv
se_rob3 = np.sqrt(np.diag(var_beta_rob3))
t_stats_rob3 = coeffs3 / se_rob3
p_vals_rob3 = stats.t.sf(np.abs(t_stats_rob3), df=n3 - k3) * 2

# --- Create Output Tables ---
get_stars = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''

# Table 1: Standard Errors
capm_std_str = [f"{coeffs1[i]:.4f}{get_stars(p_vals_std1[i])}\n({se_std1[i]:.4f})" for i in range(k1)]
f2_std_str = [f"{coeffs2[i]:.4f}{get_stars(p_vals_std2[i])}\n({se_std2[i]:.4f})" for i in range(k2)]
f3_std_str = [f"{coeffs3[i]:.4f}{get_stars(p_vals_std3[i])}\n({se_std3[i]:.4f})" for i in range(k3)]

results_df_standard = pd.DataFrame({
    'CAPM': capm_std_str + ['-'] * 2,
    '2-Factor': f2_std_str + ['-'] * 1,
    '3-Factor': f3_std_str
}, index=['Alpha', 'Mkt-RF', 'SMB', 'HML'])

# Table 2: Robust Standard Errors
capm_rob_str = [f"{coeffs1[i]:.4f}{get_stars(p_vals_rob1[i])}\n({se_rob1[i]:.4f})" for i in range(k1)]
f2_rob_str = [f"{coeffs2[i]:.4f}{get_stars(p_vals_rob2[i])}\n({se_rob2[i]:.4f})" for i in range(k2)]
f3_rob_str = [f"{coeffs3[i]:.4f}{get_stars(p_vals_rob3[i])}\n({se_rob3[i]:.4f})" for i in range(k3)]

results_df_robust = pd.DataFrame({
    'CAPM': capm_rob_str + ['-'] * 2,
    '2-Factor': f2_rob_str + ['-'] * 1,
    '3-Factor': f3_rob_str
}, index=['Alpha', 'Mkt-RF', 'SMB', 'HML'])

print("\n--- Table 1: Standard Errors (OLS) ---")
print(results_df_standard)
print("\n--- Table 2: Heteroskedasticity-Robust Standard Errors (White) ---")
print(results_df_robust)
print("\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.10")


# ============================================================================
# QUESTION 4: R-squared and Nested F-tests
# ============================================================================

print("\n" + "="*70)
print("QUESTION 4: Model Goodness-of-Fit and Comparison")
print("="*70)

# Total Sum of Squares
tss = np.sum((y - y.mean())**2)

# Sum of Squared Residuals
ssr1 = np.sum(residuals1**2)
ssr2 = np.sum(residuals2**2)
ssr3 = np.sum(residuals3**2)

# R-squared
r_squared1 = 1 - (ssr1 / tss)
r_squared2 = 1 - (ssr2 / tss)
r_squared3 = 1 - (ssr3 / tss)

print("\nR-squared (Percentage of Variance Explained):")
print(f"  Model 1 (CAPM):     {r_squared1:.2%}")
print(f"  Model 2 (2-Factor): {r_squared2:.2%}")
print(f"  Model 3 (FF3):      {r_squared3:.2%}")
print(f"\n  Improvement adding SMB:  {r_squared2 - r_squared1:+.2%}")
print(f"  Improvement adding HML:  {r_squared3 - r_squared2:+.2%}")

# Nested F-tests
print("\nNested F-Tests:")

# Test A: Model 2 vs Model 1
q_A = 1
f_stat_A = ((ssr1 - ssr2) / q_A) / (ssr2 / (n2 - k2))
p_value_A = stats.f.sf(f_stat_A, dfn=q_A, dfd=(n2 - k2))

print(f"\n  Test A: Model 2 vs Model 1 (adding SMB)")
print(f"    F-statistic: {f_stat_A:.4f}")
print(f"    P-value:     {p_value_A:.6f}")
print(f"    Conclusion:  {'Significant' if p_value_A < 0.05 else 'Not significant'}")

# Test B: Model 3 vs Model 2
q_B = 1
f_stat_B = ((ssr2 - ssr3) / q_B) / (ssr3 / (n3 - k3))
p_value_B = stats.f.sf(f_stat_B, dfn=q_B, dfd=(n3 - k3))

print(f"\n  Test B: Model 3 vs Model 2 (adding HML)")
print(f"    F-statistic: {f_stat_B:.4f}")
print(f"    P-value:     {p_value_B:.6f}")
print(f"    Conclusion:  {'Significant' if p_value_B < 0.05 else 'Not significant'}")


# ============================================================================
# QUESTION 5: Variance Decomposition
# ============================================================================

print("\n" + "="*70)
print("QUESTION 5: Variance Decomposition (FF3 Model)")
print("="*70)

# Covariance matrix of factors
factors = merged_df[['Mkt_RF', 'SMB', 'HML']]
cov_matrix = factors.cov().values

# Extract betas (exclude intercept)
betas = coeffs3[1:]

# Total variance explained
total_var_explained = betas.T @ cov_matrix @ betas

print(f"\nTotal Variance Explained by Factors: {total_var_explained:.6f}")

# Decompose variance
M = cov_matrix @ betas
contributions = betas * M
pct_contribution = (contributions / total_var_explained) * 100

# Results table
decomp_df = pd.DataFrame({
    'Beta': betas,
    'Var Contribution': contributions,
    '% of Variance': pct_contribution
}, index=['Mkt-RF', 'SMB', 'HML'])

print("\n" + decomp_df.to_string())

biggest_factor = decomp_df['% of Variance'].idxmax()
print(f"\nThe '{biggest_factor}' factor has the biggest impact ({decomp_df.loc[biggest_factor, '% of Variance']:.2f}%)")


# ============================================================================
# QUESTION 6: Statsmodels Re-estimation and F-tests
# ============================================================================

print("\n" + "="*70)
print("QUESTION 6: Statsmodels Re-estimation and Hypothesis Tests")
print("="*70)

# Re-estimate using statsmodels
X1_const = sm.add_constant(merged_df[['Mkt_RF']])
X2_const = sm.add_constant(merged_df[['Mkt_RF', 'SMB']])
X3_const = sm.add_constant(merged_df[['Mkt_RF', 'SMB', 'HML']])

model1_sm = sm.OLS(y, X1_const).fit()
model2_sm = sm.OLS(y, X2_const).fit()
model3_sm = sm.OLS(y, X3_const).fit()

# Overall F-tests
print("\nOverall Model Significance:")
for name, model in [('CAPM', model1_sm), ('2-Factor', model2_sm), ('FF3', model3_sm)]:
    print(f"  {name:10s}: F={model.fvalue:7.2f}, p={model.f_pvalue:.4f}")

# Joint hypothesis tests
print("\nJoint and Individual Hypothesis Tests:")

joint_test = model3_sm.f_test('SMB = 0, HML = 0')
print(f"  H0: SMB=0 AND HML=0:  F={joint_test.fvalue:.2f}, p={joint_test.pvalue:.4f}")

hml_test = model3_sm.f_test('HML = 0')
print(f"  H0: HML=0:            F={hml_test.fvalue:.2f}, p={hml_test.pvalue:.4f}")

smb_test = model3_sm.f_test('SMB = 0')
print(f"  H0: SMB=0:            F={smb_test.fvalue:.2f}, p={smb_test.pvalue:.4f}")

print(f"\nConclusion: FF3 model {'adds' if joint_test.pvalue < 0.05 else 'does not add'} significant value over CAPM")


# ============================================================================
# QUESTION 7: Benchmark Portfolio and Time Series Analysis
# ============================================================================

print("\n" + "="*70)
print("QUESTION 7: Benchmark Portfolio Analysis")
print("="*70)

# CRITICAL FIX: Use coefficients from matrix estimation (already in correct units)
beta_mkt = coeffs3[1]  # From our matrix estimation
beta_smb = coeffs3[2]
beta_hml = coeffs3[3]

print(f"\nBeta coefficients being used:")
print(f"  Mkt-RF: {beta_mkt:.4f}")
print(f"  SMB:    {beta_smb:.4f}")
print(f"  HML:    {beta_hml:.4f}")

# Calculate benchmark returns
merged_df['Benchmark_Return'] = ((1 - beta_mkt) * merged_df['RF'] + 
                                  beta_mkt * merged_df['Mkt_RF'] + 
                                  beta_smb * merged_df['SMB'] + 
                                  beta_hml * merged_df['HML'])

# Market total return
merged_df['Market_Total_Return'] = merged_df['RF'] + merged_df['Mkt_RF']

# Cumulative values
merged_df['FV_BRK'] = (1 + merged_df['Berkshire_Return']).cumprod()
merged_df['FV_Benchmark'] = (1 + merged_df['Benchmark_Return']).cumprod()
merged_df['FV_Market'] = (1 + merged_df['Market_Total_Return']).cumprod()

# Alpha
merged_df['alpha_t'] = merged_df['Berkshire_Return'] - merged_df['Benchmark_Return']

# Create date column
merged_df['Date_dt'] = pd.to_datetime(merged_df['Date'], format='%Y%m%d')

# Visualization
plt.figure(figsize=(14, 8))
plt.plot(merged_df['Date_dt'], np.log(merged_df['FV_BRK']), 
         label='Berkshire Hathaway', color='navy', linewidth=2)
plt.plot(merged_df['Date_dt'], np.log(merged_df['FV_Benchmark']), 
         label='Benchmark Portfolio', color='red', linestyle='--', linewidth=2)
plt.plot(merged_df['Date_dt'], np.log(merged_df['FV_Market']), 
         label='Market Portfolio', color='gray', linestyle=':', linewidth=2)
plt.axvline(pd.Timestamp('2000-01-01'), color='black', linestyle=':', alpha=0.5, label='Year 2000')
plt.title('Growth of $1 Investment (Log Scale)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('ln(Cumulative Value of $1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Period analysis
pre_2000_df = merged_df[merged_df['Date_dt'] < '2000-01-01'].copy()
post_2000_df = merged_df[merged_df['Date_dt'] >= '2000-01-01'].copy()

# Recalculate cumulative values within each period
pre_2000_df['FV_BRK'] = (1 + pre_2000_df['Berkshire_Return']).cumprod()
pre_2000_df['FV_Benchmark'] = (1 + pre_2000_df['Benchmark_Return']).cumprod()
pre_2000_df['FV_Market'] = (1 + pre_2000_df['Market_Total_Return']).cumprod()

post_2000_df['FV_BRK'] = (1 + post_2000_df['Berkshire_Return']).cumprod()
post_2000_df['FV_Benchmark'] = (1 + post_2000_df['Benchmark_Return']).cumprod()
post_2000_df['FV_Market'] = (1 + post_2000_df['Market_Total_Return']).cumprod()

# Summary statistics
summary_data = {
    'Metric': ['Berkshire FV', 'Benchmark FV', 'Market FV', 'Mean Alpha (monthly)', 'Cumulative Alpha'],
    'Pre-2000': [
        f"${pre_2000_df['FV_BRK'].iloc[-1]:.2f}",
        f"${pre_2000_df['FV_Benchmark'].iloc[-1]:.2f}",
        f"${pre_2000_df['FV_Market'].iloc[-1]:.2f}",
        f"{pre_2000_df['alpha_t'].mean():.4%}",
        f"${pre_2000_df['FV_BRK'].iloc[-1] - pre_2000_df['FV_Benchmark'].iloc[-1]:.2f}"
    ],
    'Post-2000': [
        f"${post_2000_df['FV_BRK'].iloc[-1]:.2f}",
        f"${post_2000_df['FV_Benchmark'].iloc[-1]:.2f}",
        f"${post_2000_df['FV_Market'].iloc[-1]:.2f}",
        f"{post_2000_df['alpha_t'].mean():.4%}",
        f"${post_2000_df['FV_BRK'].iloc[-1] - post_2000_df['FV_Benchmark'].iloc[-1]:.2f}"
    ],
    'Full Period': [
        f"${merged_df['FV_BRK'].iloc[-1]:.2f}",
        f"${merged_df['FV_Benchmark'].iloc[-1]:.2f}",
        f"${merged_df['FV_Market'].iloc[-1]:.2f}",
        f"{merged_df['alpha_t'].mean():.4%}",
        f"${merged_df['FV_BRK'].iloc[-1] - merged_df['FV_Benchmark'].iloc[-1]:.2f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n--- Period Analysis Summary ---")
print(summary_df.to_string(index=False))

print("\n--- Interpretation ---")
print("\nA. Performance Comparison:")
print("   Pre-2000: Berkshire dramatically outperformed both the benchmark and market.")
print("   Post-2000: Performance moderated significantly, tracking closer to benchmarks.")

print("\nB. Alpha Over Time:")
print("   Berkshire's alpha has clearly declined from pre-2000 to post-2000 period,")
print("   indicating reduced outperformance relative to factor exposures.")

print("\nC. Possible Reasons:")
print("   1. Law of Large Numbers - Harder to find impactful opportunities as size grew")
print("   2. Increased Market Efficiency - Fewer undervalued assets available")
print("   3. Competition - More investors using value-investing principles")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)