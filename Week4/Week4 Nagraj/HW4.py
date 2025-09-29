# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:57:40 2025

@author: nagra
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the US Treasury returns data
try:
    # Replace backslashes with forward slashes
    ust_returns_df = pd.read_csv("C:/Users/rohil/iCloudDrive/UMN _ MSF/Curriculum/2) Fall A 2025/2) Econometrics/MSF_Econometric_Lab/Week4/Week4 Nagraj/USTreturns.csv")
    # Display the first 5 rows of the dataframe
    ust_returns_df.dropna(inplace=True)
    print(ust_returns_df.head())
except FileNotFoundError:
    print("Error: 'USTreturns.xlsx - USTreturns.csv' not found. Please check the file path.")

print("-" * 30) # Separator for clarity

# Load the Fama-French 3-factor model data
try:
    ff3_factors_df = pd.read_csv("C:/Users/rohil/iCloudDrive/UMN _ MSF/Curriculum/2) Fall A 2025/2) Econometrics/MSF_Econometric_Lab/Week4/Week4 Nagraj/FF3factors.csv")
    # Display the first 5 rows of the dataframe
    print(ff3_factors_df.head())
except FileNotFoundError:
    print("Error: 'FF3factors.xlsx - FF3factors.csv' not found. Please check the file path.")

# ff3 file
# Condition 1: Select all dates on or after July 1963
start_condition = (ff3_factors_df['YEAR'] > 1963) | \
                  ((ff3_factors_df['YEAR'] == 1963) & (ff3_factors_df['MONTH'] >= 7))

# Condition 2: Select all dates on or before December 2019
end_condition = (ff3_factors_df['YEAR'] < 2019) | \
                ((ff3_factors_df['YEAR'] == 2019) & (ff3_factors_df['MONTH'] <= 12))

filtered_ff3_returns = ff3_factors_df[start_condition & end_condition]
# --- Verification ---
# Print the first and last few rows to confirm the filter worked correctly
print("--- Start of Filtered ff3 Data (from July 1963) ---")
print(ff3_factors_df.head())
print("\n--- End of Filtered ff3 Data (to December 2019) ---")
print(ff3_factors_df.tail())

####USTreturns file
# --- Filtering Logic (same as before) ---

# Condition 1: Select all dates on or after July 1963
start_condition = (ust_returns_df['YEAR'] > 1963) | \
                  ((ust_returns_df['YEAR'] == 1963) & (ust_returns_df['MONTH'] >= 7))

# Condition 2: Select all dates on or before December 2019
end_condition = (ust_returns_df['YEAR'] < 2019) | \
                ((ust_returns_df['YEAR'] == 2019) & (ust_returns_df['MONTH'] <= 12))

# Apply both conditions to filter the DataFrame
filtered_ust_returns = ust_returns_df[start_condition & end_condition]
# --- Verification ---
# Print the first and last few rows to confirm the filter worked correctly
print("--- Start of Filtered USTreturns Data (from July 1963) ---")
print(filtered_ust_returns.head())
print("\n--- End of Filtered USTreturns Data (to December 2019) ---")
print(filtered_ust_returns.tail())

# 3. Calculate average and standard deviation of bond maturities
avg_maturities = filtered_ust_returns[[f'MAT {i+1}' for i in range(7)]].mean()
std_maturities = filtered_ust_returns[[f'MAT {i+1}' for i in range(7)]].std()

print("Average Maturities:\n", avg_maturities)
print("Standard Deviations of Maturities:\n", std_maturities)

# --- Step 2: Corrected Plotting Code ---

# Create a combined 'Year-Month' column for labeling
years_months = filtered_ust_returns.apply(lambda row: f"{int(row['YEAR'])}-{int(row['MONTH']):02d}", axis=1).values

# Define the maturities to label the x-axis
# NOTE: These are simplified labels for the 7 bonds in the data
maturities = np.array([1, 2, 5, 7, 10, 20, 30])

# Extract the 7 yield-to-maturity columns
ytm_columns = [f'YTM {i+1}' for i in range(7)]
yields = filtered_ust_returns[ytm_columns].values

# Create the meshgrid for the plot's X and Y axes
X, Y = np.meshgrid(maturities, np.arange(len(years_months)))

# Create the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface, using strides to make it less dense and easier to see
ax.plot_surface(X, Y, yields, cmap='viridis', rstride=6, cstride=1)

# --- Step 3: Customize the Plot for Readability ---
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Time')
ax.set_zlabel('Yield to Maturity (%)')
ax.set_title('Yield Curve Over Time (1963-2019)')

# Set y-axis ticks to show the date every 5 years for a cleaner look
tick_positions = np.arange(0, len(years_months), 12 * 5) # Every 60 months
ax.set_yticks(tick_positions)
ax.set_yticklabels(years_months[tick_positions], rotation=-15, va='baseline', ha='left')

# Adjust the viewing angle for a better perspective
ax.view_init(elev=25, azim=-120)

# 5. Calculate Annualized Excess Return, Standard Deviation, and Sharpe Ratio for M
# For Market Excess Return (column 3 in FF3factors)
monthly_avg_excess_return = filtered_ff3_returns['MER'].mean()
monthly_std_excess_return = filtered_ff3_returns['MER'].std()

# Annualize
annualized_avg_excess_return = monthly_avg_excess_return * 12
annualized_std_excess_return = monthly_std_excess_return * np.sqrt(12)

# Sharpe Ratio for the market
sharpe_ratio_market = annualized_avg_excess_return / annualized_std_excess_return

# For Bond Returns (from USTreturns)
bond_columns = [f'RET {i+1}' for i in range(7)]
monthly_avg_bond_returns = filtered_ust_returns[bond_columns].mean()
monthly_std_bond_returns = filtered_ust_returns[bond_columns].std()
# Annualize bond returns and standard deviations
annualized_avg_bond_returns = monthly_avg_bond_returns * 12
annualized_std_bond_returns = monthly_std_bond_returns * np.sqrt(12)
# Sharpe Ratios for each bond
bond_sharpe_ratios = annualized_avg_bond_returns / annualized_std_bond_returns

print("Annualized Average Market Excess Return:", annualized_avg_excess_return)
print("Annualized Standard Deviation (Market):", annualized_std_excess_return)
print("Sharpe Ratio for Market:", sharpe_ratio_market)
print("Sharpe Ratios for Bonds:\n", bond_sharpe_ratios)

# 6. Correlation Matrix between Market and Bond Returns
combined_data = filtered_ff3_returns[['MER']].copy()
for i in range(7):
 combined_data[f'RET {i+1}'] = filtered_ust_returns[f'RET {i+1}']
# Compute correlation matrix
correlation_matrix = combined_data.corr()
print("Correlation Matrix:\n", correlation_matrix)
# Plot heatmap of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Market and Bonds')
plt.show()


# 1. Select the correct columns for market and bond returns
market_returns = combined_data['MER']
bond_30year_returns = combined_data['RET 7'] # Corrected column name

# 2. Drop rows where EITHER of these two columns has a NaN value
valid_data = combined_data.dropna(subset=['MER', 'RET 7'])
market_returns_cleaned = valid_data['MER']
bond_30year_returns_cleaned = valid_data['RET 7']

# 3. Create the histogram using the CLEANED data
plt.figure(figsize=(8, 6))
plt.hist2d(market_returns_cleaned, bond_30year_returns_cleaned, bins=5, cmap='Blues')
plt.colorbar(label='Frequency Count')
plt.xlabel('Market Excess Return (MER)')
plt.ylabel('30-Year Bond Return')
plt.title('Bivariate Histogram of Market vs. 30-Year Bond Returns')

# --- Step 2: Create a clean dataset for this specific analysis ---
# This ensures we use the exact same data for the model and the plot
analysis_df = combined_data[['MER', 'RET 7']].dropna()

# Define the dependent (y) and independent (X) variables
y = analysis_df['RET 7']
X = sm.add_constant(analysis_df['MER']) # Add constant for the intercept


# --- Step 3: Run the OLS Regression ---
model = sm.OLS(y, X).fit()
print("--- Regression Summary: 30-Year Bond vs. Market ---")
print(model.summary())


# --- Step 4: Create the Final Plot ---
plt.figure(figsize=(10, 6))

# Scatter plot of the data used in the model
plt.scatter(X['MER'], y, alpha=0.5, label='Monthly Returns')

# Plot the regression line using the model's predictions
plt.plot(X['MER'], model.fittedvalues, color='red', linewidth=2, label='Fitted Line (OLS)')

# Add labels and title
plt.title('Market vs. 30-Year Bond Returns')
plt.xlabel('Market Excess Return (MER)')
plt.ylabel('30-Year Bond Return')
plt.grid(True)
plt.legend()

print(filtered_ff3_returns["MER"])