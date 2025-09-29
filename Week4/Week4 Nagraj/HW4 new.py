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

#Question 1- Both FF3factors.txt and USTreturns.txt are tab-delimited files, 
#with each column separated by tabs. The datasets consist of numerical values
# representing returns and yields across different time periods.


#############################Question 2#################################
try:
    
    ust_returns_df = pd.read_csv('C:/Users/nagra/Desktop/School/Fall A/Econometrics & Computational methods/USTreturns.csv')
    
    ust_returns_df.dropna(inplace=True)
    print(ust_returns_df.head())
except FileNotFoundError:
    print("Error: 'USTreturns.xlsx - USTreturns.csv' not found. Please check the file path.")

print("-" * 30) # Separator for clarity

# Load the Fama-French 3-factor model data
try:
    ff3_factors_df = pd.read_csv('C:/Users/nagra/Desktop/School/Fall A/Econometrics & Computational methods/FF3factors.csv')
    # Display the first 5 rows of the dataframe
    print(ff3_factors_df.head())
except FileNotFoundError:
    print("Error: 'FF3factors.xlsx - FF3factors.csv' not found. Please check the file path.")

#The returns in both files appear to be presented in decimal format 
#. No transformation is 
#necessary unless the data needs to be displayed in percentage 
#terms.
################################Question 3#################################
# ff3 file
# Condition 1: Select all dates on or after July 1963
start_condition = (ff3_factors_df['YEAR'] > 1963) | \
                  ((ff3_factors_df['YEAR'] == 1963) & (ff3_factors_df['MONTH'] >= 7))

# Condition 2: Select all dates on or before December 2019
end_condition = (ff3_factors_df['YEAR'] < 2019) | \
                ((ff3_factors_df['YEAR'] == 2019) & (ff3_factors_df['MONTH'] <= 12))

filtered_ff3_returns = ff3_factors_df[start_condition & end_condition]

print("--- Start of Filtered ff3 Data (from July 1963) ---")
print(ff3_factors_df.head())
print("\n--- End of Filtered ff3 Data (to December 2019) ---")
print(ff3_factors_df.tail())

####USTreturns file


# Condition 1: Select all dates on or after July 1963
start_condition = (ust_returns_df['YEAR'] > 1963) | \
                  ((ust_returns_df['YEAR'] == 1963) & (ust_returns_df['MONTH'] >= 7))

# Condition 2: Select all dates on or before December 2019
end_condition = (ust_returns_df['YEAR'] < 2019) | \
                ((ust_returns_df['YEAR'] == 2019) & (ust_returns_df['MONTH'] <= 12))


filtered_ust_returns = ust_returns_df[start_condition & end_condition]

print("--- Start of Filtered USTreturns Data (from July 1963) ---")
print(filtered_ust_returns.head())
print("\n--- End of Filtered USTreturns Data (to December 2019) ---")
print(filtered_ust_returns.tail())

# 3. Calculate average and standard deviation of bond maturities
avg_maturities = filtered_ust_returns[[f'MAT {i+1}' for i in range(7)]].mean()
std_maturities = filtered_ust_returns[[f'MAT {i+1}' for i in range(7)]].std()

print("Average Maturities:\n", avg_maturities)
print("Standard Deviations of Maturities:\n", std_maturities)


#Average maturities: The calculated average maturities across the 
#bonds range within a certain span.

#Standard deviation: Bonds with shorter maturities generally 
#exhibit lower standard deviations.

#Original maturity: The bonds most likely correspond to 
#securities that were initially issued with specific maturity terms.

###########################Question 4###################################


# Create a combined 'Year-Month' column for labeling
years_months = filtered_ust_returns.apply(lambda row: f"{int(row['YEAR'])}-{int(row['MONTH']):02d}", axis=1).values

# Define the maturities to label the x-axis

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


ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Time')
ax.set_zlabel('Yield to Maturity (%)')
ax.set_title('Yield Curve Over Time (1963-2019)')


tick_positions = np.arange(0, len(years_months), 12 * 5) 
ax.set_yticks(tick_positions)
ax.set_yticklabels(years_months[tick_positions], rotation=-15, va='baseline', ha='left')


ax.view_init(elev=25, azim=-120)

#Interest rates: A clear trend shows that rates were significantly 
#higher in the early 1980s and have generally declined since then.

#Yield curve: This pattern is visible in the “surfboard” shape of 
#the yield curve plot.

#Recessions: Short-term yields typically experience sharper 
#declines than long-term yields during recessions.

#Implication: These movements suggest that economic downturns are 
#often accompanied by a flattening of the yield curve.
###################################Question 5############################

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


#########################Question 6############################
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
bond_30year_returns = combined_data['RET 7']

# 2. Drop rows where EITHER of these two columns has a NaN value
valid_data = combined_data.dropna(subset=['MER', 'RET 7'])
market_returns_cleaned = valid_data['MER']
bond_30year_returns_cleaned = valid_data['RET 7']

#Maturity and correlation: Bonds with longer maturities generally 
#show stronger correlations with each other.

#Correlation among bonds: Securities with similar maturities, 
#such as the 20-year and 30-year bonds, are more closely 
#correlated, whereas shorter-term bonds (e.g., 2-year, 5-year) 
#display weaker correlations with long-term bonds.

# ################################Question 7#######################
plt.figure(figsize=(8, 6))
plt.hist2d(market_returns_cleaned, bond_30year_returns_cleaned, bins=5, cmap='Blues')
plt.colorbar(label='Frequency Count')
plt.xlabel('Market Excess Return (MER)')
plt.ylabel('30-Year Bond Return')
plt.title('Bivariate Histogram of Market vs. 30-Year Bond Returns')

#Market vs. Bond performance: The 30-year bond often moves 
#inversely to the market—when market returns are lower, bond 
#returns are usually higher, and vice versa.

#Natural hedge: The 30-year bond can serve as a natural hedge for 
#equities, since its performance typically strengthens during 
#periods of weak market returns.

###################################Question 8######################

analysis_df = combined_data[['MER', 'RET 7']].dropna()


y = analysis_df['RET 7']
X = sm.add_constant(analysis_df['MER']) # Add constant for the intercept

#Regression result: The analysis indicates a negative correlation 
#between the 30-year bond return and market excess return.

#Slope interpretation: The slope of the regression line is 
#slightly negative, suggesting that as market returns rise, 
#the 30-year bond return tends to decline.

#Visualization: The regression line highlights this inverse 
#relationship between the 30-year bond and market performance

###############################Question 9###################################
model = sm.OLS(y, X).fit()
print("--- Regression Summary: 30-Year Bond vs. Market ---")
print(model.summary())


plt.figure(figsize=(10, 6))


plt.scatter(X['MER'], y, alpha=0.5, label='Monthly Returns')

# Plot the regression line using the model's predictions
plt.plot(X['MER'], model.fittedvalues, color='red', linewidth=2, label='Fitted Line (OLS)')

# Add labels and title
plt.title('Market vs. 30-Year Bond Returns')
plt.xlabel('Market Excess Return (MER)')
plt.ylabel('30-Year Bond Return')
plt.grid(True)
plt.legend()