# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 15:17:08 2025

@author: nagra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

df_FF3 = pd.read_csv("C:/Users/rohil/iCloudDrive/UMN _ MSF/Curriculum/2) Fall A 2025/2) Econometrics/MSF_Econometric_Lab/Week4/FF3factors.txt", delimiter="\t", header=None)
df_UST = pd.read_csv("C:/Users/rohil/iCloudDrive/UMN _ MSF/Curriculum/2) Fall A 2025/2) Econometrics/MSF_Econometric_Lab/Week4/USTreturns.txt", delimiter="\t", header=None)

df_FF3.columns = ["Year", "Month", "Market Excess (Mkt Return - r_f)",  "SMB (ME1-ME5)", "HML (BM5--value -BM1--growth)","Risk-free rate r_f"]

df_UST.columns = (["Year","Month"] +[f"ret_{i}" for i in range(1,8)] + [f"ytm_{i}" for i in range(1,8)] +[f"mat_{i}" for i in range(1,8)])

# ret = Price previous month/Price this month -1
# YTM is the yield of the bond until maturity
# Mat is the maturity left of the bond

# Covert the pandas df to numpy

for i in range(len(df_FF3.columns)):
    df_FF3[df_FF3.columns[i]] = pd.to_numeric(df_FF3[df_FF3.columns[i]])

for i in range(len(df_UST.columns)):
    df_UST[df_UST.columns[i]] = pd.to_numeric(df_UST[df_UST.columns[i]])

df_FF3.iloc[:, 2:]=df_FF3.iloc[:, 2:]/100
df_UST.iloc[:, 9:16]=df_UST.iloc[:, 9:16]/100

print(df_FF3.head())
print(df_UST.head())

df_UST = df_UST[(df_UST["Year"]>=1963) & (df_UST["Month"]>=7)]
df_FF3 = df_FF3[((df_FF3["Year"]>=1963) & (df_FF3["Month"]>=7)) & (df_FF3["Year"]<=2019) & (df_FF3["Month"]<=12)]

df_UST.reset_index(inplace=True)
df_UST.drop(columns="index", inplace=True)
df_FF3.reset_index(inplace=True)
df_FF3.drop(columns="index", inplace=True)

print(df_FF3.head())
print(df_UST.head())

# --- Step 2: Corrected Plotting Code ---

# Create a combined 'Year-Month' column for labeling
years_months = df_UST.apply(lambda row: f"{int(row['Year'])}-{int(row['Month']):02d}", axis=1).values

# Define the maturities to label the x-axis
# NOTE: These are simplified labels for the 7 bonds in the data
maturities = np.array([1, 2, 5, 7, 10, 20, 30])

# Extract the 7 yield-to-maturity columns
ytm_columns = [f'ytm_{i+1}' for i in range(7)]
yields = df_UST[ytm_columns].values

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
########################################################################
# 5. Calculate Annualized Excess Return, Standard Deviation, and Sharpe Ratio for M
# For Market Excess Return (column 3 in FF3factors)
monthly_avg_excess_return = df_FF3['Market Excess (Mkt Return - r_f)'].mean()
monthly_std_excess_return = df_FF3['Market Excess (Mkt Return - r_f)'].std()

# Annualize
annualized_avg_excess_return = monthly_avg_excess_return * 12
annualized_std_excess_return = monthly_std_excess_return * np.sqrt(12)

# Sharpe Ratio for the market
sharpe_ratio_market = annualized_avg_excess_return / annualized_std_excess_return

# For Bond Returns (from USTreturns)
bond_columns = [f'ret_{i+1}' for i in range(7)]
monthly_avg_bond_returns = df_UST[bond_columns].mean()
monthly_std_bond_returns = df_UST[bond_columns].std()
# Annualize bond returns and standard deviations
annualized_avg_bond_returns = monthly_avg_bond_returns * 12
annualized_std_bond_returns = monthly_std_bond_returns * np.sqrt(12)
# Sharpe Ratios for each bond
bond_sharpe_ratios = annualized_avg_bond_returns / annualized_std_bond_returns

print("Annualized Average Market Excess Return:", annualized_avg_excess_return)
print("Annualized Standard Deviation (Market):", annualized_std_excess_return)
print("Sharpe Ratio for Market:", sharpe_ratio_market)
print("Sharpe Ratios for Bonds:\n", bond_sharpe_ratios)