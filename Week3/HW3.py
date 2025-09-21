# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 13:08:56 2025

@author: nagra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the data, skipping the first 15 rows and reading only the 1174 rows of the first table
df = pd.read_csv('comp_file.csv', skiprows=15, nrows=1174)

# Rename the first column to 'Date' for clarity
df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Replace missing value indicators with NumPy's Not a Number
df.replace(-99.99, np.nan, inplace=True)
df.replace(-999, np.nan, inplace=True)

# Convert the 'Date' column from text to a numeric type
df['Date'] = pd.to_numeric(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df['Date'] = df['Date'].astype(int)
df = df[df['Date'] >= 196307]

################################Question 2###################################

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['SMALL HiBM'], label='Small Value (Small HiBM)', color='blue')
plt.plot(df['Date'], df['BIG LoBM'], label='Big Growth (Big LoBM)', color='red')

# Add descriptive labels and title
plt.title('Monthly Returns: Small Value vs. Big Growth Firms')
plt.xlabel('Date')
plt.ylabel('Monthly Return (%)')
plt.legend()
plt.grid(True)

# ###############################Question 3######################################

# Create a new, smaller DataFrame that only has the 25 portfolio return columns
portfolios_df = df.iloc[:, 1:]
mean_returns = portfolios_df.mean()
variances = np.diag(portfolios_df.cov())


# Reshape our lists of means and variances into 5x5 grids for the 3D plot
mean_grid = np.array(mean_returns).reshape(5, 5)
variance_grid = np.array(variances).reshape(5, 5)

# Create X and Y coordinates for the bars (from 0 to 4 for Size and Value)
x = np.arange(5)
y = np.arange(5)
X, Y = np.meshgrid(x, y)

# Flatten the coordinates and the data to match what the plotting function needs
x_pos, y_pos = X.flatten(), Y.flatten()
z_pos = np.zeros_like(x_pos) # This makes the bars start from a height of 0
dz_means = mean_grid.flatten()
dz_variances = variance_grid.flatten()

# -- Chart 1: Mean Returns --
fig_means = plt.figure(figsize=(12, 8))
ax_means = fig_means.add_subplot(111, projection='3d')
ax_means.bar3d(x_pos, y_pos, z_pos, dx=0.8, dy=0.8, dz=dz_means, color='dodgerblue')

# Add labels to the chart
ax_means.set_title('Mean Monthly Returns of 25 Portfolios')
ax_means.set_xlabel('Size (Small to Big)')
ax_means.set_ylabel('Value')
ax_means.set_zlabel('Mean Return (%)')
ax_means.set_xticks(x)
ax_means.set_yticks(y)
ax_means.set_xticklabels(['Small', 'ME2', 'ME3', 'ME4', 'Big'])
ax_means.set_yticklabels(['LoBM', 'BM2', 'BM3', 'BM4', 'HiBM'])
plt.savefig('mean_returns_3d.png') # Save the plot

# -- Chart 2: Variances --
fig_vars = plt.figure(figsize=(12, 8))
ax_vars = fig_vars.add_subplot(111, projection='3d')
ax_vars.bar3d(x_pos, y_pos, z_pos, dx=0.8, dy=0.8, dz=dz_variances, color='orangered')

# Add labels to the chart
ax_vars.set_title('Variance of Monthly Returns of 25 Portfolios')
ax_vars.set_xlabel('Size (Small to Big)')
ax_vars.set_ylabel('Value')
ax_vars.set_zlabel('Variance')
ax_vars.set_xticks(x)
ax_vars.set_yticks(y)
ax_vars.set_xticklabels(['Small', 'ME2', 'ME3', 'ME4', 'Big'])
ax_vars.set_yticklabels(['LoBM', 'BM2', 'BM3', 'BM4', 'HiBM'])

#Highest and Lowest Returns: The Small Value portfolio earned the 
#highest average monthly return. The Big Growth portfolio earned 
#the lowest. This highlights the historical "size" and "value" premiums.

#Highest Variance: The Small Value portfolio  also had the highest 
#variance. This is a classic risk-return tradeoff: the portfolio with the 
#highest average return was also the most volatile.

#Most Covariance: The two portfolios that covary the most are adjacent portfolios, 
#specifically ME2 BM2 and 
#ME2 BM3. This makes sense, as they are very similar in size and only one 
#step apart in their value (book-to-market) characteristic.

# Question 4#################################################################

# Isolate the two portfolios of interest
small_value = df['SMALL HiBM']
big_growth = df['BIG LoBM']

# --- Step 1: Calculate Standard Error ---
se_small_value = stats.sem(small_value, nan_policy='omit')
se_big_growth = stats.sem(big_growth, nan_policy='omit')

# --- Step 2: Construct 95% Confidence Intervals ---
mean_small_value = small_value.mean()
mean_big_growth = big_growth.mean()
n = len(small_value.dropna()) # Get the count of non-NaN values
dof = n - 1 # Degrees of freedom
confidence_level = 0.95
t_critical = stats.t.ppf((1 + confidence_level) / 2, dof)

# Calculate the margin of error
moe_small_value = t_critical * se_small_value
moe_big_growth = t_critical * se_big_growth

# Calculate the confidence intervals
ci_small_value = (mean_small_value - moe_small_value, mean_small_value + moe_small_value)
ci_big_growth = (mean_big_growth - moe_big_growth, mean_big_growth + moe_big_growth)

# --- Print the Results ---
print("--- Statistical Analysis for Monthly Returns (July 1963 onwards) ---")
print(f"\nPortfolio: Small Value (SMALL HiBM)")
print(f"Mean Return: {mean_small_value:.4f}%")
print(f"Standard Error: {se_small_value:.4f}")
print(f"95% Confidence Interval: ({ci_small_value[0]:.4f}%, {ci_small_value[1]:.4f}%)")

print(f"\nPortfolio: Big Growth (BIG LoBM)")
print(f"Mean Return: {mean_big_growth:.4f}%")
print(f"Standard Error: {se_big_growth:.4f}")
print(f"95% Confidence Interval: ({ci_big_growth[0]:.4f}%, {ci_big_growth[1]:.4f}%)")

# --- Step 2: Calculate Annual Compounded Returns ---
# Set 'Date' as the index to use time-series functionalities
df_for_annual = df.set_index('Date')

# Isolate only the portfolio columns for the calculation
portfolio_columns = [col for col in df.columns if col != 'Date']

# Compound the monthly returns to get annual returns
# 1. Convert percentage returns to decimal returns (e.g., 1.45 -> 0.0145) and add 1
# 2. Resample by 'A' (year-end frequency) and take the product of (1+r) for each year
# 3. Subtract 1 to get the annual compounded return
# 4. Multiply by 100 to convert back to percentage format
annual_returns_df = (df_for_annual[portfolio_columns] / 100 + 1).resample('A').prod() - 1
annual_returns_df = annual_returns_df * 100

# --- Step 3: Plot the Annual Returns ---
plt.figure(figsize=(12, 6))

# Plot the calculated annual returns for the two specified portfolios
plt.plot(annual_returns_df.index, annual_returns_df['SMALL HiBM'], label='Small Value (Small HiBM)', color='blue')
plt.plot(annual_returns_df.index, annual_returns_df['BIG LoBM'], label='Big Growth (Big LoBM)', color='red')

# Add descriptive labels and title for clarity
plt.title('Annual Compounded Returns: Small Value vs. Big Growth Firms')
plt.xlabel('Year')
plt.ylabel('Annual Return (%)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.75, linestyle='--') # Add a zero line for reference
plt.tight_layout()
########################Question 5###########################################################
# --- Step 3: Calculate Average Annual Returns ---
average_annual_returns = annual_returns_df.mean()

# --- Step 4: Prepare Data and Generate 3D Plot ---
mean_grid = np.array(average_annual_returns).reshape(5, 5)

x = np.arange(5)
y = np.arange(5)
X, Y = np.meshgrid(x, y)

x_pos, y_pos = X.flatten(), Y.flatten()
z_pos = np.zeros_like(x_pos)
dz_values = mean_grid.flatten()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(x_pos, y_pos, z_pos, dx=0.8, dy=0.8, dz=dz_values, color='dodgerblue')

ax.set_title('Average Annual Returns of 25 Portfolios (1963-2024)')
ax.set_xlabel('Size (Small to Big)')
ax.set_ylabel('Value (LoBM to HiBM)')
ax.set_zlabel('Average Annual Return (%)')
ax.set_xticks(x)
ax.set_yticks(y)
ax.set_xticklabels(['Small', 'ME2', 'ME3', 'ME4', 'Big'])
ax.set_yticklabels(['LoBM', 'BM2', 'BM3', 'BM4', 'HiBM'])
ax.view_init(elev=20., azim=-65)

#The analysis of both average monthly returns and compounded annual returns 
#yields a consistent result: the portfolio composed of 
#small-capitalization, high book-to-market (value) firms—the SMALL HiBM 
#portfolio—demonstrably generates the highest returns within 
#the 25-portfolio universe.

#This finding serves as a strong  validation of two foundational 
#concepts in modern asset pricing the size premium and the value premium. 
#The superior performance is not an artifact of the measurement frequency 
#but rather a persistent characteristic of this asset class within the dataset.
# The Small Value portfolio is uniquely positioned at the powerful intersection 
#of these two return factors, which explains its robust outperformance relative 
#to other portfolios, particularly the BIG LoBM portfolio.

########################Question 6############################################


portfolio_columns = [col for col in df.columns if col != 'Date']
df_numeric = df[portfolio_columns]


df_decimal = df_numeric / 100 + 1


df_decimal['Date'] = df['Date']
df_decimal.set_index('Date', inplace=True)


rolling_returns_3yr = df_decimal.rolling(window=36).apply(np.prod, raw=True) - 1
rolling_returns_3yr = rolling_returns_3yr * 100

# --- Step 3: Plot the Rolling Returns for the Two Portfolios ---
plt.figure(figsize=(12, 6))
plt.plot(rolling_returns_3yr.index, rolling_returns_3yr['SMALL HiBM'], label='Small Value (3-Year Rolling)', color='blue')
plt.plot(rolling_returns_3yr.index, rolling_returns_3yr['BIG LoBM'], label='Big Growth (3-Year Rolling)', color='red')
plt.title('3-Year Rolling Compounded Returns: Small Value vs. Big Growth')
plt.xlabel('Date')
plt.ylabel('3-Year Compounded Return (%)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.75)
plt.savefig('rolling_returns_3yr.png')

# --- Step 4: Calculate and Plot the Outperformance ---
outperformance = rolling_returns_3yr['SMALL HiBM'] - rolling_returns_3yr['BIG LoBM']
plt.figure(figsize=(12, 6))
plt.plot(outperformance.index, outperformance, label='Small Value Outperformance (3-Yr Rolling)', color='purple')
plt.fill_between(outperformance.index, outperformance, 0, where=outperformance >= 0, facecolor='green', interpolate=True, alpha=0.3)
plt.fill_between(outperformance.index, outperformance, 0, where=outperformance < 0, facecolor='red', interpolate=True, alpha=0.3)
plt.title('Outperformance of Small Value vs. Big Growth (3-Year Rolling Return)')
plt.xlabel('Date')
plt.ylabel('Difference in Return (%)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.75)


small_portfolios = [c for c in df.columns if 'SMALL' in c or 'ME1' in c or 'ME2' in c]
big_portfolios = [c for c in df.columns if 'BIG' in c or 'ME4' in c or 'ME5' in c]
low_bm_portfolios = [c for c in df.columns if 'LoBM' in c or 'BM1' in c or 'BM2' in c]
high_bm_portfolios = [c for c in df.columns if 'HiBM' in c or 'BM4' in c or 'BM5' in c]


small_avg = df[small_portfolios].mean(axis=1)
big_avg = df[big_portfolios].mean(axis=1)
low_bm_avg = df[low_bm_portfolios].mean(axis=1)
high_bm_avg = df[high_bm_portfolios].mean(axis=1)

smb = small_avg - big_avg
hml = high_bm_avg - low_bm_avg

avg_smb = smb.mean()
avg_hml = hml.mean()
correlation = smb.corr(hml)


print("--- Fama-French Factor Analysis ---")
print(f"Average SMB (Small Minus Big) Return: {avg_smb:.4f}% per month")
print(f"Average HML (High Minus Low) Return: {avg_hml:.4f}% per month")
print(f"Correlation between SMB and HML: {correlation:.4f}")

#1973-75 (Stagflation): During this period of high inflation and low economic 
#growth, Small Value stocks suffered significant losses. As riskier assets, 
#they are highly sensitive to broad economic distress, and investors flock 
#to safer, larger companies during such prolonged downturns.

#Early 1990s (Recovery Phase): The recession ended in 1991. The period around 
#1993 was characterized by a strong economic recovery. Small Value stocks 
#delivered robust outperformance during this time. Coming out of a recession, 
#smaller firms have more room for growth and benefit disproportionately 
#from renewed economic activity and investor optimism.

#2000 (Dot-com Bubble Burst): In this unique, sector-specific recession, 
#Small Value stocks performed exceptionally well. The downturn was 
#concentrated in overvalued large-cap technology and "growth" stocks. 
#As a result, Small Value stocks, being fundamentally different, acted as a
# relative safe haven and a hedge against the collapse of the growth-stock 
#bubble.

#2008 (Global Financial Crisis): Small Value stocks experienced severe losses, 
#often greater than the market average. This was a systemic crisis driven by a 
#credit crunch. Smaller firms are typically more reliant on credit and are 
#perceived as less stable, making them particularly vulnerable when financial 
#markets freeze up.

#2020 (COVID-19 Crash & Recovery): They saw a very sharp initial drop 
#followed by a massive and rapid rebound. The unprecedented government 
#stimulus that followed the crash heavily favored riskier assets. 
#Small Value stocks, with their higher growth potential, were prime
# beneficiaries of this rapid, liquidity-fueled recovery.

##########################Question 7#########################################

# --- Step 2: Isolate Small Value Portfolio and Calculate Returns ---
# Get the monthly returns for the 'SMALL HiBM' (Small Value) portfolio
monthly = df[['Date', 'SMALL HiBM']].set_index('Date')['SMALL HiBM'].dropna()

# Convert to decimal for compounding
monthly_decimal = monthly / 100

# Calculate 3-year compounded returns (36 months)
three_year = (1 + monthly_decimal).rolling(window=36).apply(np.prod, raw=True) - 1
three_year = three_year * 100 # Convert back to percentage

# --- Step 3: Plot the Histogram ---
# Create a single figure
plt.figure(figsize=(8, 6))

# Plot for 3-Year Returns
plt.hist(three_year.dropna(), bins=10, density=True, color='salmon', edgecolor='black')
plt.title("3-Year Compounded Returns (Small Value)")
plt.xlabel("Return (%)")
plt.ylabel("Probability Density")

#As we aggregate returns over longer horizons—from monthly to annual—their 
#distribution becomes less extreme and begins to look much more like a normal 
#bell-shaped curve.

#This happens because of a statistical concept called the 
#Central Limit Theorem. a single month's stock return can be a  outlier.
# An annual return is the compounded result of 12 of these months. 
#The extreme months tend to be smoothed out by the more typical months, 
#leading to a more predictable, bell-shaped distribution.
###################################Question 8###############################


small_value = df['SMALL HiBM'].dropna()
big_growth = df['BIG LoBM'].dropna()

bins = 10
H, xedges, yedges = np.histogram2d(small_value, big_growth, bins=bins)


# The positions are the centers of the bins
xpos, ypos = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# The height of the bars is the count in each bin
dz = H.ravel()


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Width of the bars
dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]

# Plot the 3D bars
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# Add labels and title
ax.set_title('Joint Return Distribution: Small Value vs. Big Growth')
ax.set_xlabel('Small Value Returns (%)')
ax.set_ylabel('Big Growth Returns (%)')
ax.set_zlabel('Frequency (Count)')


ax.view_init(elev=25., azim=-125)

#No, the two portfolios do not provide a hedge for each other. 
#In fact, the analysis shows the opposite they have a strong positive correlation.

#Interpreting the 3D Histogram- As you can see in the plot, 
#the tallest bars are clustered along a diagonal line.
#This means the most frequent outcomes are:

#Bottom-Left Quadrant: Both portfolios experience negative returns at the same time.

#Top-Right Quadrant: Both portfolios experience positive returns at the same time.

#There are very few instances  where one portfolio has high returns while 
#the other has low returns. This visual evidence clearly indicates that 
#the two portfolios tend to move in the same direction, driven by the 
#same broad market forces. They succeed together and fail together, 
#which is the opposite of a hedging relationship.

##############################################Question 9#########################

mean_sv = small_value.mean()
std_sv = small_value.std()
standardized_sv = (small_value - mean_sv) / std_sv


mean_bg = big_growth.mean()
std_bg = big_growth.std()
standardized_bg = (big_growth - mean_bg) / std_bg


final_mean_sv = standardized_sv.mean()
final_std_sv = standardized_sv.std()

final_mean_bg = standardized_bg.mean()
final_std_bg = standardized_bg.std()


print("--- Analysis of Standardized Returns (Z-Scores) ---")
print("\nFor the 'Small Value' Portfolio:")
print(f"Mean of Standardized Returns: {final_mean_sv:.4f}")
print(f"Standard Deviation of Standardized Returns: {final_std_sv:.4f}")

print("\nFor the 'Big Growth' Portfolio:")
print(f"Mean of Standardized Returns: {final_mean_bg:.4f}")
print(f"Standard Deviation of Standardized Returns: {final_std_bg:.4f}")


skew_sv = stats.skew(standardized_sv, bias=False)
kurt_sv = stats.kurtosis(standardized_sv, fisher=False, bias=False)  # kurtosis, normal=3

print("\n Small Value Skewness:",skew_sv)
print("  Small Value Kurtosis:",kurt_sv)

skew_bg = stats.skew(standardized_bg, bias=False)
kurt_bg = stats.kurtosis(standardized_bg, fisher=False, bias=False)

print(" \n Big growth Skewness:",skew_bg)
print("  Big growth Kurtosis: ",kurt_bg)

#The mean and standard deviation are a perfect match. 
#This is expected, as the process of standardization 
#mathematically forces the resulting data to have these values. 

#The skewness values for both portfolios are very close to zero, 
#suggests their return distributions are mostly symmetric. 
#The slight negative skew in the Big Growth portfolio indicates
# a slightly longer tail on the left, meaning large negative 
#returns occurred a bit more frequently than large positive ones.

#The kurtosis values are significantly higher than 3. 
#This is the most important deviation from normality. 
#It indicates that both distributions are leptokurtic

#extreme outcomes (both large gains and large losses) 
#are much more common for these portfolios 
#than a normal bell curve would predict.

####################################Question 10#######################

sv_monthly = df['SMALL HiBM'].dropna()
bg_monthly = df['BIG LoBM'].dropna()


sv_monthly_dec = sv_monthly / 100
bg_monthly_dec = bg_monthly / 100


sv_annual = (1 + sv_monthly_dec).rolling(window=12).apply(np.prod, raw=True) - 1
bg_annual = (1 + bg_monthly_dec).rolling(window=12).apply(np.prod, raw=True) - 1
sv_annual = (sv_annual * 100).dropna()
bg_annual = (bg_annual * 100).dropna()


sv_monthly_std = (sv_monthly - sv_monthly.mean()) / sv_monthly.std()
sv_annual_std = (sv_annual - sv_annual.mean()) / sv_annual.std()
bg_monthly_std = (bg_monthly - bg_monthly.mean()) / bg_monthly.std()
bg_annual_std = (bg_annual - bg_annual.mean()) / bg_annual.std()


fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
x_norm = np.linspace(-4, 4, 1000)
y_norm = stats.norm.pdf(x_norm, 0, 1)


# Top-Left: Small Value, Monthly
axes[0, 0].hist(sv_monthly_std, bins=120, density=True, color='skyblue', label='Monthly Returns')
axes[0, 0].plot(x_norm, y_norm, color='red', linestyle='--', label='Normal PDF')
axes[0, 0].set_title('Small Value - Monthly')
axes[0, 0].set_ylabel('Probability Density')
axes[0, 0].legend()

# Top-Right: Small Value, Annual
axes[0, 1].hist(sv_annual_std, bins=10, density=True, color='lightgreen', label='Annual Returns')
axes[0, 1].plot(x_norm, y_norm, color='red', linestyle='--', label='Normal PDF')
axes[0, 1].set_title('Small Value - Annual')
axes[0, 1].legend()

# Bottom-Left: Big Growth, Monthly
axes[1, 0].hist(bg_monthly_std, bins=120, density=True, color='salmon', label='Monthly Returns')
axes[1, 0].plot(x_norm, y_norm, color='red', linestyle='--', label='Normal PDF')
axes[1, 0].set_title('Big Growth - Monthly')
axes[1, 0].set_xlabel('Standard Deviations (Z-Score)')
axes[1, 0].set_ylabel('Probability Density')
axes[1, 0].legend()

# Bottom-Right: Big Growth, Annual
axes[1, 1].hist(bg_annual_std, bins=10, density=True, color='plum', label='Annual Returns')
axes[1, 1].plot(x_norm, y_norm, color='red', linestyle='--', label='Normal PDF')
axes[1, 1].set_title('Big Growth - Annual')
axes[1, 1].set_xlabel('Standard Deviations (Z-Score)')
axes[1, 1].legend()

plt.tight_layout()

# monthly stock returns are not normally distributed. 
#The presence of fat tails is a well-documented phenomenon in finance, 
#and it means that relying on a normal distribution for risk management 
#would lead one to severely underestimate the probability of
# extreme market events (both crashes and rallies).

#Conversely, the analysis shows that annual returns are much closer to 
#being normally distributed. The process of aggregating and compounding 
#returns over a longer horizon (12 months) tends to smooth out the 
#extreme short-term fluctuations, leading to a distribution that 
#more closely resembles a normal bell curve.

#############################Question 11#######################################

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-Left: Small Value, Monthly
stats.probplot(sv_monthly_std, dist="norm", plot=axes[0, 0])
axes[0, 0].set_title('Small Value - Monthly Returns')

# Top-Right: Small Value, Annual
stats.probplot(sv_annual_std, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Small Value - Annual Returns')

# Bottom-Left: Big Growth, Monthly
stats.probplot(bg_monthly_std, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Big Growth - Monthly Returns')

# Bottom-Right: Big Growth, Annual
stats.probplot(bg_annual_std, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Big Growth - Annual Returns')

plt.tight_layout()

#Monthly Returns (Left Column): The data points form a distinct "S" shape 
#that deviates significantly from the straight red line, especially 
#at the ends. This is the classic signature of a distribution with 
#"fat tails." It provides strong evidence that monthly returns 
#are not normally distributed, as extreme events are more common 
#than a normal distribution would predict.

#Annual Returns (Right Column): The data points follow the straight 
#red line much more closely. This indicates that the distribution 
#of annual returns is a much better approximation of a normal distribution.

#In short, the QQ-plots provide a more precise confirmation of 
#what we saw in the histograms: aggregating returns over a 
#longer horizon (annually) smooths out the extreme short-term behavior,
# making the distribution appear more normal.