import pandas as pd
from fredapi import Fred
import numpy as np
from scipy.stats import zscore
from scipy.stats import skew, kurtosis
from scipy.stats import percentileofscore
from scipy.stats import norm
from math import erf, sqrt

# Initialize FRED API with your API key
fred = Fred(api_key='eb36db0397ab6c514f1670c79fe0d3a2')


# Fetch Quarterly GDP (use the real GDP series ID, "GDPC1" for real GDP in the United States)
gdp_series = fred.get_series('GDPC1')  # Real GDP

# Calculate GDP Growth Rate as percentage change from the previous period
gdp_growth_rate = gdp_series.pct_change() * 100  # Convert to percentage

# Specify lookback period
lookback_period = 12  # Adjust based on your needs, noting GDP data is quarterly

# Calculate moving average and standard deviation for the GDP growth rate
gdp_avg = gdp_growth_rate.rolling(window=lookback_period).mean()
gdp_std = gdp_growth_rate.rolling(window=lookback_period).std()

# Calculate dynamic thresholds
upper_threshold = gdp_avg + gdp_std
lower_threshold = gdp_avg - gdp_std

# Assess current GDP growth rate
current_gdp_growth_rate = gdp_growth_rate.iloc[-1]
current_upper_threshold = upper_threshold.iloc[-1]
current_lower_threshold = lower_threshold.iloc[-1]

# Determine the status of the current GDP growth rate
if current_gdp_growth_rate > current_upper_threshold:
    status = 'above the upper threshold'
elif current_gdp_growth_rate < current_lower_threshold:
    status = 'below the lower threshold'
else:
    status = 'within the thresholds'


# Assumptions for thresholds and weights
TGDPLow = -2  # Assumed low GDP growth threshold
TGDPHigh = 5  # Assumed high GDP growth threshold
w1, w2, w3 = 0.4, 0.3, 0.3  # Weights for current GDP, mean GDP, and standard deviation

# Step 2: Normalize Each Component
# Assuming MGDP (Mean GDP Growth Rate) and SDGDP (Standard Deviation of GDP Growth Rate) are calculated from historical data

# Use the historical mean and standard deviation as placeholders
# In practice, calculate these from historical GDP growth rate data
MGDP = gdp_avg.mean()  # Mean of historical GDP growth rates
SDGDP = gdp_std.mean()  # Mean of historical standard deviation of GDP growth rates

# Normalization of Current GDP Growth Rate
NCGDP = (current_gdp_growth_rate - TGDPLow) / (TGDPHigh - TGDPLow)

# Normalization of Mean GDP Growth Rate
# Assuming Min and Max of historical mean GDP growth rates for normalization
Min_MGDP_hist = gdp_avg.min()
Max_MGDP_hist = gdp_avg.max()
NMGDP = (MGDP - Min_MGDP_hist) / (Max_MGDP_hist - Min_MGDP_hist)

# Normalization of Standard Deviation of GDP Growth Rate
# Assuming a maximum standard deviation from historical data for normalization
Max_SDGDP_hist = gdp_std.max()
NSDGDP = 1 - (SDGDP / Max_SDGDP_hist)

# Step 3: Composite Score Calculation
# Calculate composite score with applied weights
CS = (w1 * NCGDP) + (w2 * NMGDP) + (w3 * NSDGDP)

# Step 4: Final Adjustments
# Ensure CS is between 0 and 1
GDPCompositeScore = max(0, min(1, CS))

# Print the composite score
#print(f"GDP Composite Score: {GDPCompositeScore:.2f}")



################################################################
################################################################
################################################################


# Fetch Consumer Price Index data
cpi_data = fred.get_series('CPIAUCSL')

# Convert CPI data to pandas DataFrame
cpi_df = pd.DataFrame(cpi_data, columns=['CPI'])

# Calculate inflation rate as percentage change in CPI
cpi_df['Inflation_Rate'] = cpi_df['CPI'].pct_change() * 100

# Calculate 12-month moving average for the inflation rate
cpi_df['MA_12'] = cpi_df['Inflation_Rate'].rolling(window=12).mean()

# Calculate standard deviation of the inflation rate over a 12-month period
cpi_df['Std_Dev_12'] = cpi_df['Inflation_Rate'].rolling(window=12).std()

# Calculate key percentiles of historical inflation rate data
percentiles = cpi_df['Inflation_Rate'].quantile([0.25, 0.5, 0.75])

# Define dynamic thresholds
low_inflation_threshold = percentiles[0.25] - cpi_df['Std_Dev_12'].mean()
moderate_inflation_threshold = percentiles[0.5]
high_inflation_threshold = percentiles[0.75] + cpi_df['Std_Dev_12'].mean()

# Get the latest inflation rate and its rate of change
latest_inflation_rate = cpi_df['Inflation_Rate'].iloc[-1]
rate_of_change = cpi_df['Inflation_Rate'].diff().iloc[-1]

# Normalization of Each Component
N_CIR = (latest_inflation_rate - low_inflation_threshold) / (high_inflation_threshold - low_inflation_threshold)
N_RCIR = (rate_of_change - (-0.1)) / (0.1 - (-0.1))

# Assuming dynamic calculation for thresholds normalization
if latest_inflation_rate <= low_inflation_threshold:
    N_Thresholds = 0
elif low_inflation_threshold < latest_inflation_rate <= moderate_inflation_threshold:
    N_Thresholds = 0.5
else:
    N_Thresholds = 1

# Percentiles Normalization
# Assuming linear interpolation for simplicity
N_Percentiles = (latest_inflation_rate - percentiles[0.5]) / (percentiles[0.75] - percentiles[0.5])

# Moving Average Normalization
N_MA = ((latest_inflation_rate - cpi_df['MA_12'].iloc[-1]) / (latest_inflation_rate + cpi_df['MA_12'].iloc[-1])) + 0.5

# Ensure all normalized scores are within [0, 1]
N_CIR, N_RCIR, N_Percentiles, N_MA = [max(0, min(1, x)) for x in [N_CIR, N_RCIR, N_Percentiles, N_MA]]

# Weight Assignment
W_CIR = 0.25
W_RCIR = 0.20
W_Thresholds = 0.15
W_Percentiles = 0.20
W_MA = 0.20

# Composite Score Calculation
Inflation_CS = (N_CIR * W_CIR) + (N_RCIR * W_RCIR) + (N_Thresholds * W_Thresholds) + (N_Percentiles * W_Percentiles) + (N_MA * W_MA)

# Print the composite score
#print(f"Inflation Composite Score: {Inflation_CS:.4f}")


################################################################
################################################################
################################################################



#Fetch historical unemployment rate data
unemployment_rate = fred.get_series('UNRATE')  # 'UNRATE' is the series ID for Civilian Unemployment Rate
# Convert to DataFrame for easier handling
unemployment_rate_df = pd.DataFrame(unemployment_rate, columns=['UnemploymentRate'])

# Calculate statistical metrics
mean_unemployment_rate = np.mean(unemployment_rate)
std_dev_unemployment_rate = np.std(unemployment_rate)
percentiles_unemployment_rate = np.percentile(unemployment_rate, [25, 50, 75, 90])

# Dynamic thresholds determination
normal_range_low = np.percentile(unemployment_rate, 25)
normal_range_high = np.percentile(unemployment_rate, 75)
elevated_unemployment_threshold = np.percentile(unemployment_rate, 75) + std_dev_unemployment_rate
low_unemployment_threshold = np.percentile(unemployment_rate, 25) - std_dev_unemployment_rate

current_unemployment_rate = unemployment_rate_df['UnemploymentRate'].iloc[-1]


# Step 1: Normalize Individual Components
# Current Unemployment Rate (CUR) normalization
if current_unemployment_rate <= normal_range_high:
    cur_score = 1 - (current_unemployment_rate - normal_range_low) / (normal_range_high - normal_range_low)
else:
    cur_score = 1 - ((current_unemployment_rate - normal_range_high) / (elevated_unemployment_threshold - normal_range_high)) * 0.5

# Mean Unemployment Rate (MUR) normalization
median_unemployment_rate = percentiles_unemployment_rate[1]
if mean_unemployment_rate > median_unemployment_rate:
    mur_score = 1 - abs(mean_unemployment_rate - median_unemployment_rate) / (elevated_unemployment_threshold - median_unemployment_rate)
else:
    mur_score = 1

# Standard Deviation of Unemployment Rate (SDUR) normalization
sdur_score = 1 - std_dev_unemployment_rate / (elevated_unemployment_threshold - low_unemployment_threshold)

# Percentile Scores (PS) normalization
ps_score = sum(1 - abs(current_unemployment_rate - percentile) / (elevated_unemployment_threshold - low_unemployment_threshold) for percentile in percentiles_unemployment_rate) / 4

# Step 2: Composite Score Calculation
# Assuming equal weights for simplification
weights = {'cur': 0.25, 'mur': 0.25, 'sdur': 0.25, 'ps': 0.25}
composite_score = (cur_score * weights['cur'] + mur_score * weights['mur'] + sdur_score * weights['sdur'] + ps_score * weights['ps'])

# Step 3: Ensuring the Score is Between 0 and 1
Unemployment_final_composite_score = min(max(composite_score, 0), 1)

# Print the final composite score
#print(f"Unemployment Composite Score: {Unemployment_final_composite_score}")


################################################################
################################################################
################################################################
interest_rate_data = fred.get_series('FEDFUNDS')

# Convert to DataFrame for easier handling
interest_rates_df = pd.DataFrame(interest_rate_data, columns=['InterestRate'])

# Calculate statistical measures
mean_interest_rate = interest_rates_df['InterestRate'].mean()
std_deviation = interest_rates_df['InterestRate'].std()
percentile_25 = interest_rates_df['InterestRate'].quantile(0.25)
median_interest_rate = interest_rates_df['InterestRate'].median()
percentile_75 = interest_rates_df['InterestRate'].quantile(0.75)

# Dynamic thresholds with volatility adjustment
volatility_adjustment_factor = std_deviation * 0.1  # Example adjustment factor
adjusted_low_interest_threshold = percentile_25 - volatility_adjustment_factor
adjusted_high_interest_threshold = percentile_75 + volatility_adjustment_factor

# Trend analysis: Assuming the series is at a monthly frequency
interest_rates_df['12M_MA'] = interest_rates_df['InterestRate'].rolling(window=12).mean()

# Current Interest Rate
current_interest_rate = interest_rates_df['InterestRate'].iloc[-1]
current_trend = interest_rates_df['12M_MA'].iloc[-1]

# Determine current interest rate environment
if current_interest_rate < adjusted_low_interest_threshold:
    current_environment = 'Low Interest Rate Environment'
elif adjusted_low_interest_threshold <= current_interest_rate <= adjusted_high_interest_threshold:
    current_environment = 'Normal Interest Rate Environment'
else:
    current_environment = 'High Interest Rate Environment'


# Placeholder values for MinIR, MaxIR, MinStdDev, and MaxStdDev
MinIR = interest_rates_df['InterestRate'].min() - 1  # Hypothetical minimum
MaxIR = interest_rates_df['InterestRate'].max() + 1  # Hypothetical maximum
MinStdDev = 0  # Hypothetical minimum standard deviation
MaxStdDev = std_deviation * 2  # Hypothetical maximum standard deviation

# Normalization of Individual Components
NormalizedCIR = (current_interest_rate - MinIR) / (MaxIR - MinIR)
NormalizedMA = (current_trend - MinIR) / (MaxIR - MinIR)
NormalizedIRE = 1 if current_environment == 'Normal Interest Rate Environment' else 0.5
NormalizedPercentiles = (median_interest_rate - MinIR) / (MaxIR - MinIR)
NormalizedStdDev = 1 - (std_deviation - MinStdDev) / (MaxStdDev - MinStdDev)

# Weighted Composite Score Calculation
w_1, w_2, w_3, w_4, w_5 = 0.2, 0.2, 0.2, 0.2, 0.2  # Example weights
CompositeScore = (w_1 * NormalizedCIR + w_2 * NormalizedMA + w_3 * NormalizedIRE +
                  w_4 * NormalizedPercentiles + w_5 * NormalizedStdDev)

# Final Adjustments
Interest_Rate_FinalCompositeScore = min(max(CompositeScore, 0), 1)

#print(f"Interest Rate Composite Score: {Interest_Rate_FinalCompositeScore}")


################################################################
################################################################
################################################################

# Fetch Consumer Confidence Index (CCI) data
cci = fred.get_series('UMCSENT')  # UMCSENT is the series ID for the University of Michigan's Consumer Sentiment Index, similar to CCI

# Convert to DataFrame for easier manipulation
cci_df = pd.DataFrame(cci, columns=['CCI'])

# Calculate statistical metrics
mean_cci = cci_df['CCI'].mean()
median_cci = cci_df['CCI'].median()
std_dev_cci = cci_df['CCI'].std()
percentile_25_cci = cci_df['CCI'].quantile(0.25)
percentile_75_cci = cci_df['CCI'].quantile(0.75)
percentile_90_cci = cci_df['CCI'].quantile(0.90)

# Dynamic Threshold Setting
low_confidence_threshold = percentile_25_cci
moderate_confidence_range_start = mean_cci - std_dev_cci
high_confidence_threshold = percentile_75_cci

# Adjusting for Economic Context
# Trend Analysis - Calculate 12-month Moving Average
cci_df['12M_MA'] = cci_df['CCI'].rolling(window=12).mean()

# Determine current CCI status
current_cci = cci_df['CCI'].iloc[-1]

# Normalization of Individual Components
min_cci = percentile_25_cci
max_cci = percentile_90_cci

# 1.1 Current Print Normalization (CPN)
cpn = (current_cci - min_cci) / (max_cci - min_cci)

# 1.2 Deviation from Mean Normalization (DMN)
dmn = 1 - abs(mean_cci - current_cci) / std_dev_cci

# 1.3 Position Within Confidence Thresholds (PWCT)
if current_cci < moderate_confidence_range_start:
    pwct = 0.5 * (current_cci - low_confidence_threshold) / (moderate_confidence_range_start - low_confidence_threshold)
elif current_cci <= high_confidence_threshold:
    pwct = 0.5 + 0.5 * (current_cci - moderate_confidence_range_start) / (high_confidence_threshold - moderate_confidence_range_start)
else:
    pwct = 1

# 1.4 Percentile Rank Normalization (PRN)
prn = (current_cci - percentile_25_cci) / (percentile_90_cci - percentile_25_cci)

# Step 2: Composite Score Calculation
# Assigning equal weights for simplicity; adjust as needed
weights = [0.25, 0.25, 0.25, 0.25]  # w1, w2, w3, w4
CCI_composite_score = weights[0] * cpn + weights[1] * dmn + weights[2] * pwct + weights[3] * prn

#print(f"Consumer Confidence Composite Score: {CCI_composite_score}")


################################################################
################################################################
################################################################

# Fetch Industrial Production (IP) data
ip_series_id = 'INDPRO'  # Replace with the correct series ID for Industrial Production
ip_data = fred.get_series(ip_series_id)

# Convert to DataFrame for easier manipulation
ip_df = pd.DataFrame(ip_data, columns=['IP'])

# 1. Historical Data Analysis
# Assuming ip_data is already a comprehensive dataset

# 2. Calculate Descriptive Statistics
mean_ip = ip_df['IP'].mean()
std_dev_ip = ip_df['IP'].std()
percentile_25th = ip_df['IP'].quantile(0.25)
median_ip = ip_df['IP'].quantile(0.5)  # 50th percentile
percentile_75th = ip_df['IP'].quantile(0.75)


# 3. Dynamic Threshold Setting
growth_threshold = mean_ip + std_dev_ip  # Example for rising
stability_range_low = percentile_25th  # Example for stable
stability_range_high = percentile_75th
decline_threshold = mean_ip - std_dev_ip  # Example for falling

# 4. Trend Analysis
# Calculate moving averages for trend analysis
ip_df['Short_Term_MA'] = ip_df['IP'].rolling(window=12).mean()  # 12-period (e.g., months) moving average for short-term trend
ip_df['Long_Term_MA'] = ip_df['IP'].rolling(window=60).mean()  # 60-period (e.g., months) moving average for long-term trend

# Assuming the last value is the current IP
current_ip = ip_df['IP'].iloc[-1]

# Evaluate current IP against thresholds
if current_ip > growth_threshold:
    current_status = "rising"
elif stability_range_low <= current_ip <= stability_range_high:
    current_status = "stable"
elif current_ip < decline_threshold:
    current_status = "falling"
else:
    current_status = "undefined"
# Step 1: Deviation from Mean Score (D)
D = np.exp(-(abs(current_ip - mean_ip)**2) / (2 * std_dev_ip**2))

# Step 2: Percentile Score (P)
if current_ip <= median_ip:
    P = 2 * ((current_ip - percentile_25th) / (median_ip - percentile_25th))
else:
    P = 1 - 2 * ((current_ip - median_ip) / (percentile_75th - median_ip))
P = max(min(P, 1), 0)  # Ensure P is within the 0 to 1 range

# Step 3: Trend Adjustment (T)
T = 1.05 if current_status == "rising" else 0.95

# Step 4: Composite Score Calculation
CompositeScore = ((D + P) / 2) * T

# Final Adjustment
InudstrialProd_FinalCompositeScore = min(max(CompositeScore, 0), 1)

#print(f"Industrial Production Composite Score: {InudstrialProd_FinalCompositeScore}")


################################################################
################################################################
################################################################
# 1. Historical Data Analysis
# Collect Historical Housing Starts Data
housing_starts_series = fred.get_series('HOUST', observation_start='1960-01-01')

# Convert the fetched series to a DataFrame
housing_starts = pd.DataFrame(housing_starts_series, columns=['Housing_Starts'])

# Ensure the data is treated as numeric, attempting to convert non-numeric data
housing_starts['Housing_Starts'] = pd.to_numeric(housing_starts['Housing_Starts'], errors='coerce')

# Now that the data is ensured to be numeric, proceed with the analysis

# Calculate Moving Averages
housing_starts['12_month_MA'] = housing_starts['Housing_Starts'].rolling(window=12).mean()
housing_starts['3_month_MA'] = housing_starts['Housing_Starts'].rolling(window=3).mean()

# Calculate Standard Deviation Bands
housing_starts['Std_Dev'] = housing_starts['Housing_Starts'].rolling(window=12).std()
housing_starts['Upper_Band'] = housing_starts['12_month_MA'] + (2 * housing_starts['Std_Dev'])
housing_starts['Lower_Band'] = housing_starts['12_month_MA'] - (2 * housing_starts['Std_Dev'])

# Calculate Percentile Ranges dynamically
housing_starts['25th_Percentile'] = housing_starts['Housing_Starts'].expanding().quantile(0.25)
housing_starts['50th_Percentile'] = housing_starts['Housing_Starts'].expanding().quantile(0.5)
housing_starts['75th_Percentile'] = housing_starts['Housing_Starts'].expanding().quantile(0.75)

# Adjust for outliers (simple method)
mean = housing_starts['Housing_Starts'].mean()
std_dev = housing_starts['Housing_Starts'].std()
housing_starts_filtered = housing_starts[(housing_starts['Housing_Starts'] > (mean - 3 * std_dev)) & (housing_starts['Housing_Starts'] < (mean + 3 * std_dev))]

# Calculate Growth Rate and Volatility
housing_starts_filtered['MoM_Growth'] = housing_starts_filtered['Housing_Starts'].pct_change(periods=1) * 100
housing_starts_filtered['YoY_Growth'] = housing_starts_filtered['Housing_Starts'].pct_change(periods=12) * 100
housing_starts_filtered['Volatility'] = housing_starts_filtered['Housing_Starts'].rolling(window=12).std() / housing_starts_filtered['Housing_Starts'].rolling(window=12).mean()

# Extracting the latest values
current_housing_starts = housing_starts_filtered['Housing_Starts'].iloc[-1]
current_yoy_growth = housing_starts_filtered['YoY_Growth'].iloc[-1]
current_mom_growth = housing_starts_filtered['MoM_Growth'].iloc[-1]
current_volatility = housing_starts_filtered['Volatility'].iloc[-1]

# Assuming σ (sigma) for MoM and YoY based on historical norms (these values should be adjusted based on actual historical data)
sigma_mom = housing_starts_filtered['MoM_Growth'].std()
sigma_yoy = housing_starts_filtered['YoY_Growth'].std()

# Extracting percentiles for normalization
percentile_25th = housing_starts_filtered['25th_Percentile'].iloc[-1]
percentile_75th = housing_starts_filtered['75th_Percentile'].iloc[-1]

# Normalizing Current Housing Starts
norm_current = (current_housing_starts - percentile_25th) / (percentile_75th - percentile_25th)
norm_current = np.clip(norm_current, 0, 1)  # Ensuring the score is within 0-1 range

# Normalizing MoM Growth
norm_mom = np.exp(-((current_mom_growth / 100) ** 2) / (2 * sigma_mom ** 2))

# Normalizing YoY Growth
norm_yoy = np.exp(-((current_yoy_growth / 100) ** 2) / (2 * sigma_yoy ** 2))

# Normalizing Volatility
# Assuming κ (kappa) based on historical norms (adjust based on actual data)
kappa = housing_starts_filtered['Volatility'].mean()
norm_volatility = np.exp(-current_volatility * kappa)

# Weighting and Aggregating for Composite Score
housing_starts_composite_score = (norm_current + norm_mom + norm_yoy + norm_volatility) / 4

# Print the Composite Score
#print(f"Housing Starts Composite Score: {housing_starts_composite_score}")


################################################################
################################################################
################################################################

# Fetch Retail Sales data (ensure you use the correct series ID for seasonally adjusted Retail Sales)
series_id = 'RSXFS'  # Example series ID for Retail Sales
retail_sales = fred.get_series(series_id, observation_start='1994-01-01')

# 1. Historical Data Analysis
# Assuming 'retail_sales' is already seasonally adjusted. If not, apply seasonal adjustment methodology.

# 2. Moving Averages
retail_sales_3m = retail_sales.rolling(window=3).mean()  # 3-month moving average
retail_sales_6m = retail_sales.rolling(window=6).mean()  # 6-month moving average
retail_sales_12m = retail_sales.rolling(window=12).mean()  # 12-month moving average
retail_sales_24m = retail_sales.rolling(window=24).mean()  # 24-month moving average

# 3. Volatility Measurement
retail_sales_pct_change = retail_sales.pct_change()  # Calculate percentage change for growth rate
retail_sales_std_dev = retail_sales_pct_change.std()  # Standard deviation of Retail Sales growth rates

# Volatility bands
upper_band = retail_sales_pct_change.mean() + retail_sales_std_dev
lower_band = retail_sales_pct_change.mean() - retail_sales_std_dev

# 4. Percentile Ranks
percentiles = np.percentile(retail_sales_pct_change.dropna(), [25, 50, 75, 90])

# Determine current Retail Sales growth rate percentile
current_growth_rate = retail_sales_pct_change.iloc[-1]  # Get the most recent growth rate
current_percentile = np.sum(current_growth_rate > percentiles) / len(percentiles) * 100

# 5. Trend Analysis
# Rate of change (second derivative of Retail Sales)
rate_of_change = retail_sales_pct_change.diff()


# Historical data for normalization (these values need to be defined based on your historical data analysis)
MRRS_min = retail_sales.min()
MRRS_max = retail_sales.max()
SDRSGR_min, SDRSGR_max = retail_sales.pct_change().std().min(), retail_sales.pct_change().std().max()
VB_min, VB_max = (retail_sales.pct_change().mean() - retail_sales.pct_change().std()).min(), (retail_sales.pct_change().mean() + retail_sales.pct_change().std()).max()
RCRSGR_min, RCRSGR_max = retail_sales.pct_change().diff().min(), retail_sales.pct_change().diff().max()

# Normalize components
Norm_MRRS = (retail_sales.iloc[-1] - MRRS_min) / (MRRS_max - MRRS_min) if MRRS_max != MRRS_min else 0

# Moving Averages Normalization
MA_values = [retail_sales.rolling(window=i).mean().iloc[-1] for i in [3, 6, 12, 24]]
Norm_MA = np.mean([(MA - MRRS_min) / (MRRS_max - MRRS_min) for MA in MA_values]) if MRRS_max != MRRS_min else 0

# Standard Deviation of Retail Sales Growth Rates Normalization
SDRSGR = retail_sales.pct_change().std()
Norm_SDRSGR = (SDRSGR - SDRSGR_min) / (SDRSGR_max - SDRSGR_min) if SDRSGR_max != SDRSGR_min else 0

# Volatility Bands Normalization
distance = VB_max - VB_min
if distance > 0:
    upper_band = retail_sales.pct_change().mean() + SDRSGR
    lower_band = retail_sales.pct_change().mean() - SDRSGR
    Norm_VB = ((upper_band - lower_band) - VB_min) / distance
else:
    Norm_VB = 0

# Rate of Change in Retail Sales Growth Rate Normalization
RCRSGR = retail_sales.pct_change().diff().iloc[-1]
Norm_RCRSGR = (RCRSGR - RCRSGR_min) / (RCRSGR_max - RCRSGR_min) if RCRSGR_max != RCRSGR_min else 0

# Current Growth Rate Percentile Normalization
# Assuming 'current_percentile' is calculated as shown in your initial script
Norm_CGRP = current_percentile / 100

# Weights (example weights, these should be adjusted based on your analysis)
weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]

# Composite Score Calculation
components = [Norm_MRRS, Norm_MA, Norm_SDRSGR, Norm_VB, Norm_RCRSGR, Norm_CGRP]
components = [0 if np.isnan(comp) else comp for comp in components]  # Replace NaN with 0 for this example
Retail_Sales_composite_score = sum(w * comp for w, comp in zip(weights, components))


#print(f"Retail Sales Composite Score: {Retail_Sales_composite_score}")



################################################################
################################################################
################################################################

# 1. Historical Data Analysis
# Data Collection
sp500_data = fred.get_series('SP500')  # Fetch S&P 500 historical data. Replace 'SP500' with the correct series ID if necessary.

# Trend Analysis
sp500_data = pd.DataFrame(sp500_data, columns=['Closing Price'])
sp500_data['50-day MA'] = sp500_data['Closing Price'].rolling(window=50).mean()
sp500_data['200-day MA'] = sp500_data['Closing Price'].rolling(window=200).mean()

# 2. Statistical Measures for Thresholds
# Standard Deviation
sp500_data['Price Change'] = sp500_data['Closing Price'].pct_change()
sp500_data['Volatility'] = sp500_data['Price Change'].rolling(window=50).std() * (50 ** 0.5)  # Annualized Volatility

# Percentiles
sp500_data['95th Percentile'] = sp500_data['Price Change'].rolling(window=50).quantile(0.95)
sp500_data['5th Percentile'] = sp500_data['Price Change'].rolling(window=50).quantile(0.05)

# Mean Reversion
long_term_mean = sp500_data['Closing Price'].expanding().mean()
sp500_data['Distance from Mean'] = sp500_data['Closing Price'] - long_term_mean

# 3. Dynamic Threshold Adjustment
# This section suggests a method rather than specific code, as dynamic threshold adjustment would be based on 
# ongoing analysis and could involve complex decision-making algorithms depending on your specific criteria.

# Example: Update volatility threshold based on the most recent year's data
recent_volatility = sp500_data['Volatility'].iloc[-252:].mean()  # Assuming 252 trading days in a year

# Normalization functions (placeholders)
def normalize_percentile_rank(value, historical_data):
    return (value - historical_data.min()) / (historical_data.max() - historical_data.min())

# Placeholder weights
weights = {
    'CP': 0.2,
    'AV': 0.2,
    'PPC': 0.2,
    'P5C': 0.2,
    'DLM': 0.1,
    'RVT': 0.1,
}

# Calculate normalized values
normalized_values = {
    'CP': normalize_percentile_rank(sp500_data['Closing Price'].iloc[-1], sp500_data['Closing Price']),
    'AV': 1 - normalize_percentile_rank(sp500_data['Volatility'].iloc[-1], sp500_data['Volatility']),
    'PPC': normalize_percentile_rank(sp500_data['95th Percentile'].iloc[-1], sp500_data['95th Percentile']),
    'P5C': 1 - abs(sp500_data['5th Percentile'].iloc[-1]),
    'DLM': normalize_percentile_rank(sp500_data['Distance from Mean'].iloc[-1], sp500_data['Distance from Mean']),
    'RVT': 1 - normalize_percentile_rank(recent_volatility, sp500_data['Volatility']),
}

# Composite Score Calculation
SP500_composite_score = sum(normalized_values[key] * weights[key] for key in normalized_values)

# Print the composite score
#print(f"SP500 Composite Score: {SP500_composite_score:.4f}")



################################################################
################################################################
################################################################

# Fetch building permits data
building_permits_series = 'PERMIT'  # Example series ID; replace with actual series ID for building permits
building_permits_data = fred.get_series(building_permits_series)

# Historical Data Analysis
historical_period_years = 20  # Example: 20 years
historical_data = building_permits_data.last('20Y')  # Adjust based on the period of interest

# Historical Averages
long_term_average = historical_data.mean()

# Seasonal Adjustments (Example, implement actual seasonal adjustment calculation or method)
# Placeholder for seasonal adjustment factors or method
seasonally_adjusted_data = historical_data  # This should be replaced with actual seasonal adjustment calculation

# Volatility Assessment
standard_deviation = historical_data.std()

# Trend Analysis
moving_average_period = 12  # Example: 12-month moving average
trend_data = historical_data.rolling(window=moving_average_period).mean()

# Dynamic Threshold Setting
# Percentiles for Extreme Values
lower_threshold = historical_data.quantile(0.1)
upper_threshold = historical_data.quantile(0.9)

# Standard Deviation Bands
upper_band = long_term_average + standard_deviation
lower_band = long_term_average - standard_deviation

# Trend Adjustments (Example, implement actual trend adjustment logic)
# Placeholder for trend adjustment logic
adjusted_upper_band = upper_band  # Adjust these based on actual trend analysis
adjusted_lower_band = lower_band

# Implementation Considerations
# Adaptive Mechanism & Anomaly Detection
# Placeholder for adaptive mechanism and anomaly detection logic
# This should include mechanisms to recalibrate thresholds and identify outliers

# Current Building Permits Insight
current_building_permits = building_permits_data[-1]

# Assessing current data against thresholds

# Normalization of Each Component
NormCBP = (current_building_permits - lower_threshold) / (upper_threshold - lower_threshold)
NormMA = (trend_data[-1] - long_term_average) / (upper_band - long_term_average)
NormSDB = (upper_band - current_building_permits) / (upper_band - lower_band)

# Weight Assignment
W_CBP = W_MA = W_SDB = 1/3

# Composite Score Calculation
Building_Permits_Composite_Score = (W_CBP * NormCBP) + (W_MA * NormMA) + (W_SDB * NormSDB)


#print(f"Composite Building Permits Score: {Building_Permits_Composite_Score}")



################################################################
################################################################
################################################################

# Replace 'NOMG_SERIES_ID' with the actual FRED series ID for New Orders for Manufactured Goods
nomg_data = fred.get_series('DGORDER')

# Historical Data Analysis
# Data Range Selection - Adjust the start and end date as needed
start_date = '1995-01-01'
end_date = '2023-01-01'
nomg_data = nomg_data[start_date:end_date]

# Seasonality Adjustment - Assuming the data is already seasonally adjusted. If not, apply seasonal adjustment techniques.

# Calculating Dynamic Thresholds
# Moving Averages
nomg_data_ma = nomg_data.rolling(window=12).mean()

# Standard Deviation
nomg_std = nomg_data.rolling(window=12).std()

# Percentiles
percentiles = {
    '25th': nomg_data.quantile(0.25),
    '50th': nomg_data.quantile(0.5),
    '75th': nomg_data.quantile(0.75),
}

# Current NOMG Value
current_nomg = nomg_data.iloc[-1]


# Calculate P_Score
P_Score = (current_nomg - percentiles['25th']) / (percentiles['75th'] - percentiles['25th'])
P_Score = min(P_Score, 1)  # Capping the score at 1

# Calculate MAD_Score
MAD_Score = (current_nomg - nomg_data_ma.iloc[-1]) / nomg_std.iloc[-1]

# Normalize MAD_Score to a 0-1 range using a sigmoid function
MAD_Score_norm = 1 / (1 + np.exp(-MAD_Score))

# Calculate overall Composite Score, assuming equal importance
NOMG_Composite_Score = (P_Score + MAD_Score_norm) / 2


#print(f"NOMG Composite Score: {NOMG_Composite_Score}")




################################################################
################################################################
################################################################


# Fetch the 10-year and 2-year Treasury yields data
ten_year_yield = fred.get_series('DGS10')  # Series ID for 10-Year Treasury Constant Maturity Rate
two_year_yield = fred.get_series('DGS2')  # Series ID for 2-Year Treasury Constant Maturity Rate

# Calculate the yield spread (10-year yield minus 2-year yield)
yield_spread = ten_year_yield - two_year_yield

# Calculate historical average spread
average_spread = yield_spread.mean()

# Calculate standard deviation of the yield spread
std_dev_spread = yield_spread.std()

# Calculate percentiles of the yield spread
percentile_25th = yield_spread.quantile(0.25)
percentile_50th = yield_spread.quantile(0.5)
percentile_75th = yield_spread.quantile(0.75)

# Moving averages of the yield spread
moving_average_short = yield_spread.rolling(window=12).mean()  # 12-month moving average
moving_average_long = yield_spread.rolling(window=60).mean()  # 60-month moving average

# Volatility analysis (standard deviation of the short-term moving average to assess volatility)
volatility = moving_average_short.std()

# Signal strength and levels
# Assuming historical analysis suggests a -0.5% spread as a recession indicator
recession_threshold = -0.5
current_spread = yield_spread.iloc[-1]  # Latest available yield spread
signal_strength = (current_spread - average_spread) / std_dev_spread
alert_level = 'low'
if current_spread < recession_threshold:
    alert_level = 'high'
elif signal_strength > 1 or signal_strength < -1:
    alert_level = 'medium'

# Normalization of Each Component
# These min and max values should be replaced with the actual min/max from historical data
min_spread, max_spread = yield_spread.min(), yield_spread.max()
min_std, max_std = yield_spread.std().min(), yield_spread.std().max()
min_signal_strength, max_signal_strength = -3, 3  # Example range for signal strength normalization

# Average Spread Normalization (ASN)
ASN = (average_spread - min_spread) / (max_spread - min_spread)

# Standard Deviation Normalization (SDN)
# Standard Deviation Normalization (SDN)
if max_std != min_std:
    SDN = (std_dev_spread - min_std) / (max_std - min_std)
else:
    # Handle the case where there's no variability in standard deviation
    # You might choose to set SDN to 0, 0.5, or any other value that fits your analysis needs
    SDN = 0.5  # Example: setting to a neutral value since there's no variability
    

# Current Spread Normalization (CSN)
CSN = (current_spread - percentile_25th) / (percentile_75th - percentile_25th)

# Signal Strength Normalization (SSN)
SSN = (signal_strength - min_signal_strength) / (max_signal_strength - min_signal_strength)

# Alert Level Adjustment (ALA)
alert_scores = {'low': 1, 'medium': 0.5, 'high': 0}
ALA = alert_scores[alert_level]

# Weight Assignment
# These weights should be adjusted based on their importance
W_ASN, W_SDN, W_CSN, W_SSN, W_ALA = 0.2, 0.2, 0.2, 0.2, 0.2

# Composite Score Calculation
Composite_Score = (ASN * W_ASN) + (SDN * W_SDN) + (CSN * W_CSN) + (SSN * W_SSN) + (ALA * W_ALA)

# Normalization to 0-1 Scale
# Assuming you have multiple composite scores to find the min and max, otherwise, this step may be adjusted
# For demonstration, setting static min and max composite score range
min_composite_score, max_composite_score = 0, 1  # Adjust based on actual composite score range
YieldCurve_Final_Composite_Score = (Composite_Score - min_composite_score) / (max_composite_score - min_composite_score)

#print("Final Composite Yield Curve Score:", YieldCurve_Final_Composite_Score)



####################################################################
####################################################################
####################################################################

# Fetch historical data
federal_funds_rate = fred.get_series('FEDFUNDS')  # Federal Funds Effective Rate
ten_year_treasury_yield = fred.get_series('DGS10')  # 10-Year Treasury Constant Maturity Rate

# Ensure both series cover the same time span
data = pd.DataFrame({
    'Federal_Funds_Rate': federal_funds_rate,
    'Ten_Year_Treasury_Yield': ten_year_treasury_yield,
}).dropna()

# Calculate the spread between 10-Year Treasury Yield and Federal Funds Rate
data['Spread'] = data['Ten_Year_Treasury_Yield'] - data['Federal_Funds_Rate']

# Statistical Analysis
mean_spread = data['Spread'].mean()
median_spread = data['Spread'].median()
std_dev_spread = data['Spread'].std()
percentile_25 = data['Spread'].quantile(0.25)
percentile_50 = data['Spread'].quantile(0.50)  # Same as median
percentile_75 = data['Spread'].quantile(0.75)



# Dynamic Thresholds Setting based on statistical analysis
# Example thresholds, adjust based on further analysis or desired sensitivity
expansion_threshold = percentile_25
recession_warning_threshold = percentile_75
neutral_watch_zone_low = percentile_25
neutral_watch_zone_high = percentile_75

# Determine the current spread condition
current_spread = data['Spread'].iloc[-1]

# 1. Normalization of Current Spread Relative to Statistical Distribution
Z_score = (current_spread - mean_spread) / std_dev_spread
CDF_value = (1 + erf(Z_score / sqrt(2))) / 2

# 2. Adjustment for Economic Interpretation
if current_spread < percentile_25:
    # Economic Expansion Adjustment
    EEA = CDF_value * (1 + abs(current_spread - percentile_25) / percentile_25)
    final_CDF = EEA
elif current_spread > percentile_75:
    # Economic Contraction Adjustment
    ECA = CDF_value * (1 - abs(current_spread - percentile_75) / percentile_75)
    final_CDF = ECA
else:
    # No adjustment needed
    final_CDF = CDF_value

# 3. Composite Score Finalization
FFRv10Yr_composite_score = min(max(final_CDF, 0), 1)

# Print the composite CPS score
#print(f"Composite Fed Funds Rate vs. 10-Year Treasury Yield Score: {FFRv10Yr_composite_score}")



################################################################
################################################################
################################################################

# Fetch VIX data
vix = fred.get_series('VIXCLS')

# Historical Averages
# Adjust the periods as per your analysis goal
historical_avg_10y = vix.last('10Y').mean()
historical_avg_20y = vix.last('20Y').mean()

# Standard Deviations
std_dev_10y = vix.last('10Y').std()
std_dev_20y = vix.last('20Y').std()

# Percentiles
percentiles = [25, 50, 75, 90]
vix_percentiles = np.percentile(vix.dropna(), percentiles)

# Moving Averages
moving_avg_50d = vix.rolling(window=50).mean().iloc[-1]
moving_avg_200d = vix.rolling(window=200).mean().iloc[-1]

# Volatility Regimes
low_threshold = np.percentile(vix.dropna(), 25)
high_threshold = np.percentile(vix.dropna(), 75)
current_vix = vix.iloc[-1]

if current_vix < low_threshold:
    volatility_regime = 'Low VIX Regime'
elif current_vix > high_threshold:
    volatility_regime = 'High VIX Regime'
else:
    volatility_regime = 'Moderate VIX Regime'


# Normalization Functions
def normalize_current_vix(current_vix, vix_series):
    return (current_vix - vix_series.min()) / (vix_series.max() - vix_series.min())

def normalize_historical_averages(current_vix, historical_avg):
    return (historical_avg - current_vix) / historical_avg

def normalize_std_dev(current_vix, mean, std_dev):
    return abs(current_vix - mean) / std_dev

def normalize_percentiles(current_vix, percentiles):
    return np.searchsorted(percentiles, current_vix) / len(percentiles)

def normalize_regime(volatility_regime):
    regimes = {'Low VIX Regime': 1, 'Moderate VIX Regime': 0.5, 'High VIX Regime': 0}
    return regimes[volatility_regime]

# Weight Assignment
weights = {
    'WeightCurrentVIX': 0.30,
    'WeightHistAvg': 0.20,
    'WeightStdDev': 0.15,
    'WeightPercentiles': 0.20,
    'WeightRegime': 0.15,
}

# Normalization
NormCurrentVIX = normalize_current_vix(current_vix, vix)
NormHistAvg10Y = normalize_historical_averages(current_vix, historical_avg_10y)
NormHistAvg20Y = normalize_historical_averages(current_vix, historical_avg_20y)
NormStdDev10Y = normalize_std_dev(current_vix, vix.last('10Y').mean(), std_dev_10y)
NormStdDev20Y = normalize_std_dev(current_vix, vix.last('20Y').mean(), std_dev_20y)
NormPercentiles = normalize_percentiles(current_vix, vix_percentiles)
NormRegime = normalize_regime(volatility_regime)

# Composite Score Calculation
CompositeScore = (
    (NormCurrentVIX * weights['WeightCurrentVIX']) +
    ((NormHistAvg10Y + NormHistAvg20Y) / 2 * weights['WeightHistAvg']) +
    ((NormStdDev10Y + NormStdDev20Y) / 2 * weights['WeightStdDev']) +
    (NormPercentiles * weights['WeightPercentiles']) +
    (NormRegime * weights['WeightRegime'])
)

# Ensuring Score is Between 0 and 1
VIX_CompositeScore = max(0, min(CompositeScore, 1))

#print(f"Composite VIX Score: {VIX_CompositeScore}")



################################################################
################################################################
################################################################

# Assuming corp_bond_spreads and govt_bond_spreads are fetched correctly,
# Calculate the credit spread by subtracting government bond yields from corporate bond yields
credit_spreads = fred.get_series('BAA10Y')

# Calculate baseline metrics
average_credit_spread = np.mean(credit_spreads)
standard_deviation = np.std(credit_spreads)




# Determine the current credit spread
current_credit_spread = credit_spreads[-1]

# Step 1: Normalize the current spread (Z-score)
z_cs = (current_credit_spread - average_credit_spread) / standard_deviation

# Step 2: Sigmoid function to map Z-score to [0,1] interval
sigmoid_z_cs = 1 / (1 + np.exp(-z_cs))

# Step 3: Adjusting for Historical Volatility
# Assuming an arbitrary 'AverageStdDev' as the "normal" volatility. Adjust this based on historical data.
AverageStdDev = np.mean(standard_deviation)  # Example, use actual historical average standard deviation
volatility_factor = np.exp(-((standard_deviation / AverageStdDev) - 1)**2)

# Step 4: Composite Score Calculation
credit_spread_composite_score = sigmoid_z_cs * volatility_factor

# Print the results
#print(f'Composite Credit Spread Score: {credit_spread_composite_score}')



################################################################
################################################################
################################################################

# Fetch High Yield Bond Performance data
hybp_series_id = 'BAMLH0A0HYM2EY'  # Replace with the correct series ID for High Yield Bond Performance
hybp_data = fred.get_series(hybp_series_id)

# Convert to DataFrame for easier manipulation
hybp_df = pd.DataFrame(hybp_data, columns=['HYBP'])

# 1. Historical Averages
# Long-Term Average Performance
long_term_avg = hybp_df['HYBP'].mean()

# Rolling Averages
hybp_df['12_month_MA'] = hybp_df['HYBP'].rolling(window=12).mean()
hybp_df['24_month_MA'] = hybp_df['HYBP'].rolling(window=24).mean()

# 2. Volatility (Standard Deviation)
std_dev = hybp_df['HYBP'].std()

# Rolling Standard Deviation
hybp_df['Rolling_STD'] = hybp_df['HYBP'].rolling(window=12).std()


# 4. Trend Analysis
# Assuming a simple moving average for trend analysis
hybp_df['Trend'] = hybp_df['HYBP'].rolling(window=12).mean()

# 5. Risk-Adjusted Returns (Example: Sharpe Ratio)
# Assuming you have risk-free rate data (rfr) and calculating Sharpe Ratio as an example
ten_year_treasury_yield = fred.get_series('DGS10')  # 10-Year Treasury Constant Maturity Rate

risk_free_rate = ten_year_treasury_yield  # Placeholder for risk-free rate
returns = hybp_df['HYBP'].pct_change().dropna()
excess_returns = returns - risk_free_rate
sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
hybp_df['sharpe_ratio'] = sharpe_ratio

# Define historical min/max values for normalization purposes
# Placeholder values - replace these with actual historical min/max values for each metric
historical_min_max = {
    'HYBP': {'min': hybp_df['HYBP'].min(), 'max': hybp_df['HYBP'].max()},
    '12_month_MA': {'min': hybp_df['12_month_MA'].min(), 'max': hybp_df['12_month_MA'].max()},
    '24_month_MA': {'min': hybp_df['24_month_MA'].min(), 'max': hybp_df['24_month_MA'].max()},
    'Rolling_STD': {'min': hybp_df['Rolling_STD'].min(), 'max': hybp_df['Rolling_STD'].max()},
    # Add other metrics as needed
}

# Normalization function
def normalize(value, min_value, max_value, invert=False):
    normalized = (value - min_value) / (max_value - min_value)
    return 1 - normalized if invert else normalized

# Apply normalization
for metric in historical_min_max.keys():
    invert = True if metric in ['Rolling_STD'] else False  # Example: Invert normalization for volatility metrics
    hybp_df[f'{metric}_normalized'] = hybp_df[metric].apply(lambda x: normalize(x, historical_min_max[metric]['min'], historical_min_max[metric]['max'], invert))

# Special normalization for Sharpe Ratio
sharpe_min = hybp_df['sharpe_ratio'].min()  # Assuming sharpe_ratio is calculated and added to hybp_df
sharpe_max = hybp_df['sharpe_ratio'].max()
hybp_df['sharpe_ratio_normalized'] = hybp_df['sharpe_ratio'].apply(lambda x: normalize(x, sharpe_min, sharpe_max))

# Composite Score Calculation
# Assuming equal weighting for simplicity, adjust as necessary
metrics_normalized = [metric for metric in hybp_df.columns if 'normalized' in metric]
hybp_df['High Yield Bonds composite_score'] = hybp_df[metrics_normalized].mean(axis=1)

hybpcomposite_score = hybp_df[['High Yield Bonds composite_score']].iloc[-1].item()

# Print the composite CPS score
#print(f'Composite High Yield Bonds Score: {hybpcomposite_score}')



################################################################
################################################################
################################################################


# Fetch the three-month LIBOR rate
libor_3m = fred.get_series('LIOR3M', observation_start='1980-01-01')

# Fetch the three-month Treasury Bill rate
t_bill_3m = fred.get_series('DTB3', observation_start='1980-01-01')

# Calculate the TED Spread (difference between LIBOR and T-Bill rates)
ted_spread = libor_3m - t_bill_3m

# Ensure both series cover the same date range
common_dates = libor_3m.dropna().index.intersection(t_bill_3m.dropna().index)
ted_spread = ted_spread[common_dates]

# Long-term Averages
long_term_average = ted_spread.mean()

# Volatility Measurement
volatility = ted_spread.std()

# Percentiles
percentiles = np.percentile(ted_spread.dropna(), [25, 50, 75, 90])

# Dynamic Thresholds Concept
normal_range_lower = long_term_average - volatility
normal_range_upper = long_term_average + volatility

# Alert Zones
mild_concern_threshold = percentiles[2]
high_concern_threshold = percentiles[3]

# Calculate the last non-NaN TED Spread
current_ted_spread = ted_spread.dropna().iloc[-1]

# Calculate the rolling average based on the last available data
rolling_average = ted_spread.dropna().rolling(window=30).mean().iloc[-1]

# Step 1: Normalization of Individual Components
# Current TED Spread Relative Position (CTRP)
ctrp = (current_ted_spread - percentiles[0]) / (percentiles[3] - percentiles[0])
ctrp = max(min(ctrp, 1), 0)  # Clamping the value between 0 and 1

# Volatility Adjustment (VA)
va = 1 - (volatility / long_term_average)
va = max(min(va, 1), 0)  # Clamping the value between 0 and 1

# Trend Analysis (TA)
ta = 1 if current_ted_spread <= rolling_average else 0

# Deviation From Normal Range (DFNR)
dfnr_distance = min(abs(current_ted_spread - normal_range_lower), abs(current_ted_spread - normal_range_upper))
dfnr = 1 - (dfnr_distance / (normal_range_upper - normal_range_lower))
dfnr = max(min(dfnr, 1), 0)  # Clamping the value between 0 and 1

# Threshold Analysis (ThA)
if current_ted_spread <= mild_concern_threshold:
    tha = 1
elif current_ted_spread >= high_concern_threshold:
    tha = 0
else:
    tha = (high_concern_threshold - current_ted_spread) / (high_concern_threshold - mild_concern_threshold)

# Step 2: Weighted Sum of Normalized Components
weights = {'ctrp': 0.3, 'va': 0.2, 'ta': 0.2, 'dfnr': 0.15, 'tha': 0.15}
composite_score = (ctrp * weights['ctrp'] + 
                   va * weights['va'] + 
                   ta * weights['ta'] + 
                   dfnr * weights['dfnr'] + 
                   tha * weights['tha'])

# Step 3: Final Composite Score Adjustment
TED_final_composite_score = min(max(composite_score, 0), 1)

#print(f"Final Composite TED Spread Score: {TED_final_composite_score}")


################################################################
################################################################
################################################################

# Fetch the data
commercial_paper_rate = fred.get_series('CPN3M')  # Placeholder ID for 3-month commercial paper rate
t_bill_rate = fred.get_series('TB3MS')  # Actual ID for 3-month T-bill rate

# Calculate Commercial Paper Spread (CPS)
cps = commercial_paper_rate - t_bill_rate

# Historical Data Analysis
mean_cps = cps.mean()
median_cps = cps.median()
std_dev_cps = cps.std()
percentile_25th = cps.quantile(0.25)
percentile_75th = cps.quantile(0.75)
percentile_90th = cps.quantile(0.90)

# Dynamic Thresholds Development
std_dev_band_1_upper = mean_cps + std_dev_cps
std_dev_band_1_lower = mean_cps - std_dev_cps
std_dev_band_2_upper = mean_cps + 2*std_dev_cps
std_dev_band_2_lower = mean_cps - 2*std_dev_cps

# Nuance and Context Incorporation
cps_volatility = cps.rolling(window=30).std().mean()  # 30-day moving average of standard deviation
volatility_adjusted_mean = mean_cps + cps_volatility
volatility_adjusted_median = median_cps + cps_volatility

# Trend analysis (using a simple moving average as an example)
cps_sma = cps.rolling(window=30).mean()  # 30-day simple moving average

# Placeholder bounds for normalization, these should be replaced with actual historical data bounds
L = cps.min()  # Lower bound, replace with minimum of historical CPS data
U = cps.max()  # Upper bound, replace with maximum of historical CPS data

def normalize(x, L, U):
    #Normalize a value to a [0, 1] range.
    return (x - L) / (U - L) if U - L != 0 else 0

# Weight Assignment
weights = {
    'mean': 0.1,
    'median': 0.1,
    'std_dev': 0.1,
    'percentiles': 0.2,  # Combined weight for all percentiles
    'thresholds': 0.2,  # Combined weight for both thresholds
    'volAdj': 0.2,  # Combined weight for volatility-adjusted mean and median
    'current': 0.1
}

# Composite Score Calculation
commercial_paper_composite_score = (
    weights['mean'] * normalize(mean_cps, L, U) +
    weights['median'] * normalize(median_cps, L, U) +
    weights['std_dev'] * normalize(std_dev_cps, 0, cps.std().max()) +  # Normalizing std deviation differently
    weights['percentiles'] * (normalize(percentile_25th, L, U) + normalize(percentile_75th, L, U) + normalize(percentile_90th, L, U)) / 3 +
    weights['thresholds'] * (normalize(std_dev_band_1_upper, L, U) + normalize(std_dev_band_2_upper, L, U)) / 2 +
    weights['volAdj'] * (normalize(volatility_adjusted_mean, L, U) + normalize(volatility_adjusted_median, L, U)) / 2 +
    weights['current'] * normalize(cps.iloc[-1], L, U)  # Assuming cps.iloc[-1] is the current CPS
)

#print(f"Composite commercial paper spread Score: {commercial_paper_composite_score}")


################################################################
################################################################
################################################################

# Fetch Nonfarm Payroll (NFP) data
nfp_data = fred.get_series('PAYEMS')  # 'PAYEMS' is the series ID for Nonfarm Payroll

# Historical Data Analysis (your existing script)
mean_nfp = nfp_data.mean()
median_nfp = nfp_data.median()
std_dev_nfp = nfp_data.std()
percentile_25th = nfp_data.quantile(0.25)
percentile_50th = nfp_data.quantile(0.50)  # same as median
percentile_75th = nfp_data.quantile(0.75)

# Dynamic Threshold Setting (your existing script)
std_dev_band_1 = (mean_nfp - std_dev_nfp, mean_nfp + std_dev_nfp)
std_dev_band_2 = (mean_nfp - 2*std_dev_nfp, mean_nfp + 2*std_dev_nfp)
std_dev_band_3 = (mean_nfp - 3*std_dev_nfp, mean_nfp + 3*std_dev_nfp)

top_10th_percentile = nfp_data.quantile(0.90)
bottom_10th_percentile = nfp_data.quantile(0.10)

# Nuanced Interpretation - Trend Analysis (your existing script)
nfp_moving_average_3m = nfp_data.rolling(window=3).mean()
nfp_moving_average_6m = nfp_data.rolling(window=6).mean()
latest_nfp = nfp_data.iloc[-1]

# Composite NFP Score Calculation

# Normalize components
def normalize_value(x, mean, std_dev):
    return norm.cdf((x - mean) / std_dev)

# Calculate component scores
score_mean = normalize_value(mean_nfp, mean_nfp, std_dev_nfp)
score_median = normalize_value(median_nfp, mean_nfp, std_dev_nfp)
score_latest_nfp = normalize_value(latest_nfp, mean_nfp, std_dev_nfp)
score_std_dev = 1 - normalize_value(std_dev_nfp, std_dev_nfp, nfp_data.std())

# Assuming the distribution of nfp_data is known and percentiles are calculated
n = len(nfp_data)
rank_latest_nfp = nfp_data.rank().iloc[-1]
score_percentile_latest_nfp = rank_latest_nfp / n

# Score extremes (using a simplified method here for demonstration)
score_extreme = np.exp(-abs(norm.ppf(score_percentile_latest_nfp)))

# Weights (example weights, adjust based on your analysis)
weights = {'mean': 0.2, 'median': 0.1, 'latest_nfp': 0.4, 'std_dev': 0.1, 'percentile_latest_nfp': 0.2}

# Calculate weighted composite score
composite_score = sum([
    score_mean * weights['mean'],
    score_median * weights['median'],
    score_latest_nfp * weights['latest_nfp'],
    score_std_dev * weights['std_dev'],
    score_percentile_latest_nfp * weights['percentile_latest_nfp']
])

# Normalize composite score to be between 0 and 1
Nonfarmpayrolls_composite_score_normalized = (composite_score - 0) / (1 - 0)

#print(f"Composite non farm payrolls Score: {Nonfarmpayrolls_composite_score_normalized:.2f}")



######################################################
######################################################
######################################################

# Fetch Initial Jobless Claims data
initial_jobless_claims = fred.get_series('ICSA')  # 'ICSA' is the series ID for Initial Jobless Claims

# Historical Data Analysis: Fetch several years of data, as comprehensive as possible
data = pd.DataFrame(initial_jobless_claims, columns=['Initial_Jobless_Claims'])

# Calculating the Moving Average: 52-week moving average
data['52_Week_MA'] = data['Initial_Jobless_Claims'].rolling(window=52).mean()

# Standard Deviation and Volatility: Standard deviation around the 52-week MA
data['52_Week_STD'] = data['Initial_Jobless_Claims'].rolling(window=52).std()

# Percentile Ranges: Establishing 25th, 50th, 75th percentiles
percentiles = data['Initial_Jobless_Claims'].quantile([0.25, 0.5, 0.75]).to_dict()

# Rate of Change: Calculating week-over-week change to assess acceleration/deceleration
data['Weekly_Change'] = data['Initial_Jobless_Claims'].diff()

# Latest data point
latest_value = data['Initial_Jobless_Claims'].iloc[-1]
latest_ma = data['52_Week_MA'].iloc[-1]
latest_std = data['52_Week_STD'].iloc[-1]
latest_change = data['Weekly_Change'].iloc[-1]





# Assess significant week-over-week change
significant_change_threshold = 10  # Placeholder threshold, adjust based on analysis


# For illustration, let's create a dummy DataFrame
data = pd.DataFrame({'Initial_Jobless_Claims': np.random.randint(180000, 250000, 100)})

data['52_Week_MA'] = data['Initial_Jobless_Claims'].rolling(window=52).mean()
data['52_Week_STD'] = data['Initial_Jobless_Claims'].rolling(window=52).std()
percentiles = data['Initial_Jobless_Claims'].quantile([0.25, 0.5, 0.75]).to_dict()
data['Weekly_Change'] = data['Initial_Jobless_Claims'].diff()

# Getting the latest values
latest_value = data['Initial_Jobless_Claims'].iloc[-1]
latest_ma = data['52_Week_MA'].iloc[-1]
latest_std = data['52_Week_STD'].iloc[-1]
latest_change = data['Weekly_Change'].iloc[-1]

# Normalization function
def normalize(value, min_val, max_val, inverse=False):
    if max_val == min_val:  # Prevent division by zero
        return 0
    normalized = (value - min_val) / (max_val - min_val)
    return 1 - normalized if inverse else normalized

# Define min and max values for normalization, considering the economic context
max_claim = data['Initial_Jobless_Claims'].max()
min_claim = data['Initial_Jobless_Claims'].min()

max_std = data['52_Week_STD'].max()

# Assuming changes can be both positive and negative, find the absolute max for normalization
max_change = max(abs(data['Weekly_Change'].min()), abs(data['Weekly_Change'].max()))

# Assigning weights
weights = {
    'L': 0.3,
    'MA': 0.25,
    'SD': 0.15,
    'WoWC': 0.3,  # Adjusted weight for WoWC to ensure total weights sum to 1.0
}

# Normalizing scores
scores = {}
scores['L'] = normalize(latest_value, min_claim, max_claim, inverse=True)
scores['MA'] = normalize(latest_ma, min_claim, max_claim, inverse=True)
scores['SD'] = normalize(latest_std, 0, max_std, inverse=True)

# For WoWC, we normalize such that decreases in claims increase the score
scores['WoWC'] = normalize(abs(latest_change), 0, max_change, inverse=latest_change < 0)

# Calculating the composite score with corrected weights and normalization
InitialJoblessClaimscomposite_score = sum(weights[k] * scores[k] for k in weights)

#print(f"Composite Initial Jobless Claims Score: {InitialJoblessClaimscomposite_score:.4f}")
    


################################################################
################################################################
################################################################

# Fetch historical oil prices (use the appropriate series ID for oil prices from FRED)
oil_prices = fred.get_series('DCOILWTICO')  # Example: 'DCOILWTICO' might be the series ID for Crude Oil Prices: West Texas Intermediate (WTI)

# Historical Data Analysis
# Assuming oil_prices is a pandas Series with datetime index

# 1. Calculate Statistical Metrics
mean_price = oil_prices.mean()
std_deviation = oil_prices.std()
percentiles = oil_prices.quantile([0.1, 0.5, 0.9])  # 10th, 50th, 90th percentiles

# 2. Moving Averages
moving_average_24 = oil_prices.rolling(window=50).mean()
moving_average_99 = oil_prices.rolling(window=99).mean()

# 3. Dynamic Thresholds Based on Volatility
upper_band = moving_average_24 + (std_deviation * 2)
lower_band = moving_average_24 - (std_deviation * 2)



current_price = oil_prices[-1]


# 1. Calculate Statistical Metrics
mean_price = oil_prices.mean()
std_deviation = oil_prices.std()
percentiles = oil_prices.quantile([0.1, 0.5, 0.9])  # 10th, 50th, 90th percentiles

# 2. Moving Averages
moving_average_24 = oil_prices.rolling(window=50).mean()
moving_average_99 = oil_prices.rolling(window=99).mean()

# 3. Dynamic Thresholds Based on Volatility
upper_band = moving_average_24 + (std_deviation * 2)
lower_band = moving_average_24 - (std_deviation * 2)

current_price = oil_prices[-1]

# Composite Score Calculation
# Weights
w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 0.2, 0.2  # Example weights, adjust based on your analysis

# N(P) - Normalization of Current Price within Percentiles
if current_price <= percentiles[0.1]:
    NP = 0
elif current_price <= percentiles[0.5]:
    NP = (current_price - percentiles[0.1]) / (percentiles[0.5] - percentiles[0.1]) * 0.5
elif current_price <= percentiles[0.9]:
    NP = 0.5 + (current_price - percentiles[0.5]) / (percentiles[0.9] - percentiles[0.5]) * 0.5
else:
    NP = 1

# N(T) - Normalization Relative to Mean and Standard Deviation
Z_score = (current_price - mean_price) / std_deviation
# Assuming Z-scores between -2 and 2 are considered normal
NT = (min(max(Z_score, -2), 2) + 2) / 4

# N(V) - Normalization of Volatility (Placeholder)
NV = 0.5  # Placeholder value, replace with your volatility calculation

# D(Trend) - Directional Factor for Trend (Placeholder)
DTrend = 0.5  # Placeholder value, replace based on trend direction: 1 for rising, 0.5 for stable, 0 for falling

# N(Volatility) - Normalization within Volatility Bands
if current_price >= lower_band[-1] and current_price <= upper_band[-1]:
    NVolatility = 1
else:
    NVolatility = 0  # Simplified approach; consider a more nuanced calculation

# Final Composite Score Calculation
OilPrices_composite_score = w1 * NP + w2 * NT + w3 * NV + w4 * DTrend + w5 * NVolatility

#print(f"Composite Score for Oil Prices: {OilPrices_composite_score}")




###########################################################
###########################################################
###########################################################

# Fetch Case Shiller Home Price Index data
cshpi = fred.get_series('CSUSHPISA')  # Make sure to use the correct series ID for CSHPI

# Convert to DataFrame for easier manipulation
cshpi_df = pd.DataFrame(cshpi, columns=['CSHPI'])

# 1. Historical Data Analysis
# Assuming historical data is already fetched into cshpi_df

# 2. Statistical Metrics Calculation
mean_cshpi = cshpi_df['CSHPI'].mean()
std_dev_cshpi = cshpi_df['CSHPI'].std()
percentiles = cshpi_df['CSHPI'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])

# 3. Dynamic Thresholds
# Year-over-Year Growth Rate
cshpi_df['YoY_Growth'] = cshpi_df['CSHPI'].pct_change(periods=12) * 100  # 12 months for annual growth rate
mean_growth = cshpi_df['YoY_Growth'].mean()
std_dev_growth = cshpi_df['YoY_Growth'].std()

# Current year-over-year growth rate
current_growth = cshpi_df['YoY_Growth'].iloc[-1]

# Volatility (using rolling standard deviation as a proxy)
cshpi_df['Volatility'] = cshpi_df['CSHPI'].rolling(window=12).std()

# Relative Position (latest value's percentile)
latest_value = cshpi_df['CSHPI'].iloc[-1]
relative_position = pd.qcut(cshpi_df['CSHPI'], q=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], labels=False, duplicates='drop').iloc[-1]

# 4. Trend Analysis (example with moving average)
cshpi_df['MA_12'] = cshpi_df['CSHPI'].rolling(window=12).mean()  # 12-month moving average
cshpi_df['MA_60'] = cshpi_df['CSHPI'].rolling(window=60).mean()  # 60-month moving average

# 5. Adjustment for Inflation
# Fetch CPI data for inflation adjustment
cpi = fred.get_series('CPIAUCSL')
cpi_df = pd.DataFrame(cpi, columns=['CPI']).resample('M').last()  # Resample CPI to monthly, if not already
cshpi_df = cshpi_df.merge(cpi_df, left_index=True, right_index=True, how='left')

# Adjust CSHPI for inflation (Base year can be chosen as needed, here using latest CPI as base)
cshpi_df['CSHPI_Inflation_Adjusted'] = (cshpi_df['CSHPI'] / cshpi_df['CPI']) * cshpi_df['CPI'].iloc[-1]

# Helper function for sigmoid normalization
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 1. Relative Position Score (RPS)
percentile_10th = cshpi_df['CSHPI'].quantile(0.1)
percentile_90th = cshpi_df['CSHPI'].quantile(0.9)
latest_cshpi_value = cshpi_df['CSHPI'].iloc[-1]
RPS = (latest_cshpi_value - percentile_10th) / (percentile_90th - percentile_10th)
RPS = np.clip(RPS, 0, 1)  # Ensure RPS is within 0-1

# 2. Growth Score (GS)
mean_growth = cshpi_df['YoY_Growth'].mean()
std_dev_growth = cshpi_df['YoY_Growth'].std()
current_growth = cshpi_df['YoY_Growth'].iloc[-1]
GS = sigmoid((current_growth - mean_growth) / std_dev_growth)

# 3. Volatility Score (VS)
mean_cshpi = cshpi_df['CSHPI'].mean()
std_dev_cshpi = cshpi_df['CSHPI'].std()
VS = sigmoid((latest_cshpi_value - mean_cshpi) / std_dev_cshpi)

# 4. Current Value Score (CVS)
CVS = (latest_cshpi_value - percentile_10th) / (percentile_90th - percentile_10th)
CVS = np.clip(CVS, 0, 1)  # Ensure CVS is within 0-1

# Composite Score Calculation
# Assign your weights based on analysis importance
W_RPS = 0.25
W_GS = 0.25
W_VS = 0.25
W_CVS = 0.25

CaseShillerHomePrices_CompositeScore = (RPS * W_RPS + GS * W_GS + VS * W_VS + CVS * W_CVS) / (W_RPS + W_GS + W_VS + W_CVS)

#print(f"Case Shiller Home Prices Composite Score: {CaseShillerHomePrices_CompositeScore}")




################################################################
################################################################
################################################################


# Fetch REIT performance data. Replace 'REIT_SERIES_ID' with the actual series ID for REITs from FRED
reit_data = fred.get_series('WILLREITIND')

# Ensure reit_data contains only numeric values
reit_data = pd.to_numeric(reit_data, errors='coerce').dropna()

# 1. Historical Data Analysis
# Averages
simple_moving_average_5y = reit_data.rolling(window=5*12).mean()  # 5-year simple moving average
exponential_moving_average_5y = reit_data.ewm(span=5*12).mean()  # 5-year exponential moving average
simple_moving_average_10y = reit_data.rolling(window=10*12).mean()  # 10-year simple moving average
exponential_moving_average_10y = reit_data.ewm(span=10*12).mean()  # 10-year exponential moving average

# Percentiles
top_10th_percentile = reit_data.quantile(0.9)
bottom_10th_percentile = reit_data.quantile(0.1)

# Standard Deviations
standard_deviation = reit_data.std()

# 2. Economic and Market Factors
# Fetching inflation data. Replace 'INFLATION_SERIES_ID' with the actual series ID for inflation rates from FRED
inflation_data = fred.get_series('CPIAUCSL')
# Adjust inflation data dates to match REIT data dates
inflation_data_aligned = inflation_data.reindex(reit_data.index).ffill().bfill()

# 3. Methodology for Setting Dynamic Thresholds
# Rolling Analysis for standard deviation as an example
rolling_std = reit_data.rolling(window=12).std()

# Adaptive Z-Scores
historical_mean = reit_data.mean()
historical_std = reit_data.std()
z_scores = (reit_data - historical_mean) / historical_std

# Calculate skewness and kurtosis for REIT data
reit_skewness = skew(reit_data)
reit_kurtosis = kurtosis(reit_data)


# Normalization Functions
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

# 1. Normalize the Components
# Calculate the latest values for normalization
latest_sma_5y = simple_moving_average_5y.iloc[-1]
latest_ema_5y = exponential_moving_average_5y.iloc[-1]
latest_sma_10y = simple_moving_average_10y.iloc[-1]
latest_ema_10y = exponential_moving_average_10y.iloc[-1]

# Normalized Moving Averages (MA)
norm_ma = normalize(max(latest_sma_5y, latest_ema_5y), reit_data.min(), reit_data.max())

# Normalized Percentile Returns
norm_percentile_returns = normalize(top_10th_percentile, bottom_10th_percentile, top_10th_percentile)

# Normalized Volatility Measures
norm_volatility = 1 - normalize(standard_deviation, reit_data.std().min(), reit_data.std().max())

# Normalized Z-Score (using latest z-score)
latest_z_score = z_scores.iloc[-1]
norm_z_score = normalize(latest_z_score, z_scores.min(), z_scores.max())

# Normalized Distribution Characteristics (Skewness and Kurtosis)
norm_skewness = normalize(reit_skewness, skew(reit_data).min(), skew(reit_data).max())
norm_kurtosis = normalize(reit_kurtosis, kurtosis(reit_data).min(), kurtosis(reit_data).max())

# 2. Assign Weights
weights = {
    'ma': 0.25,
    'percentile_returns': 0.20,
    'volatility': 0.20,
    'z_score': 0.20,
    'distribution_characteristics': 0.15  # Assuming equal weight for skewness and kurtosis
}

# 3. Calculate the Composite Score
REIT_composite_score = (
    weights['ma'] * norm_ma +
    weights['percentile_returns'] * norm_percentile_returns +
    weights['volatility'] * norm_volatility +
    weights['z_score'] * norm_z_score +
    weights['distribution_characteristics'] * (norm_skewness + norm_kurtosis) / 2
)

#print(f"Composite REIT Score: {REIT_composite_score}")



################################################################
################################################################
################################################################


# Fetch DXY data
dxy_data = fred.get_series('DTWEXBGS')  # Replace 'DXY_SERIES_ID' with the actual ID

# Ensure data is numeric and handle NaN values
dxy_data = pd.to_numeric(dxy_data, errors='coerce').dropna()

# Create DataFrame from series
dxy_df = pd.DataFrame(dxy_data, columns=['DXY'])

# Historical Averages
dxy_df['50_day_MA'] = dxy_df['DXY'].rolling(window=50).mean()
dxy_df['200_day_MA'] = dxy_df['DXY'].rolling(window=200).mean()

# Percentiles
dxy_df['Percentile_Rank'] = dxy_df['DXY'].apply(lambda x: percentileofscore(dxy_df['DXY'], x))

# Standard Deviations
dxy_df['Annual_Volatility'] = dxy_df['DXY'].pct_change().rolling(window=252).std() * np.sqrt(252)

# Rate of Change
dxy_df['Momentum_1M'] = dxy_df['DXY'].pct_change(periods=30)
dxy_df['Momentum_3M'] = dxy_df['DXY'].pct_change(periods=90)
dxy_df['Momentum_6M'] = dxy_df['DXY'].pct_change(periods=180)

# Threshold Adjustments
long_term_avg = dxy_df['200_day_MA'].iloc[-1]
long_term_std = dxy_df['Annual_Volatility'].iloc[-1]

# Dynamic Bullish Threshold
bullish_threshold = long_term_avg + long_term_std
# Dynamic Bearish Threshold
bearish_threshold = long_term_avg - long_term_std

# Scores based on conditions
dxy_latest = dxy_df.iloc[-1]  # Latest row of the DataFrame

# MAs Comparison
ma_comparison_score = 1 if dxy_latest['50_day_MA'] > dxy_latest['200_day_MA'] else 0

# Percentile Rank - Assuming direct normalization based on the last known percentile rank
percentile_rank_score = dxy_latest['Percentile_Rank'] / 100

# Annual Volatility - Inverse normalization
annual_volatility_score = 1 - dxy_latest['Annual_Volatility']

# Momentum Indicators - Normalized based on sign
momentum_1m_score = 1 if dxy_latest['Momentum_1M'] > 0 else 0
momentum_3m_score = 1 if dxy_latest['Momentum_3M'] > 0 else 0
momentum_6m_score = 1 if dxy_latest['Momentum_6M'] > 0 else 0
combined_momentum_score = (momentum_1m_score + momentum_3m_score + momentum_6m_score) / 3

# Position Relative to Thresholds
if dxy_latest['DXY'] > bullish_threshold:
    position_score = 1
elif dxy_latest['DXY'] < bearish_threshold:
    position_score = 0
else:
    position_score = 0.5

# Weights
weights = {
    'ma_comparison': 0.20,
    'percentile_rank': 0.20,
    'annual_volatility': 0.15,
    'combined_momentum': 0.30,
    'position_thresholds': 0.15,
}

# Composite Score Calculation
DXY_composite_score = (
    ma_comparison_score * weights['ma_comparison'] +
    percentile_rank_score * weights['percentile_rank'] +
    annual_volatility_score * weights['annual_volatility'] +
    combined_momentum_score * weights['combined_momentum'] +
    position_score * weights['position_thresholds']
)

# Printing the composite DXY score
#print(f"Composite DXY Score: {DXY_composite_score}")



################################################################
################################################################
################################################################

# 1. Historical Data Analysis
# Data Collection
usslind_series = fred.get_series('USSLIND')

# Trend Analysis
usslind_trend = usslind_series.rolling(window=12, center=True).mean()

# 2. Statistical Metrics Calculation
# Moving Averages
usslind_12m_ma = usslind_series.rolling(window=12).mean()
usslind_24m_ma = usslind_series.rolling(window=24).mean()

# Standard Deviation
usslind_12m_std = usslind_series.rolling(window=12).std()
usslind_24m_std = usslind_series.rolling(window=24).std()

# Percentiles
percentiles = [25, 50, 75]
usslind_percentiles = np.percentile(usslind_series.dropna(), percentiles)

# 3. Dynamic Thresholds
# Adaptive Baselines
# Using the moving averages as baselines

# Volatility Bands
upper_band_12m = usslind_12m_ma + (usslind_12m_std * 2)
lower_band_12m = usslind_12m_ma - (usslind_12m_std * 2)
upper_band_24m = usslind_24m_ma + (usslind_24m_std * 2)
lower_band_24m = usslind_24m_ma - (usslind_24m_std * 2)

# Percentile Ranges
# Utilizing calculated percentiles

# 4. Acceleration and Deceleration
# Identification based on moving averages and standard deviation bands
latest_usslind_value = usslind_series.iloc[-1]

# Initialize weights for each component
# NOTE: You'll need to define these weights based on your analysis
weights = {
    'latest_value_adjustment': 0.2,
    'moving_averages_adjustment': 0.2,
    'standard_deviation_adjustment': 0.2,
    'percentile_based_adjustment': 0.2,
    'volatility_bands_adjustment': 0.2,
}


# Step 1 & 2: Component-Specific Adjustments and Normalization
# Normalize the latest USSLIND value based on its position within the historical percentile range
latest_value_score = (latest_usslind_value - usslind_series.min()) / (usslind_series.max() - usslind_series.min())

# Calculate and normalize deviations from moving averages
deviation_12m_score = (latest_usslind_value - usslind_12m_ma.iloc[-1]) / usslind_12m_std.iloc[-1]
deviation_24m_score = (latest_usslind_value - usslind_24m_ma.iloc[-1]) / usslind_24m_std.iloc[-1]

# Normalize standard deviations
std_12m_score = (usslind_12m_std.iloc[-1] - usslind_12m_std.min()) / (usslind_12m_std.max() - usslind_12m_std.min())
std_24m_score = (usslind_24m_std.iloc[-1] - usslind_24m_std.min()) / (usslind_24m_std.max() - usslind_24m_std.min())

# Normalize percentile position
percentile_score = np.sum([np.searchsorted(usslind_percentiles, latest_usslind_value) / len(usslind_percentiles)])

# Normalize volatility band position
volatility_band_score = (latest_usslind_value - lower_band_12m.iloc[-1]) / (upper_band_12m.iloc[-1] - lower_band_12m.iloc[-1])

# Step 3: Weighted Aggregation
# Aggregate the scores with their respective weights
component_scores = np.array([
    latest_value_score,
    np.mean([deviation_12m_score, deviation_24m_score]),  # Averaging the deviation scores for simplicity
    np.mean([std_12m_score, std_24m_score]),  # Averaging the std deviation scores
    percentile_score,
    volatility_band_score
])

weights_array = np.array(list(weights.values()))

composite_score = np.dot(component_scores, weights_array)

# Step 4: Final Normalization
# Ensure composite score is between 0 and 1
leadingIndex_composite_score_normalized = (composite_score - np.min(component_scores)) / (np.max(component_scores) - np.min(component_scores))

#print(f"Leading Index Composite Score: {leadingIndex_composite_score_normalized}")




################################################################
################################################################
################################################################

composite_scores = {
    "GDPCompositeScore": GDPCompositeScore,
    "Inflation_CS": Inflation_CS,
    "Unemployment_final_composite_score": Unemployment_final_composite_score,
    "Interest_Rate_FinalCompositeScore": Interest_Rate_FinalCompositeScore,
    "CCI_composite_score": CCI_composite_score,
    "InudstrialProd_FinalCompositeScore": InudstrialProd_FinalCompositeScore,
    "housing_starts_composite_score": housing_starts_composite_score,
    "Retail_Sales_composite_score": Retail_Sales_composite_score,
    "SP500_composite_score": SP500_composite_score,
    "Building_Permits_Composite_Score": Building_Permits_Composite_Score,
    "NOMG_Composite_Score": NOMG_Composite_Score,
    "YieldCurve_Final_Composite_Score": YieldCurve_Final_Composite_Score,
    "FFRv10Yr_composite_score": FFRv10Yr_composite_score,
    "VIX_CompositeScore": VIX_CompositeScore,
    "credit_spread_composite_score": credit_spread_composite_score,
    "hybpcomposite_score": hybpcomposite_score,
    "TED_final_composite_score": TED_final_composite_score,
    "commercial_paper_composite_score": commercial_paper_composite_score,
    "Nonfarmpayrolls_composite_score_normalized": Nonfarmpayrolls_composite_score_normalized,
    "InitialJoblessClaimscomposite_score": InitialJoblessClaimscomposite_score,
    "OilPrices_composite_score": OilPrices_composite_score,
    "CaseShillerHomePrices_CompositeScore": CaseShillerHomePrices_CompositeScore,
    "REIT_composite_score": REIT_composite_score,
    "DXY_composite_score": DXY_composite_score,
    "leadingIndex_composite_score_normalized": leadingIndex_composite_score_normalized,
}




totalcompositescoreweightings = {
    "GDPCompositeScore": 0.9,
    "Inflation_CS": 0.9,
    "Unemployment_final_composite_score": 0.9,
    "Interest_Rate_FinalCompositeScore": 0.07,
    "CCI_composite_score": 0.06,
    "InudstrialProd_FinalCompositeScore": 0.05,
    "housing_starts_composite_score": 0.05,
    "Retail_Sales_composite_score": 0.05,
    "SP500_composite_score": 0.04,
    "Building_Permits_Composite_Score": 0.04,
    "NOMG_Composite_Score": 0.04,
    "YieldCurve_Final_Composite_Score": 0.04,
    "FFRv10Yr_composite_score": 0.03,
    "VIX_CompositeScore": 0.03,
    "credit_spread_composite_score": 0.03,
    "hybpcomposite_score": 0.02,
    "TED_final_composite_score": 0.02,
    "commercial_paper_composite_score": 0.02,
    "Nonfarmpayrolls_composite_score_normalized": 0.02,
    "InitialJoblessClaimscomposite_score": 0.02,
    "OilPrices_composite_score": 0.02,
    "CaseShillerHomePrices_CompositeScore": 0.02,
    "REIT_composite_score": 0.02,
    "DXY_composite_score": 0.02,
    "leadingIndex_composite_score_normalized": 0.02
}

# Assuming f(S_i) = S_i (i.e., no transformation) and no significant interaction pairs for simplicity
# Compute the Final Composite Score (FCS)
FCS = sum(totalcompositescoreweightings[metric] * composite_scores[metric] for metric in composite_scores) / sum(totalcompositescoreweightings.values())

print(f"Total Macro Score: {FCS}")

def determine_business_cycle_phase(FCS, composite_scores):
    # Expansion Phase Criteria
    if 0.75 <= FCS <= 1.00:
        if all([
            composite_scores['GDPCompositeScore'] > 0.5,
            composite_scores['InudstrialProd_FinalCompositeScore'] > 0.5,
            composite_scores['Retail_Sales_composite_score'] > 0.5,
            composite_scores['housing_starts_composite_score'] > 0.5,
            composite_scores['Unemployment_final_composite_score'] < 0.5,
            composite_scores['SP500_composite_score'] > 0.5,
            composite_scores['VIX_CompositeScore'] < 0.5
        ]):
            return "Expansion"
    
    # Peak Phase Criteria
    elif 0.60 < FCS <= 0.75:
        if all([
            0.4 < composite_scores['Inflation_CS'] <= 0.9,
            0.4 < composite_scores['Interest_Rate_FinalCompositeScore'] <= 0.9,
            composite_scores['credit_spread_composite_score'] > 0.2
        ]):
            return "Peak"
    
    # Contraction (Recession) Phase Criteria
    elif 0.25 < FCS <= 0.60:
        if all([
            composite_scores['Unemployment_final_composite_score'] > 0.5,
            composite_scores['GDPCompositeScore'] < 0.6,
            composite_scores['InudstrialProd_FinalCompositeScore'] < 0.6,
            composite_scores['Retail_Sales_composite_score'] < 0.6,
            composite_scores['VIX_CompositeScore'] > 0.3,
            composite_scores['SP500_composite_score'] < 0.7
        ]):
            return "Contraction"
    
    # Trough Phase Criteria
    elif 0 <= FCS <= 0.25:
        if all([
            composite_scores['GDPCompositeScore'] < 0.4,
            composite_scores['InudstrialProd_FinalCompositeScore'] < 0.4,
            composite_scores['Retail_Sales_composite_score'] < 0.4,
            composite_scores['Unemployment_final_composite_score'] > 0.6,
            composite_scores['CCI_composite_score'] < 0.4
        ]):
            return "Trough"
    
    # If no specific phase criteria are met, return "Indeterminate"
    return "Indeterminate"

# Now, call the function with your FCS and composite_scores
business_cycle_phase = determine_business_cycle_phase(FCS, composite_scores)

# Print the determined business cycle phase
print("The current phase of the business cycle is:", business_cycle_phase)
