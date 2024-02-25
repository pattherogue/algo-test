import pandas as pd
from fredapi import Fred
import numpy as np
from scipy.stats import zscore
from scipy.stats import skew, kurtosis
from scipy.stats import percentileofscore


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

print(f"Current GDP Growth Rate: {current_gdp_growth_rate:.2f}% and it is {status}.")


# Fetch Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL) as an inflation rate indicator
cpi_data = fred.get_series('CPIAUCSL')

# Convert CPI data to pandas DataFrame for easier manipulation
cpi_df = pd.DataFrame(cpi_data, columns=['CPI'])

# Calculate inflation rate as the percentage change in CPI
cpi_df['Inflation_Rate'] = cpi_df['CPI'].pct_change() * 100

# Calculate 12-month moving average for the inflation rate
cpi_df['MA_12'] = cpi_df['Inflation_Rate'].rolling(window=12).mean()

# Calculate standard deviation of the inflation rate over a 12-month period
cpi_df['Std_Dev_12'] = cpi_df['Inflation_Rate'].rolling(window=12).std()

# Calculate key percentiles (25th, 50th, 75th) of the historical inflation rate data
percentiles = cpi_df['Inflation_Rate'].quantile([0.25, 0.5, 0.75])

# Define dynamic thresholds
low_inflation_threshold = percentiles[0.25] - cpi_df['Std_Dev_12']
moderate_inflation_threshold = percentiles[0.5]
high_inflation_threshold = percentiles[0.75] + cpi_df['Std_Dev_12']

# Get the latest inflation rate and its rate of change
latest_inflation_rate = cpi_df['Inflation_Rate'].iloc[-1]
rate_of_change = cpi_df['Inflation_Rate'].diff().iloc[-1]

# Output the required values
print(f"Latest Inflation Rate: {latest_inflation_rate:.2f}%")
print(f"Rate of Change in Inflation Rate: {rate_of_change:.2f}%")
print(f"Low Inflation Threshold: {low_inflation_threshold.iloc[-1]:.2f}%")
print(f"Moderate Inflation Threshold: {moderate_inflation_threshold:.2f}%")
print(f"High Inflation Threshold: {high_inflation_threshold.iloc[-1]:.2f}%")

#Fetch historical unemployment rate data
unemployment_rate = fred.get_series('UNRATE')  # 'UNRATE' is the series ID for Civilian Unemployment Rate

# Calculate statistical metrics
mean_unemployment_rate = np.mean(unemployment_rate)
std_dev_unemployment_rate = np.std(unemployment_rate)
percentiles_unemployment_rate = np.percentile(unemployment_rate, [25, 50, 75, 90])

# Dynamic thresholds determination
normal_range_low = np.percentile(unemployment_rate, 25)
normal_range_high = np.percentile(unemployment_rate, 75)
elevated_unemployment_threshold = np.percentile(unemployment_rate, 75) + std_dev_unemployment_rate
low_unemployment_threshold = np.percentile(unemployment_rate, 25) - std_dev_unemployment_rate

# Output the calculated metrics
print(f"Mean Unemployment Rate: {mean_unemployment_rate}")
print(f"Standard Deviation of Unemployment Rate: {std_dev_unemployment_rate}")
print(f"Percentiles (25th, 50th, 75th, 90th): {percentiles_unemployment_rate}")
print(f"Normal Unemployment Rate Range: {normal_range_low} to {normal_range_high}")
print(f"Elevated Unemployment Threshold: {elevated_unemployment_threshold}")
print(f"Low Unemployment Threshold: {low_unemployment_threshold}")

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

# Print results
print(f"Current Interest Rate: {current_interest_rate}")
print(f"Current Trend (12M Moving Average): {current_trend}")
print(f"Current Interest Rate Environment: {current_environment}")

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
moderate_confidence_range = (mean_cci - std_dev_cci, mean_cci + std_dev_cci)
high_confidence_threshold = percentile_75_cci

# Adjusting for Economic Context
# Trend Analysis - Calculate 12-month Moving Average
cci_df['12M_MA'] = cci_df['CCI'].rolling(window=12).mean()

# Determine current CCI status
current_cci = cci_df['CCI'].iloc[-1]
current_status = ''

if current_cci < low_confidence_threshold:
    current_status = 'Low Confidence'
elif low_confidence_threshold <= current_cci <= high_confidence_threshold:
    current_status = 'Moderate Confidence'
else:
    current_status = 'High Confidence'

print(f"Current CCI: {current_cci}")
print(f"Current Consumer Confidence Status: {current_status}")

# Print statistical metrics for reference
print(f"Mean CCI: {mean_cci}")
print(f"Median CCI: {median_cci}")
print(f"Standard Deviation: {std_dev_cci}")
print(f"25th Percentile: {percentile_25_cci}")
print(f"75th Percentile: {percentile_75_cci}")
print(f"90th Percentile: {percentile_90_cci}")
print(f"Low Confidence Threshold: {low_confidence_threshold}")
print(f"Moderate Confidence Range: {moderate_confidence_range}")
print(f"High Confidence Threshold: {high_confidence_threshold}")

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

# Print descriptive statistics
print(f"Mean (Average) Ind. Prod.: {mean_ip}")
print(f"Standard Deviation of Ind. Prod.: {std_dev_ip}")
print(f"25th Percentile of Ind. Prod.: {percentile_25th}")
print(f"Median (50th Percentile) of Ind. Prod: {median_ip}")
print(f"75th Percentile of Ind. Prod: {percentile_75th}")

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

print(f"Current Ind. Prod. Status: {current_status}")


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
current_mom_growth = housing_starts_filtered['MoM_Growth'].iloc[-1]
current_volatility = housing_starts_filtered['Volatility'].iloc[-1]

print(f"Current Housing Starts: {current_housing_starts}")
print(f"Housing Starts Month-over-Month Growth Rate: {current_mom_growth}%")
print(f"Housing Starts Volatility: {current_volatility}")

# Fetch Retail Sales data (ensure you use the correct series ID for seasonally adjusted Retail Sales)
series_id = 'RSXFS'  # Example series ID for Retail Sales
retail_sales = fred.get_series(series_id, observation_start='2010-01-01')

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

# Print the most recent Retail Sales data
print("Most Recent Retail Sales:", retail_sales.iloc[-1])

# Moving Averages Comparison
print("\nMoving Averages Comparison:")
print(f"3-month MA: {retail_sales_3m.iloc[-1]}, 6-month MA: {retail_sales_6m.iloc[-1]}")
print(f"12-month MA: {retail_sales_12m.iloc[-1]}, 24-month MA: {retail_sales_24m.iloc[-1]}")

# Compare short-term and long-term moving averages to detect shifts
if retail_sales_3m.iloc[-1] > retail_sales_12m.iloc[-1]:
    print("Short-term consumer spending is increasing relative to the long-term trend.")
else:
    print("Short-term consumer spending is decreasing or stable relative to the long-term trend.")

# Current Volatility Levels
print("\nCurrent Volatility Levels:")
print(f"Standard Deviation of Retail Sales Growth Rates: {retail_sales_std_dev}")
print(f"Upper Volatility Band: {upper_band}, Lower Volatility Band: {lower_band}")

# Check if the current growth rate is considered significantly positive or negative
if current_growth_rate > upper_band:
    print("Current Retail Sales growth rate is significantly positive.")
elif current_growth_rate < lower_band:
    print("Current Retail Sales growth rate is significantly negative.")
else:
    print("Current Retail Sales growth rate is within normal volatility range.")

# Rate of Change in Retail Sales
print("\nRate of Change in Retail Sales Growth Rate (Trend Analysis):")
print(f"Most Recent Rate of Change: {rate_of_change.iloc[-1]}")

# Determine if the growth rate is accelerating or decelerating
if rate_of_change.iloc[-1] > 0:
    print("Retail Sales growth rate is accelerating.")
elif rate_of_change.iloc[-1] < 0:
    print("Retail Sales growth rate is decelerating.")
else:
    print("No significant change in Retail Sales growth rate.")

# Current Growth Rate Percentile
print("\nCurrent Growth Rate Percentile:", current_percentile)

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

# Print latest stock price
print("Latest S&P 500 Closing Price:", sp500_data['Closing Price'].iloc[-1])

# Trend Analysis Insights
print("\nTrend Analysis Insights:")
#print("Latest 50-day Moving Average:", sp500_data['50-day MA'].iloc[-1])
#print("Latest 200-day Moving Average:", sp500_data['200-day MA'].iloc[-1])
#print("Trend Indicator: Bullish" if sp500_data['50-day MA'].iloc[-1] > sp500_data['200-day MA'].iloc[-1] else "Trend Indicator: Bearish")

# Statistical Measures Insights
print("\nStatistical Measures Insights:")
print("Latest Annualized Volatility:", sp500_data['Volatility'].iloc[-1])
print("Latest 95th Percentile of Price Change:", sp500_data['95th Percentile'].iloc[-1])
print("Latest 5th Percentile of Price Change:", sp500_data['5th Percentile'].iloc[-1])
print("Distance from Long-term Mean:", sp500_data['Distance from Mean'].iloc[-1])

# Dynamic Threshold Adjustment Insights
# Assuming dynamic thresholds are recalculated annually or quarterly, you can summarize recent adjustments.
# This is an illustrative example since actual dynamic adjustment logic was not implemented.
print("\nDynamic Threshold Adjustment Insights:")
print("Recent Volatility Threshold:", recent_volatility)

# Fetch building permits data
building_permits_series = 'PERMIT'  # Example series ID; replace with actual series ID for building permits
building_permits_data = fred.get_series(building_permits_series)

# Historical Data Analysis
historical_period_years = 20  # Example: 20 years
historical_data = building_permits_data.last('20Y')  # Adjust based on the period of interest

# Historical Averages
long_term_average = historical_data.mean()
print(f'Long-term average (20 years) of building permits issued: {long_term_average}')

# Seasonal Adjustments (Example, implement actual seasonal adjustment calculation or method)
# Placeholder for seasonal adjustment factors or method
seasonally_adjusted_data = historical_data  # This should be replaced with actual seasonal adjustment calculation

# Volatility Assessment
standard_deviation = historical_data.std()
print(f'Standard deviation of building permits issued: {standard_deviation}')

# Trend Analysis
moving_average_period = 12  # Example: 12-month moving average
trend_data = historical_data.rolling(window=moving_average_period).mean()
print(f'Last 12-month moving average of building permits: {trend_data[-1]}')

# Dynamic Threshold Setting
# Percentiles for Extreme Values
lower_threshold = historical_data.quantile(0.1)
upper_threshold = historical_data.quantile(0.9)
print(f'Lower 10th percentile threshold: {lower_threshold}')
print(f'Upper 90th percentile threshold: {upper_threshold}')

# Standard Deviation Bands
upper_band = long_term_average + standard_deviation
lower_band = long_term_average - standard_deviation
print(f'Upper standard deviation band: {upper_band}')
print(f'Lower standard deviation band: {lower_band}')

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
print(f'Current building permits issued: {current_building_permits}')

# Assessing current data against thresholds
if current_building_permits > adjusted_upper_band:
    print('Current building permits indicate above-average activity.')
elif current_building_permits < adjusted_lower_band:
    print('Current building permits indicate below-average activity.')
else:
    print('Current building permits are within normal range.')
    

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

# Threshold Implementation
# Dynamic Evaluation
if current_nomg > percentiles['75th']:
    print("Current NOMG is above the 75th percentile, indicating strong manufacturing sector demand.")
elif current_nomg < percentiles['25th']:
    print("Current NOMG is below the 25th percentile, indicating weak manufacturing sector demand.")
else:
    print("Current NOMG is within the 25th and 75th percentiles, indicating normal manufacturing sector demand.")

# Trend Analysis
# Assuming the use of the most recent moving average value for trend analysis
recent_ma = nomg_data_ma.iloc[-1]
if current_nomg > recent_ma:
    print("Current NOMG value is above its recent moving average, indicating an acceleration in manufacturing orders.")
elif current_nomg < recent_ma:
    print("Current NOMG value is below its recent moving average, indicating a deceleration in manufacturing orders.")
else:
    print("Current NOMG value is approximately equal to its recent moving average, indicating stability in manufacturing orders.")

print(f"Current NOMG Value: {current_nomg}")
print(f"NOMG Recent Moving Average: {recent_ma}")
print(f"NOMG Standard Deviation: {nomg_std.iloc[-1]}")
print(f"NOMG Percentiles: {percentiles}")

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

# Print statements for valuable insights
print(f"Yield Curve Average Spread: {average_spread}")
print(f"Yield Curve Standard Deviation: {std_dev_spread}")
print(f"Yield Curve 25th Percentile: {percentile_25th}, 50th Percentile: {percentile_50th}, 75th Percentile: {percentile_75th}")
print(f"Yield Curve Current Spread: {current_spread}, Signal Strength: {signal_strength}, Alert Level: {alert_level}")

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

# Print valuable insights
print(f"Fed Funds & 10Y Mean Spread: {mean_spread}")
print(f"Fed Funds & 10Y Median Spread: {median_spread}")
print(f"Fed Funds & 10Y Standard Deviation of Spread: {std_dev_spread}")
print(f"Fed Funds & 10Y 25th Percentile: {percentile_25}")
print(f"Fed Funds & 10Y 50th Percentile (Median): {percentile_50}")
print(f"Fed Funds & 10Y 75th Percentile: {percentile_75}")

# Dynamic Thresholds Setting based on statistical analysis
# Example thresholds, adjust based on further analysis or desired sensitivity
expansion_threshold = percentile_25
recession_warning_threshold = percentile_75
neutral_watch_zone_low = percentile_25
neutral_watch_zone_high = percentile_75

# Determine the current spread condition
current_spread = data['Spread'].iloc[-1]
print(f"Fed Funds & 10Y Current Spread: {current_spread}")

if current_spread <= expansion_threshold:
    print("Current condition suggests potential economic expansion.")
elif current_spread >= recession_warning_threshold:
    print("Current condition suggests potential recession warning.")
else:
    print("Current condition is in the neutral/watch zone.")
    
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

# Dynamic Thresholds Application Insights
print(f"Current VIX Level: {current_vix}")
print(f"10-Year Historical Average: {historical_avg_10y}")
print(f"20-Year Historical Average: {historical_avg_20y}")
print(f"10-Year Standard Deviation: {std_dev_10y}")
print(f"20-Year Standard Deviation: {std_dev_20y}")
print(f"Percentiles (25th, 50th, 75th, 90th): {vix_percentiles}")
print(f"Volatility Regime: {volatility_regime}")

# Assuming corp_bond_spreads and govt_bond_spreads are fetched correctly,
# Calculate the credit spread by subtracting government bond yields from corporate bond yields
credit_spreads = fred.get_series('BAA10Y')

# Calculate baseline metrics
average_credit_spread = np.mean(credit_spreads)
standard_deviation = np.std(credit_spreads)

# Print baseline metrics
print(f'Average Credit Spread: {average_credit_spread}')
print(f'Standard Deviation: {standard_deviation}')

# Determine the current credit spread
current_credit_spread = credit_spreads[-1]
print(f'Current Credit Spread: {current_credit_spread}')

# Fetch High Yield Bond Performance data
hybp_series_id = 'BAMLH0A0HYM2EY'  # Replace with the correct series ID for High Yield Bond Performance
hybp_data = fred.get_series(hybp_series_id)

# Convert to DataFrame for easier manipulation
hybp_df = pd.DataFrame(hybp_data, columns=['HYBP'])

# 1. Historical Averages
# Long-Term Average Performance
long_term_avg = hybp_df['HYBP'].mean()
print(f"Long-Term Average Performance: {long_term_avg}")

# Rolling Averages
hybp_df['12_month_MA'] = hybp_df['HYBP'].rolling(window=12).mean()
hybp_df['24_month_MA'] = hybp_df['HYBP'].rolling(window=24).mean()
print("Recent 12-month MA:", hybp_df['12_month_MA'].iloc[-1])
print("Recent 24-month MA:", hybp_df['24_month_MA'].iloc[-1])

# 2. Volatility (Standard Deviation)
std_dev = hybp_df['HYBP'].std()
print(f"Standard Deviation of HYBP: {std_dev}")

# Rolling Standard Deviation
hybp_df['Rolling_STD'] = hybp_df['HYBP'].rolling(window=12).std()
print("Recent Rolling Standard Deviation:", hybp_df['Rolling_STD'].iloc[-1])


# 4. Trend Analysis
# Assuming a simple moving average for trend analysis
hybp_df['Trend'] = hybp_df['HYBP'].rolling(window=12).mean()
print("Recent Trend (12-month MA):", hybp_df['Trend'].iloc[-1])

# 5. Risk-Adjusted Returns (Example: Sharpe Ratio)
# Assuming you have risk-free rate data (rfr) and calculating Sharpe Ratio as an example
ten_year_treasury_yield = fred.get_series('DGS10')  # 10-Year Treasury Constant Maturity Rate

risk_free_rate = ten_year_treasury_yield  # Placeholder for risk-free rate
returns = hybp_df['HYBP'].pct_change().dropna()
excess_returns = returns - risk_free_rate
sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio}")

# Fetch the three-month LIBOR rate
libor_3m = fred.get_series('LIOR3M', observation_start='1980-01-01')

# Fetch the three-month Treasury Bill rate
t_bill_3m = fred.get_series('DTB3', observation_start='1980-01-01')

# Calculate the TED Spread (difference between LIBOR and T-Bill rates)
ted_spread = libor_3m - t_bill_3m

# Analysis
# Long-term Averages
long_term_average = ted_spread.mean()
print(f"Long-term Average of the TED Spread: {long_term_average}")

# Volatility Measurement
volatility = ted_spread.std()
print(f"Volatility of the TED Spread: {volatility}")

# Percentiles
percentiles = np.percentile(ted_spread.dropna(), [25, 50, 75, 90])
print(f"25th Percentile: {percentiles[0]}")
print(f"50th Percentile (Median): {percentiles[1]}")
print(f"75th Percentile: {percentiles[2]}")
print(f"90th Percentile: {percentiles[3]}")

# Dynamic Thresholds Concept
normal_range_lower = long_term_average - volatility
normal_range_upper = long_term_average + volatility
print(f"Normal Range: {normal_range_lower} to {normal_range_upper}")

# Alert Zones
mild_concern_threshold = np.percentile(ted_spread.dropna(), 75)
high_concern_threshold = np.percentile(ted_spread.dropna(), 90)
print(f"Mild Concern Threshold: {mild_concern_threshold}")
print(f"High Concern Threshold: {high_concern_threshold}")

# Ensure both series cover the same date range
common_dates = libor_3m.dropna().index.intersection(t_bill_3m.dropna().index)
ted_spread = ted_spread[common_dates]

# Calculate the last non-NaN TED Spread
current_ted_spread = ted_spread.dropna().iloc[-1]

# Calculate the rolling average based on the last available data
rolling_average = ted_spread.dropna().rolling(window=30).mean().iloc[-1]

print(f"Current TED Spread: {current_ted_spread}")
print(f"30-Day Rolling Average of TED Spread: {rolling_average}")

# Adjust the trend analysis logic as necessary
if current_ted_spread > rolling_average:
    print("The TED Spread is currently widening, which might indicate developing risks.")
else:
    print("The TED Spread is currently narrowing or stable.")
    


# Fetch the data
commercial_paper_rate = fred.get_series('CPN3M')  # Placeholder ID for 3-month commercial paper rate
t_bill_rate = fred.get_series('TB3MS')  # Actual ID for 3-month T-bill rate

# Calculate Commercial Paper Spread (CPS)
cps = commercial_paper_rate - t_bill_rate

# Historical Data Analysis
print("Calculate Commercial Paper Spread (CPS) Historical Data Analysis")
mean_cps = cps.mean()
median_cps = cps.median()
std_dev_cps = cps.std()
percentile_25th = cps.quantile(0.25)
percentile_75th = cps.quantile(0.75)
percentile_90th = cps.quantile(0.90)

print(f"CPS Mean: {mean_cps}")
print(f"CPS Median: {median_cps}")
print(f"CPS Standard Deviation: {std_dev_cps}")
print(f"CPS25th Percentile: {percentile_25th}")
print(f"CPS75th Percentile: {percentile_75th}")
print(f"CPS90th Percentile: {percentile_90th}")

# Dynamic Thresholds Development
print("\nCPSDynamic Thresholds Development")
std_dev_band_1_upper = mean_cps + std_dev_cps
std_dev_band_1_lower = mean_cps - std_dev_cps
std_dev_band_2_upper = mean_cps + 2*std_dev_cps
std_dev_band_2_lower = mean_cps - 2*std_dev_cps

print(f"CPS +-1 SD Band: Lower = {std_dev_band_1_lower}, Upper = {std_dev_band_1_upper}")
print(f"CPS +-2 SD Band: Lower = {std_dev_band_2_lower}, Upper = {std_dev_band_2_upper}")

# Nuance and Context Incorporation
cps_volatility = cps.rolling(window=30).std().mean()  # 30-day moving average of standard deviation
volatility_adjusted_mean = mean_cps + cps_volatility
volatility_adjusted_median = median_cps + cps_volatility

print(f"CPS Volatility Adjusted Mean: {volatility_adjusted_mean}")
print(f"CPS Volatility Adjusted Median: {volatility_adjusted_median}")

# Trend analysis (using a simple moving average as an example)
cps_sma = cps.rolling(window=30).mean()  # 30-day simple moving average
print(f"Current CPS: {cps[-1]}")

# Fetch Nonfarm Payroll (NFP) data
nfp_data = fred.get_series('PAYEMS')  # 'PAYEMS' is the series ID for Nonfarm Payroll

# 1. Historical Data Analysis
# Calculate key statistics
mean_nfp = nfp_data.mean()
median_nfp = nfp_data.median()
std_dev_nfp = nfp_data.std()
percentile_25th = nfp_data.quantile(0.25)
percentile_50th = nfp_data.quantile(0.50)  # same as median
percentile_75th = nfp_data.quantile(0.75)

print("Historical NFP Data Analysis")
print(f"Mean (Average): {mean_nfp}")
print(f"Median: {median_nfp}")
print(f"Standard Deviation: {std_dev_nfp}")
print(f"25th Percentile: {percentile_25th}")
print(f"50th Percentile: {percentile_50th}")
print(f"75th Percentile: {percentile_75th}")

# 2. Dynamic Threshold Setting
# Standard deviation bands
std_dev_band_1 = (mean_nfp - std_dev_nfp, mean_nfp + std_dev_nfp)
std_dev_band_2 = (mean_nfp - 2*std_dev_nfp, mean_nfp + 2*std_dev_nfp)
std_dev_band_3 = (mean_nfp - 3*std_dev_nfp, mean_nfp + 3*std_dev_nfp)

print(f"NFP +-1 Standard Deviation: {std_dev_band_1}")
print(f"NFP +-2 Standard Deviations: {std_dev_band_2}")
print(f"NFP +-3 Standard Deviations: {std_dev_band_3}")

# Percentile-based thresholds
top_10th_percentile = nfp_data.quantile(0.90)
bottom_10th_percentile = nfp_data.quantile(0.10)

print(f"Top 10th Percentile: {top_10th_percentile}")
print(f"Bottom 10th Percentile: {bottom_10th_percentile}")

# 3. Nuanced Interpretation - Trend Analysis
# Calculate moving averages for trend analysis
nfp_moving_average_3m = nfp_data.rolling(window=3).mean()  # 3-month moving average
nfp_moving_average_6m = nfp_data.rolling(window=6).mean()  # 6-month moving average

# Print latest NFP data point for current insight
latest_nfp = nfp_data.iloc[-1]
print(f"Latest NFP Data: {latest_nfp}")
print("Recent NFP 3-Month Moving Average:", nfp_moving_average_3m.iloc[-1])
print("Recent NFP 6-Month Moving Average:", nfp_moving_average_6m.iloc[-1])

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

# Print statements for valuable insights
print(f"Latest Initial Jobless Claims: {latest_value}")
print(f"IJC 52-Week Moving Average: {latest_ma}")
print(f"IJC 52-Week Standard Deviation: {latest_std}")
print(f"IJC Week-over-Week Change: {latest_change}")
print(f"IJC 25th Percentile: {percentiles[0.25]}")
print(f"IJC 50th Percentile (Median): {percentiles[0.5]}")
print(f"IJC 75th Percentile: {percentiles[0.75]}")

# Classify the latest value based on percentile ranges
if latest_value <= percentiles[0.25]:
    print("Current level of Initial Jobless Claims is low relative to historical norms.")
elif latest_value <= percentiles[0.5]:
    print("Current level of Initial Jobless Claims is moderate relative to historical norms.")
elif latest_value <= percentiles[0.75]:
    print("Current level of Initial Jobless Claims is high relative to historical norms.")
else:
    print("Current level of Initial Jobless Claims is extremely high relative to historical norms.")

# Assess significant week-over-week change
significant_change_threshold = 10  # Placeholder threshold, adjust based on analysis
if abs(latest_change) >= significant_change_threshold:
    print("Significant week-over-week change detected in Initial Jobless Claims.")
    
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

# Print statements for valuable insights
print(f'Mean (Average) Oil Price: {mean_price}')
print(f'Oil Price Standard Deviation: {std_deviation}')
print('Oil Price Percentiles (10th, 50th, 90th):')
print(percentiles)

current_price = oil_prices[-1]

# Determine if the price is rising or falling based on moving averages
trend = "rising" if moving_average_24[-1] > moving_average_99[-1] else "falling"

# Determine the position relative to the volatility bands
position = ""
if current_price > upper_band[-1]:
    position = "above the upper volatility band, indicating potential overvaluation or high volatility."
elif current_price < lower_band[-1]:
    position = "below the lower volatility band, indicating potential undervaluation or low volatility."
else:
    position = "within the volatility bands, suggesting normal market conditions."

# Print the insights
print(f'Current Oil Price: ${current_price:.2f}')
print(f'The oil price trend is currently {trend}.')
print(f'The current price is {position}')


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

# Print statements for valuable insights
print(f"Mean CSHPI: {mean_cshpi}")
print(f"Standard Deviation CSHPI: {std_dev_cshpi}")
print(f"Percentiles: {percentiles}")
print(f"Current YoY Growth: {current_growth}%")
print(f"Mean YoY Growth: {mean_growth}%")
print(f"Standard Deviation of YoY Growth: {std_dev_growth}%")
print(f"Latest CSHPI Value's Relative Position: {relative_position}")
# Note: The current CSHPI value is the latest value in the dataset
print(f"Current CSHPI Value: {latest_value}")



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

# Tail Risk Assessment
# Assuming tail risk assessment through extreme value analysis, which would require a specific methodology or library

print("REIT Performance 5-Year Simple Moving Average (latest):", simple_moving_average_5y.dropna().iloc[-1])
print("REIT Performance 5-Year Exponential Moving Average (latest):", exponential_moving_average_5y.dropna().iloc[-1])
print("REIT Performance 10-Year Simple Moving Average (latest):", simple_moving_average_10y.dropna().iloc[-1])
print("REIT Performance 10-Year Exponential Moving Average (latest):", exponential_moving_average_10y.dropna().iloc[-1])
print("REIT Performance Top 10th Percentile Return:", top_10th_percentile)
print("REIT Performance Bottom 10th Percentile Return:", bottom_10th_percentile)
print("REIT Performance Standard Deviation of REIT Returns:", standard_deviation)
print("REIT Performance Latest Inflation Rate (aligned):", inflation_data_aligned.dropna().iloc[-1])
print("Rolling Standard Deviation (latest):", rolling_std.dropna().iloc[-1])
print("REIT Performance Z-Scores (latest):", z_scores.dropna().iloc[-1])

# Calculate skewness and kurtosis for REIT data
reit_skewness = skew(reit_data)
reit_kurtosis = kurtosis(reit_data)

# Print statements
print("Skewness of REIT Returns:", reit_skewness)
print("Kurtosis of REIT Returns:", reit_kurtosis)

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

# Print Statements
print("DXY Short-term and Long-term Moving Averages:")
print(f"DXY 50-day MA: {dxy_df['50_day_MA'].iloc[-1]}, 200-day MA: {dxy_df['200_day_MA'].iloc[-1]}")

print("\nDXY Historical Percentile Rank (Latest Value):")
print(f"DXY Percentile Rank: {dxy_df['Percentile_Rank'].iloc[-1]}")

print("\nDXY Volatility Assessment (Latest Annual Volatility):")
print(f"DXY Annual Volatility: {dxy_df['Annual_Volatility'].iloc[-1]}")

print("\nDXY Momentum Indicators (Latest Values):")
print(f"DXY 1-Month Momentum: {dxy_df['Momentum_1M'].iloc[-1]}")
print(f"DXY 3-Month Momentum: {dxy_df['Momentum_3M'].iloc[-1]}")
print(f"DXY 6-Month Momentum: {dxy_df['Momentum_6M'].iloc[-1]}")

print("\nDXY Adaptive Thresholds:")
print(f"DXY Dynamic Bullish Threshold: {bullish_threshold}")
print(f"DXY Dynamic Bearish Threshold: {bearish_threshold}")