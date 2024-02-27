import yfinance as yf

# Step 1: Define Sector(s) of Interest
sectors_of_interest = ['Technology', 'Healthcare', 'Consumer Discretionary']

# Step 2: Set Financial Metrics Filters
financial_metrics_filters = {
    'profitability_ratio': 0.1,  # Minimum acceptable profitability ratio
    'liquidity_ratio': 1.5,       # Minimum acceptable liquidity ratio
    # Add more financial metrics and their benchmarks as needed
}

# Step 3: Access Financial Data Using Yahoo Finance API
def get_financial_data(sectors):
    # Initialize an empty dictionary to store financial data
    financial_data = {}
    for sector in sectors:
        # Use Yahoo Finance API to get data for each sector
        stocks = yf.download(yf.tickers.Tickers, period='1d', group_by='ticker')
        for stock in stocks:
            financial_data[stock] = {}
            try:
                # Get financial metrics for each stock
                financial_data[stock]['profitability_ratio'] = stocks[stock]['Net Income'] / stocks[stock]['Revenue']
                financial_data[stock]['liquidity_ratio'] = stocks[stock]['Current Assets'] / stocks[stock]['Current Liabilities']
                # Add more financial metrics as needed
            except Exception as e:
                print(f"Error retrieving data for {stock}: {e}")
    return financial_data

financial_data = get_financial_data(sectors_of_interest)

# Step 4: Filter Stocks Based on Financial Metrics
filtered_stocks = []
for stock, metrics in financial_data.items():
    if all(metrics[metric] >= threshold for metric, threshold in financial_metrics_filters.items()):
        filtered_stocks.append(stock)

# Step 5: Output the Filtered List of Stocks
print("Filtered List of Stocks:")
for stock in filtered_stocks:
    print(stock)
