import requests
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Your Alpha Vantage API key
API_KEY = '0Z0AB7PPHCB9X8W1'

# List of stock symbols
stock_symbols = ['AAPL', 'AMZN', 'NFLX']

# Base URL for Alpha Vantage
BASE_URL = 'https://www.alphavantage.co/query?'

# Function to get historical stock data
def get_historical_data(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': 'compact'  # 'compact' for last 100 days, 'full' for full-length history
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if 'Time Series (Daily)' in data:
        historical_data = data['Time Series (Daily)']
        return {
            'symbol': symbol,
            'historical_data': historical_data
        }
    else:
        print(f"Error fetching data for {symbol}")
        return None

# Fetch historical data for each company in the list
stock_data_list = []
for symbol in stock_symbols:
    stock_data = get_historical_data(symbol)
    if stock_data:
        stock_data_list.append(stock_data)
    # Respect the API call rate limit
    time.sleep(12)

# Function to plot historical stock data and save as JPEG
def plot_and_save_stock_data(stock_data):
    df = pd.DataFrame.from_dict(stock_data['historical_data'], orient='index')
    df.index = pd.to_datetime(df.index) 
    df = df.sort_index()
    df = df.astype(float)  # Convert data types to float

    # Calculate Moving Averages
    df['20_MA'] = df['4. close'].rolling(window=20).mean()
    df['50_MA'] = df['4. close'].rolling(window=50).mean()

    plt.figure(figsize=(14, 7))

    plt.plot(df['4. close'], label=f"{stock_data['symbol']} Close Price", color='blue', linestyle='-')
    if not df['20_MA'].isna().all():
        plt.plot(df['20_MA'], label='20-Day Moving Average', color='orange', linestyle='--')
    if not df['50_MA'].isna().all():
        plt.plot(df['50_MA'], label='50-Day Moving Average', color='green', linestyle='--')

    plt.title(f"Historical Close Price with Moving Averages for {stock_data['symbol']}", fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price (USD)', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Date formatting
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Rotate date labels
    plt.gcf().autofmt_xdate()

    # Save the plot as a JPEG file
    plt.savefig(f"{stock_data['symbol']}_historical_plot.jpg", format='jpeg')
    plt.close()

# Plot and save historical data for each company
for stock_data in stock_data_list:
    plot_and_save_stock_data(stock_data)
    
    # Save data to file for future analysis
    with open(f"{stock_data['symbol']}_historical_data.json", 'w') as file:
        json.dump(stock_data, file, indent=4)