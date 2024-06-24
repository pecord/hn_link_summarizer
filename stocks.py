import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from typing import Dict
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

# Set dark mode style
plt.style.use('dark_background')

# Load configuration from the JSON file
def load_config(config_file: str) -> Dict:
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def ensure_output_directory(directory: str):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_recent_historical_data(symbol: str, period: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch recent historical stock data for the given symbol and period.

    Parameters:
        symbol (str): Stock symbol (e.g., 'AAPL')
        period (str): Time period for the data (e.g., '1y')

    Returns:
        dict: Dictionary containing the symbol and its historical data.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    hist = hist.round(2)  # Round data to 2 decimal places

    # Ensure the date index is in datetime format
    hist.index = pd.to_datetime(hist.index)

    # Convert volume to a more readable format
    hist['Volume'] = (hist['Volume'] / 1e6).round(2)  # Converts volume to millions

    return {'symbol': symbol, 'historical_data': hist}

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA)."""
    return data.rolling(window=window).mean().round(2)

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    return data.ewm(span=window, adjust=False).mean().round(2)

def calculate_rsi(data: pd.Series, window: int) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)

def plot_and_save_stock_data(stock_data: Dict[str, pd.DataFrame], output_dir: str):
    """Plot and save recent stock data with technical indicators."""
    symbol = stock_data['symbol']
    df = stock_data['historical_data']

    df['SMA_30'] = calculate_sma(df['Close'], 30)
    df['EMA_30'] = calculate_ema(df['Close'], 30)
    df['RSI'] = calculate_rsi(df['Close'], 14)

    fig, ax1 = plt.subplots(figsize=(14, 8))

    ax1.plot(df.index, df['Close'], label=f'{symbol} Close Price', color='skyblue')
    ax1.plot(df.index, df['SMA_30'], label='30-Day SMA', color='orange', linestyle='--')
    ax1.plot(df.index, df['EMA_30'], label='30-Day EMA', color='limegreen', linestyle='--')

    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Close Price (USD)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df['RSI'], label='RSI', color='red', alpha=0.55)
    ax2.set_ylabel('RSI', fontsize=14)
    ax2.axhline(70, color='grey', linestyle='--', linewidth=1)
    ax2.axhline(30, color='grey', linestyle='--', linewidth=1)
    ax2.text(df.index[-1], 70, 'Overbought', color='grey', verticalalignment='bottom', fontsize=10)
    ax2.text(df.index[-1], 30, 'Oversold', color='grey', verticalalignment='bottom', fontsize=10)
    ax2.legend(loc='upper right', fontsize=12)

    plt.title(f"Recent Close Price and Technical Indicators for {symbol}", color='white', fontsize=16)

    # Set date formatting for x-axis
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    for label in ax1.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right', fontsize=12, color='white')

    plt.savefig(os.path.join(output_dir, f"{symbol}_recent_combined_plot.png"), format='png', dpi=300)
    plt.close()

def save_data_to_csv(df: pd.DataFrame, symbol: str, output_dir: str):
    """
    Save the historical data with technical indicators to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame with historical stock data and technical indicators
        symbol (str): Stock symbol
        output_dir (str): Directory to save the CSV file
    """
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')  # Format the date to remove the time component
    file_path = os.path.join(output_dir, f"{symbol}_recent_historical_data.csv")
    df.to_csv(file_path)

def save_data_to_markdown(df: pd.DataFrame, symbol: str, output_dir: str):
    """
    Save the historical data with technical indicators to a Markdown file.

    Parameters:
        df (pd.DataFrame): DataFrame with historical stock data and technical indicators
        symbol (str): Stock symbol
        output_dir (str): Directory to save the Markdown file
    """
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')  # Format the date to remove the time component
    df = df.fillna('N/A')  # Fill NaN values with 'N/A'
    file_path = os.path.join(output_dir, f"{symbol}_recent_historical_data.md")
    with open(file_path, 'w') as file:
        file.write(f"# Historical Data for {symbol}\n")
        file.write(df.to_markdown())

def main():
    """Main function to execute the entire workflow."""
    config = load_config('config.json')
    
    output_directory = config['output_directory']
    ensure_output_directory(output_directory)

    for symbol in config['stock_symbols']:
        try:
            stock_data = get_recent_historical_data(symbol, config['recent_period'])
            plot_and_save_stock_data(stock_data, output_directory)
            save_data_to_csv(stock_data['historical_data'], symbol, output_directory)
            save_data_to_markdown(stock_data['historical_data'], symbol, output_directory)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    main()
