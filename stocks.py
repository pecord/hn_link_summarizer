import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

# Set dark mode style
plt.style.use('dark_background')

# Directory to save output files
OUTPUT_DIR = 'output'

# Duration for recent data - e.g., '1y' for one year, '6mo' for six months
RECENT_PERIOD = '1y'

# List of stock symbols
STOCK_SYMBOLS = ['AAPL', 'AMZN', 'NFLX']

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
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators (SMA, EMA, RSI) to the DataFrame.

    Parameters:
        df (pd.DataFrame): Historical stock data

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    df['SMA'] = calculate_sma(df['Close'], 30)
    df['EMA'] = calculate_ema(df['Close'], 30)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    return df

def plot_and_save_stock_data(stock_data: Dict[str, pd.DataFrame], output_dir: str):
    """
    Plot historical stock data and RSI on the same graph and save as JPEG.

    Parameters:
        stock_data (dict): Dictionary containing the symbol and its historical data
        output_dir (str): Directory to save the output files
    """
    
    symbol = stock_data['symbol']
    df = stock_data['historical_data']
    df = add_technical_indicators(df)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Close Price with SMA and EMA on primary y-axis
    ax1.plot(df.index, df['Close'], label=f"{symbol} Close Price", color='#1f77b4')  # blue
    ax1.plot(df.index, df['SMA'], label='30-Day SMA', color='#ff7f0e')  # orange
    ax1.plot(df.index, df['EMA'], label='30-Day EMA', color='#2ca02c')  # green

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price (USD)', color='white')
    ax1.tick_params(axis='y', labelcolor='white')
    ax1.legend(loc='upper left')
    
    # Create secondary y-axis for RSI
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['RSI'], label='RSI', color='#d62728', alpha=0.75)  # red with transparency
    ax2.set_ylabel('RSI', color='white')
    ax2.tick_params(axis='y', labelcolor='white')
    ax2.legend(loc='upper right')

        # RSI upper bound line
    ax2.axhline(80, color='#d62728', linestyle='--', linewidth=1)
    ax2.text(df.index[-1], 70, 'Overbought', color='#d62728', verticalalignment='bottom')
    ax2.text(df.index[-1], 30, 'Oversold', color='#d62728', verticalalignment='bottom')


    # Set title and grid
    plt.title(f"Recent Close Price and Technical Indicators for {symbol}", color='white')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Set date formatting for x-axis
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    for label in ax1.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right', color='white')

    plt.savefig(os.path.join(output_dir, f"{symbol}_recent_combined_plot.jpg"), format='jpeg')
    plt.close()

def save_data_to_csv(df: pd.DataFrame, symbol: str, output_dir: str):
    """
    Save the historical data with technical indicators to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame with historical stock data and technical indicators
        symbol (str): Stock symbol
        output_dir (str): Directory to save the CSV file
    """
    file_path = os.path.join(output_dir, f"{symbol}_recent_historical_data.csv")
    df.to_csv(file_path)

def main():
    """Main function to execute the entire workflow."""
    ensure_output_directory(OUTPUT_DIR)
    
    for symbol in STOCK_SYMBOLS:
        try:
            stock_data = get_recent_historical_data(symbol, RECENT_PERIOD)
            plot_and_save_stock_data(stock_data, OUTPUT_DIR)
            save_data_to_csv(stock_data['historical_data'], symbol, OUTPUT_DIR)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    main()