from datetime import datetime, timedelta
import os
from binance.client import Client
from numpy import nan
import pandas as pd

from binance_wrapper import binance

# Replace 'your_api_key' and 'your_api_secret' with your actual Binance API key and secret
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

bin = binance(api_key, api_secret)


# Define trading parameters
symbol = 'BTCUSDT'
interval = '1w'
ma_period = 5
end_date = datetime.now()
start_date = end_date - timedelta(days=1000)
klines = bin.get_klines(symbol, interval, start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S"))

df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Calculate moving average
df['MA'] = df['close'].rolling(window=ma_period).mean()

# Set parameters for entry and exit conditions
entry_threshold = 0.0001  # 10% below/above entry price for closing short/long
exit_threshold = 0.0001   # 10% below/above entry price for closing long/short

def clear_signal(df):
    df = df.diff()
    df[df == -1] = 0
    return pd.to_numeric(df).astype('Int64')

# Generate signals for opening and closing long/short positions
df['open_long_signal'] = 0
df['open_long_signal'][df['close'] > df['MA']*1.01] = 1  # Buy (open long)
df['open_long_signal'] = clear_signal(df['open_long_signal'])
df['open_short_signal'] = 0
df['open_short_signal'][df['close'] < df['MA']/1.01] = 1  # Sell (open short)
df['open_short_signal'] = clear_signal(df['open_short_signal'])

df['last_open_position'] = nan
df['last_open_position'][df['open_long_signal'] == 1] = 1
df['last_open_position'][df['open_short_signal'] == 1] = -1
df['last_open_position'] = df['last_open_position'].fillna(method='ffill')

# Detect position changes
#df['signal'] = df['position'].diff()

# Track entry price for each position
df['entry_price'] = nan
df['entry_price'][df['open_long_signal'] == 1] = df['close'][df['open_long_signal'] == 1]  # Set entry price for long position
df['entry_price'][df['open_short_signal'] == 1] = df['close'][df['open_short_signal'] == 1]  # Set entry price for short position
df['entry_price'] = df['entry_price'].fillna(method='ffill')

# Generate signals for closing long/short positions based on entry price and criteria
df['close_long_signal'] = 0
df['close_long_signal'][(df['open_long_signal'] == 1) & (df['close'] < df['entry_price'] * (1 - entry_threshold))] = 1  # Close long position
df['close_long_signal'][(df['open_long_signal'] == 1) & (df['close'] < df['entry_price'])] = 1  # Close long position

df['close_long_signal'] = clear_signal(df['close_long_signal'])

df['close_short_signal'] = 0
df['close_short_signal'][(df['open_short_signal'] == 1) & (df['close'] > df['entry_price'] * (1 + entry_threshold))] = 1  # Close short position

df['close_short_signal'] = clear_signal(df['open_short_signal'])


df['trade_profit'] = 0
df['trade_profit'][df['close_long_signal'] == 1]

# Print signals, position changes, and close signals
print("Signals, Position Changes, Entry Prices, and Close Signals:")
print(df.tail(50))
