from datetime import datetime, timedelta
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

class binance:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)

    def get_klines(self, symbol, interval, time_from, time_until):

        historical_klines = self.client.get_historical_klines(symbol, interval, time_from, time_until)
        # Create a DataFrame from the klines data
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(historical_klines, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Convert 'close' column to numeric
        df['close'] = pd.to_numeric(df['close'])
        return df
    
def SMA(klines, window):
    return klines['close'].rolling(window=window, min_periods=1, center=False).mean()
    klines[f'MA{window}'] = klines['close'].rolling(window=window, min_periods=1, center=False).mean()
    return klines

def RSI(klines, window=14):
    daily_returns = klines['close'].diff()

    gain = daily_returns.where(daily_returns > 0, 0)
    loss = -daily_returns.where(daily_returns < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


class strategy:
    def __init__(self, indicators=None, signal_condition=None):
        self.indicators = indicators if indicators is not None else {}
        self.signal_condition = signal_condition
    def add_indicator(self, name, indicator_function):
        self.indicators[name] = indicator_function
    def set_signal_condition(self, signal_condition):
        self.signal_condition = signal_condition
    def generate_signals(self, df):
        for indicator_name, indicator_function in self.indicators.items():
            df[indicator_name] = indicator_function(df)
        if self.signal_condition is not None:
            df['signal'] = self.signal_condition(df, **self.indicators)
        else:
            df['signal'] = 0

        df['position'] = df['signal'].diff()
        return df
    def run_backtest(self, df):
        df = self.generate_signals(df)
        # Calculate daily returns
        df['returns'] = df['close'].pct_change()
        # Calculate profit for each trade
        df['buy_price'] = df['close'][df['position'] > 0]
        df['sell_price'] = df['close'][df['position'] < 0]

        # Forward fill NaN values in buy_price column
        #df['buy_price'] = df['buy_price'].fillna(method='ffill')
        #klines['sell_price'] = klines['sell_price'].fillna(method='ffill')
        df['trade_profit'] = df['sell_price'] - df['buy_price']
        return df

bin = binance(api_key, api_secret)

symbol = 'BTCUSDT'
end_date = datetime(2021, 11, 30)# datetime.now()
start_date = end_date - timedelta(days=50)
interval = Client.KLINE_INTERVAL_1DAY
klines = bin.get_klines(symbol, interval, start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S"))

indicators = {'Short_MA': lambda df: SMA(df, window=1),
              'Long_MA':  lambda df: SMA(df, window=2),
              'RSI':      lambda df: RSI(df, window=3)}

# Custom signal condition function
def custom_signal_condition(df, **kwargs):
    short_ma = kwargs['Short_MA'](df)
    long_ma = kwargs['Long_MA'](df)
    rsi = kwargs['RSI'](df)

    return np.where(short_ma > long_ma, 1, -1)
#np.where((short_ma > long_ma) & (rsi < 30), 1,
 #                   np.where((short_ma < long_ma) & (rsi > 70), -1, 0))
strat = strategy(indicators)
strat.set_signal_condition(custom_signal_condition)

df = strat.run_backtest(klines)
print(df)
