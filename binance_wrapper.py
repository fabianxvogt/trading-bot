import pandas as pd
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

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
        # Calculate daily returns
        df['returns'] = df['close'].pct_change()
        return df