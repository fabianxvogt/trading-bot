
import numpy as np
import pandas as pd


class strategy:
    def __init__(self) -> None:
        self.current_pos = 0
    def signal_condition(self, price_data: pd.DataFrame) -> np.ndarray:
        pass
    def set_indicators(self) -> {}:
        pass
    def generate_signals(self, df):
        df = self.set_indicators(df)

        df['open_long'] = df.apply(lambda row: self.open_long_condition(row, df.shift(1), df), axis=1)
        df['position'] = self.signal_condition(df)
        df['signal'] = df['position'].diff()
        
        return df
    
    def run_backtest(self, df):
        df = self.generate_signals(df)
        # Calculate profit for each trade
        df['buy_price'] = df['close'][df['signal'] > 0]
        df['sell_price'] = df['close'][df['signal'] < 0]

        df['buy_price'] = df['buy_price'].fillna(method='ffill')
        df['sell_price'] = df['sell_price'].fillna(method='ffill')

        df['trade_profit'] = np.where(
            (df['signal'] != 0) & (abs(df['position'] - df['signal']) == 1), 
            df['sell_price'] - df['buy_price'], 0
        )
        return df
    
    def open_long_condition(self, current_price, position, price_data: pd.DataFrame) -> bool:
        pass
    def close_long_condition(self, current_price, entry_price, position, price_data: pd.DataFrame) -> bool:
        pass
    def open_short_condition(self, current_price, position, price_data: pd.DataFrame) -> bool:
        pass
    def close_short_condition(self, current_price, entry_price, position, price_data: pd.DataFrame) -> bool:
        pass



class trader:
    def __init__(self) -> None:
        self.entry_price = 0
        self.position = 0

    def backtest(self, strategy: strategy, price_data: pd.DataFrame):
        price_data['open_long'] = price_data.apply(self.open_long_condition) 
        price_data['open_long'] = price_data.apply(self.open_long_condition) 
        price_data['open_long'] = price_data.apply(self.open_long_condition) 
        price_data['open_long'] = price_data.apply(self.open_long_condition) 
        pass

    def open_long_condition(self, current_price, position, price_data: pd.DataFrame) -> bool:

        pass
    def close_long_condition(self, current_price, entry_price, position, price_data: pd.DataFrame) -> bool:
        pass
    def open_short_condition(self, current_price, position, price_data: pd.DataFrame) -> bool:
        pass
    def close_short_condition(self, current_price, entry_price, position, price_data: pd.DataFrame) -> bool:
        pass