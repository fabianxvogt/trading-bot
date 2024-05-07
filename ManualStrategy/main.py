from datetime import datetime, timedelta
from typing import Any
import numpy as np
from numpy import ndarray
import pandas as pd
import os
import matplotlib.pyplot as plt
from binance_wrapper import binance

from strategy import strategy
#from ta import ta

import talib as TA


class sma_strategy(strategy):
    def __init__(self):
        super().__init__()
    def signal_condition(self, price_data: pd.DataFrame) -> ndarray:
        neutral_threshold = 1.000
        return np.where(
            price_data['position'].shift(1) == 1, 1, np.where(
                price_data['position'].shift(1) == -1

        ))
        return np.where(price_data['UPPER_BAND'] < price_data['close'], -1, price_data['LOWER_BAND'] > price_data['close'], 1, 0)
        return np.where(price_data['RSI'] < 30, 1, np.where(price_data['RSI'] > 70, -1, 0))
        #return np.where(price_data['Short_MA'] > price_data['Long_MA'] * neutral_threshold, 1, np.where(price_data['Short_MA'] < price_data['Long_MA'] / neutral_threshold, -1, 0))
    
    def set_indicators(self, price_data) -> {}:
        #price_data['Short_MA'] = TA.SMA(price_data['close'], timeperiod=50)
        #price_data['Long_MA'] = TA.SMA(price_data['close'], timeperiod=200)
        #price_data['RSI'] = TA.RSI(price_data['close'], timeperiod=14)
        price_data['UPPER_BAND'], price_data['MIDDLE_BAND'], price_data['LOWER_BAND'] = TA.BBANDS(price_data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        return price_data

    def open_long_condition(self, current_price, position, price_data: pd.DataFrame) -> bool:
        pass
    def close_long_condition(self, current_price, position, price_data: pd.DataFrame) -> bool:
        pass
    def open_short_condition(self, current_price, position, price_data: pd.DataFrame) -> bool:
        pass
    def close_short_condition(self, current_price, position, price_data: pd.DataFrame) -> bool:
        pass
    
def run_backtest(symbol, start_date, end_date, interval, strategy, plot_results):
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    bin = binance(api_key, api_secret)
    klines = bin.get_klines(symbol, interval, start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S"))

    df = strategy().run_backtest(klines)

    profiting_trade_count = (df['trade_profit'] > 0).sum()
    loosing_trade_count = (df['trade_profit'] < 0).sum()
    trade_count = profiting_trade_count + loosing_trade_count
    total_profit = df['trade_profit'].sum()

    print(df.tail(50))
    print(f"Number of trades: {trade_count}")
    print(f"Profiting trades: {profiting_trade_count} ({round(profiting_trade_count/trade_count*100, 2)}%)")
    print(f"Loosing trades: {loosing_trade_count} ({round(loosing_trade_count/trade_count*100, 2)}%)")
    print(f"Total Profit: {df['trade_profit'].sum()}")


    if(plot_results):
        # Plot historical prices and strategy signals
        plt.figure(figsize=(12, 6))
        plt.plot(df['close'], label='Close Price')
        plt.scatter(df.index[df['signal'] == 1], df['close'][df['signal'] == 1], marker='^', color='g', label='Open Long')
        plt.scatter(df.index[df['signal'] == -1], df['close'][df['signal'] == -1], marker='v', color='r', label='Close Long')
        plt.scatter(df.index[df['signal'] == 2], df['close'][df['signal'] == 2], marker='^', color='r', label='Open Short')
        plt.scatter(df.index[df['signal'] == -2], df['close'][df['signal'] == -2], marker='v', color='g', label='Close Short')
        plt.legend()
        plt.show()


symbol = 'BTCUSDT'
end_date = datetime.now()
start_date = end_date - timedelta(days=100)
interval = "1h"
run_backtest(symbol, start_date, end_date, interval, sma_strategy, False)