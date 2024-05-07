class ta:
    def SMA(klines, window):
        return klines['close'].rolling(window=window, min_periods=1, center=False).mean()

    def RSI(klines, window=14):
        daily_returns = klines['close'].diff()

        gain = daily_returns.where(daily_returns > 0, 0)
        loss = -daily_returns.where(daily_returns < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi