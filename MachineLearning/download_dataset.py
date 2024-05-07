from gym_trading_env.downloader import download
from datetime import datetime, timedelta
from pathlib import Path
timeframe = '1m'
symbols = [
    'BTC/USDT',
    'ETH/USDT',
    'ADA/USDT',
    'SOL/USDT',
    'XRP/USDT',
    'AVAX/USDT',
    'DOGE/USDT',
    'LINK/USDT',
    'DOT/USDT',
    'MATIC/USDT',
]
for i in range(40, 200):
    delta = i*10
    length = 10
    end_date = datetime.now() - timedelta(days=delta)
    start_date = end_date - timedelta(days=length)
    path = f"datasets/{i}/"
    Path(path).mkdir(parents=True, exist_ok=True)
    download(
        exchange_names = ["binance", "bitfinex2", "huobi"],
        symbols= symbols,
        timeframe= timeframe,
        dir = path,
        since= start_date,
        until = end_date,
        
    )