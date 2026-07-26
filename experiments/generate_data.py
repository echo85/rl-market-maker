import sys
import os
import time
import datetime
import yfinance as yf

# Ensure we can import from utils if run from the experiments folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.charts import MarketCharts
except ImportError:
    pass

tickers = ['BTC-EUR', 'C3M.MI', 'VWCE.DE']
BENCHMARK_ASSET = 'VWCE.DE'

print(f'Downloading historical data for {tickers}...')
df_prices = yf.download(tickers, start='2015-01-01', end='2026-07-01')['Close']
df_prices = df_prices.dropna()
print('Data loaded successfully! Shape:', df_prices.shape)

df_normalized = (df_prices / df_prices.iloc[0]) * 100

os.makedirs("data", exist_ok=True)
date_suffix = datetime.datetime.now().strftime("%Y%m%d")
output_file = f"data/df_normalized_{date_suffix}.csv"
df_normalized.to_csv(output_file)
print(f"Successfully saved to {output_file}")
