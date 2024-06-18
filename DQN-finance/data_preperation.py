import pandas as pd
import random
from PIL import Image, ImageDraw
import requests
import os
import yfinance as yf
from tqdm import tqdm
# API has a limit just for testing
def generate_random_candlestick_data(start_date, num_days):
    dates = pd.date_range(start=start_date, periods=num_days)
    data = {'Date': dates, 'Open': [], 'High': [], 'Low': [], 'Close': []}
    open_price = 100  # Initial price
    for _ in range(num_days):
        daily_open = open_price
        daily_high = daily_open + random.uniform(1, 10)
        daily_low = daily_open - random.uniform(1, 10)
        daily_close = random.uniform(daily_low, daily_high)

        data['Open'].append(daily_open)
        data['High'].append(daily_high)
        data['Low'].append(daily_low)
        data['Close'].append(daily_close)

        open_price = daily_close  # Use the close price of today as the open price of tomorrow

    return pd.DataFrame(data)



def create_candlestick_image(data, width=84, height=84, save_path="candlestick.png"):
    candle_width = 3
    wick_width = 1
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Determine the price range
    price_min = min(data['Low'])
    price_max = max(data['High'])
    price_range = price_max - price_min

    # Normalize price values to pixel positions
    def price_to_pixel(price):
        return height - int((price - price_min) / price_range * (height - 1))

    # Draw each candlestick
    for i, row in enumerate(data.itertuples()):
        pos = i * candle_width

        # Calculate candlestick body
        open_pixel = price_to_pixel(row.Open)
        close_pixel = price_to_pixel(row.Close)
        low_pixel = price_to_pixel(row.Low)
        high_pixel = price_to_pixel(row.High)

        color = 'gray' if row.Close >= row.Open else 'black'

        # Draw the candlestick body
        draw.rectangle([pos, min(open_pixel, close_pixel), pos + candle_width - 1, max(open_pixel, close_pixel)], fill=color)

        # Draw the high-low wicks
        draw.line([pos + candle_width // 2, low_pixel, pos + candle_width // 2, high_pixel], fill=color, width=wick_width)

    image.save(save_path)


def generate_rewards(data, days, csv_file):
    rewards = [data['Close'].iloc[i + days] - data['Close'].iloc[i + days - 1] for i in range(0, len(data) - days - 1)]



    pd.DataFrame(rewards, columns=['Reward']).to_csv(csv_file, index=False)

# df = generate_random_candlestick_data(start_date='2023-04-01', num_days=num_days)
# create_candlestick_image(df, filename='train/0.png')

def data_preparation(stock_symbols, days_per_image = 28, start_date=None, end_date=None, split = 0.71):

    for stock_symbol in tqdm(stock_symbols):
        df = yf.Ticker(stock_symbol).history(period='max')

        if (len(df) < days_per_image):
            print('No data available for {}'.format(stock_symbol))
            continue
        # Each stock_symbol gets a folder
        os.makedirs(f"train/{stock_symbol}", exist_ok=True)

        os.makedirs(f"test/{stock_symbol}", exist_ok=True)

        df.index = pd.to_datetime(df.index)

        split_index = int(len(df) * split)
        train_df = df[:split_index]
        test_df = df[split_index:]

        train_df.to_csv(f"train/{stock_symbol}/{stock_symbol}.csv")
        test_df.to_csv(f"test/{stock_symbol}/{stock_symbol}.csv")

        # Save training plots and rewards
        for i in range(len(train_df) - 1  - days_per_image):
            data_slice = train_df.iloc[i:i + days_per_image]
            create_candlestick_image(data_slice, save_path=f'train/{stock_symbol}/{i}.png')

        generate_rewards(train_df, days_per_image, f'train/{stock_symbol}/train.csv')

        # Save testing plots and rewards
        for i in range(len(test_df) - days_per_image):
            data_slice = test_df.iloc[i:i + days_per_image]
            create_candlestick_image(data_slice, save_path=f'test/{stock_symbol}/{i}.png')

        generate_rewards(test_df, days_per_image, f'test/{stock_symbol}/test.csv')


if __name__ == '__main__':
    stock_symbols = [
        "MSCI",  # MSCI Inc. (index)
        "IBM",  # International Business Machines
        "META",  # Meta Platforms Inc.
        "AAPL",  # Apple Inc.
        "AMZN",  # Amazon.com Inc.
        "GOOGL",  # Alphabet Inc.
        "MSFT",  # Microsoft Corporation
        "TSLA",  # Tesla Inc.
        "NVDA",  # NVIDIA Corporation
        "JPM",  # JPMorgan Chase & Co.
        "BAC",  # Bank of America Corporation
        "NFLX",  # Netflix Inc.
        "BRK.B",  # Berkshire Hathaway Inc.
        "V",  # Visa Inc.
        "MA",  # Mastercard Incorporated
        "UNH",  # UnitedHealth Group Incorporated
        "SPY",  # SPDR S&P 500 ETF (index)
        "DIA",  # SPDR Dow Jones Industrial Average ETF (index)
        "QQQ",  # Invesco QQQ Trust (NASDAQ index)
        "IWM",  # iShares Russell 2000 ETF (small-cap index)
        "PG",  # Procter & Gamble Company
        "HD",  # The Home Depot Inc.
        "CVX",  # Chevron Corporation
        "NKE",  # NIKE Inc.
        "PFE",  # Pfizer Inc.
    ]
    data_preparation(stock_symbols)


