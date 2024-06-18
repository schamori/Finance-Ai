from data_preperation import data_preparation

from train import train_dqn
from evaluation import evaluate
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

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
        "VTI",
        "VWO",
        "AGG",
        "VGK",
        "XLF",
        "XLE",
        "VEA",
        "EEM"
]


data_preparation(stock_symbols)

performance = train_dqn(num_episodes=2000)

# Plotting the performance
plt.figure(figsize=(10, 6))
plt.plot( performance, label='Training Performance')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Training Performance of DQN')
plt.legend()
plt.grid(True)
plt.show()

q_network_path = "weights/weights_best.pth"

# Stock to evalate and visualize
stock_to_evaluate = "MSCI"
total_capital = 4000

evaluate(q_network_path, stock_to_evaluate, total_capital)