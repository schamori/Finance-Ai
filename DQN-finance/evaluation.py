import dqn.dqn as dqn
from models.StockEnv import StockEnv
import pandas as pd
import torch
from commons import *
from matplotlib.axes import Axes
import numpy as np

import matplotlib.pyplot as plt
def evaluate_dqn(weights_dir, stock=None):
    env = StockEnv(test=True, stock=stock)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = dqn.QNetworkCNN(action_dim=env.action_space.n).to(device)
    q_network.load_state_dict(torch.load(weights_dir))
    return dqn.evaluate(q_network, env, 1)


def evaluate(q_network_path, stock_symbol, total_capital=4000, num_of_days=28):

    chart_manager = ChartManager()
    chart_manager.load(f"test/{stock_symbol}/{stock_symbol}.csv")
    chart_manager.chart = chart_manager.chart.iloc[num_of_days:].reset_index(drop=True)

    avg_reward, signals = evaluate_dqn(q_network_path, stock_symbol)


    axes: Axes  # Needed for intellisense
    fig, axes = plt.subplots(1, 1, num=1)

    axes.plot(
        chart_manager.chart["Date"],
        chart_manager.chart["Close"],
        label="Closing Prices",
        lw=0.5,
        color="blue",
    )
    algorithm_course = chart_manager.calculate_return(signals)


    steps = 30
    monthly_capital = total_capital / len(chart_manager.chart) * steps

    invest_once_result = chart_manager.invest_once(chart_manager.chart["Close"], total_capital)
    axes.plot(
        chart_manager.chart["Date"],
        invest_once_result,
        label="All-in account",
        lw=0.5,
        color="blue",
    )

    invest_once_algorithm_result = chart_manager.invest_once(algorithm_course, total_capital)
    axes.plot(
        chart_manager.chart["Date"],
        invest_once_algorithm_result,
        label="All-in account via algorithm",
        lw=0.5,
        color="purple",
    )


    invest_monthly_result = chart_manager.invest_rolling(chart_manager.chart["Close"], 0, monthly_capital, steps)
    axes.plot(
        chart_manager.chart["Date"],
        invest_monthly_result,
        label="Monthly account",
        lw=0.5,
        color="red",
    )

    invest_monthly_algorithm_result = chart_manager.invest_rolling(algorithm_course, 0, monthly_capital, steps)
    axes.plot(
        chart_manager.chart["Date"],
        invest_monthly_algorithm_result,
        label="Monthly account via algorithm",
        lw=0.5,
        color="brown",
    )

    no_invest_monthly_result = chart_manager.invest_rolling(np.repeat(1, len(chart_manager.chart)), 0, monthly_capital, steps)
    axes.plot(
        chart_manager.chart["Date"],
        no_invest_monthly_result,
        label="Saving money monthly",
        lw=0.5,
        alpha=0.5,
        color="green",
    )

    plot_profit_fields(axes, signals, invest_once_algorithm_result, chart_manager.chart["Date"])

    print(f"""
    Final amount for All-in account: {round(invest_once_result[-1], 1)} $
    Final amount for All-in account via algorithm: {round(invest_once_algorithm_result[-1], 1)} $
    Final amount for Monthly account: {round(invest_monthly_result[-1], 1)} $
    Final amount for Monthly account via algorithm: {round(invest_monthly_algorithm_result[-1], 1)} $
    Final amount for just saving: {round(no_invest_monthly_result[-1], 1)} $
    """)
    plt.legend(loc="upper left")
    plt.show()
    return invest_once_algorithm_result[-1] - total_capital 


if __name__ == '__main__':
    q_network_path = "weights/best_weights_last_epoch_720.pth"

    stock_symbols = ["NVDA"]
    # Stock to evalate and visualize

    best_stock = None
    best_performance = -float('inf')  # Initialize with a very low value
    total_capital = 4000


    for stock_symbol in stock_symbols:
        performance = evaluate(q_network_path, stock_symbol, total_capital)

        if performance > best_performance:
            best_performance = performance
            best_stock = stock_symbol

    print(f'The best stock is {best_stock} with a performance of {best_performance}')


