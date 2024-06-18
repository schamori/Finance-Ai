"""
Usage:

df.readinStock

df["buying"] = df["firstSMA"] < df['secondSMA']

def function_to_plot():
    plt.plot(df['firstSMA'][higher_averages[0]:].reset_index(drop=True), label=f'{lower_average}-day SMA', color='red')
    plt.plot(df['secondSMA'][higher_averages[0]:].reset_index(drop=True), label=f'{higher_average}-day SMA', color='black')


plot_stock_strategy(df[higher_average:].reset_index(drop=True), function_to_plot)
"""

import math
from matplotlib import figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


def plot_profit_fields(ax: Axes, signals: list[bool], chart: list[float], dates, algo_course=True):
    max_y = max(chart)
    last_action_entry = (0, chart[0])
    for i, signal in enumerate(signals[1:], start=1):
        if signal != signals[i - 1] or i == len(signals) - 1:
            last_index, last_entry = last_action_entry
            if signals[i - 1]:
                _plot_entry(ax,  dates, signals[i - 1], last_index, i, last_entry, chart[i], max_y, algo_course)
            last_action_entry = (i, chart[i])



def _plot_entry(
    ax: Axes,
    dates,
    signal,
    first_index,
    second_index,
    first_entry,
    second_entry,
    max_y,
    algo_course
):
    """
    index (int): The index of the stock entry.
    stock_entry (dict): A dictionary containing stock data for a particular day. It should have keys "Close" (closing price) and "buying" (a boolean indicating whether a stock is being bought).
    total_screen_height (float): The difference between the first and the last closing prices in the dataset.
    total_screen_width (float): The total width of the screen/plot.
    """
    if algo_course:
        color = "red" if (second_entry - first_entry) < 0 else "green"
    else:
        color = "red" if (second_entry - first_entry) * signal < 0 else "green"

    borders = [0, max(first_entry, second_entry)] if signal > 0 else [min(first_entry, second_entry), max_y]

    # Add borders to the fill
    border_color = 'black'  # Define border color
    border_width = 1.5  # Define border width

    ax.fill_betweenx(
        borders,
        dates[first_index],
        dates[second_index],
        facecolor=color,
        alpha=0.3,
        edgecolor=border_color,  # Add border color
        linewidth=border_width  # Add border width
    )
