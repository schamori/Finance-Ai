"""
Usage:
import pandas as pd

# Define the strategy function
def compute_strategy(df, parameters):
    df['firstSMA'] = df['Close'].rolling(window=parameters[0]).mean()
    df['secondSMA'] = df['Close'].rolling(window=parameters[1]).mean()
    df["buying"] = df["firstSMA"] < df['secondSMA']
    return df

# Load data
df = pd.read_csv('Data/MSCI GLOBAL.csv', sep=',')

# Define parameter combinations
# Parameters can have custom sizes, but all tuples need to have the same size.
averages = [(10, 20), (20,30)]


# Compute best parameters
df, best_parameters, best_course = compute_best_parameters(df, averages, compute_strategy)
"""

from tqdm import tqdm
import numpy as np


def compute_best_parameters(df, parameters_range, compute_strategy):
    """
    df (pandas DataFrame): The historical stock data with at least a 'Close' column.
    parameters_range (iterable): An iterable containing parameter combinations to test.
    compute_strategy (function): A function that computes the strategy for a given set of parameters and returns the modified dataframe.
    Returns:
    A tuple containing:

    The dataframe with the strategy computed for the best parameters.
    The best parameters.
    The best course (highest returns).
    """
    best_parameters = None
    best_course = -float("inf")

    for parameters in tqdm(parameters_range):
        compute_strategy(df, parameters)
        buying = df["buying"].values
        close = df["Close"].values
        # Using numpy for faster computation
        diff = np.where(
            (buying[:-1] & buying[1:]) | (buying[:-1] & ~buying[1:]),
            close[1:] - close[:-1],
            0,
        )
        # The last entry is not considerd above
        last_diff = close[-1] - close[-2] if buying[-1] else 0

        algorithm_course = close[0] + diff.sum() + last_diff

        if algorithm_course > best_course:
            best_course = algorithm_course
            best_parameters = parameters

    return best_parameters
