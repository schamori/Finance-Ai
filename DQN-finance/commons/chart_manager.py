from typing import Sequence
import numpy as np
import pandas as pd
import os
import numpy.typing as npt


class ChartManager:
    original_chart: pd.DataFrame = None
    chart: pd.DataFrame = None

    def __init__(self):
        pass

    def load(self, name: str):
        df: pd.DataFrame = pd.read_csv(name, sep=",")
        df = df.dropna().reset_index(drop=True)

        self.chart = df
        self.original_chart = df.copy()

    def calculate_return(self, signals: Sequence[int]) -> Sequence[float]:
        if self.chart is None:
            raise Exception("Must first call 'load' before 'calculate_investment'!")

        df_iter = self.chart.iterrows()
        _, stock_entry = next(df_iter)

        algorithm_course = [stock_entry["Close"]]

        reward = list()
        for i, stock_entry in df_iter:
            # Sell when there is a change. There is one day where you still buy while buying is false
            # this is because you could not have known that the change will come tomorrow so you have to hold it until
            # a change comes.
            # Because we are adding the last day we are always one day behind
            reward.append((stock_entry["Close"] - algorithm_course[-1]) * signals[i])
            if (
                signals[i]
                and signals[i - 1]
                or not signals[i]
                and signals[i - 1]
            ):
                algorithm_course.append(
                    algorithm_course[-1]
                    + (stock_entry["Close"] - self.chart["Close"][i - 1]) * signals[i- 1]
                )
            else:
                algorithm_course.append(algorithm_course[-1])

        print(np.array(reward).mean())
        return np.array(algorithm_course) / algorithm_course[0]

    def invest_once(self, chart: Sequence[float], amount: int) -> Sequence[float]:
        return list((chart / chart[0]) * amount)

    def invest_rolling(
        self, chart: Sequence[float], start_amount: int, interval_amount, interval_days: int = 30
    ) -> Sequence[float]:
        """
        monthly_capital = (
        start_capital / len(df) * interval_days
        )
        :param chart:
        :param start_amount:
        :param interval_amount:
        :param interval_days:
        :return:
        """
        total_days = len(chart)

        bought_stocks = start_amount / chart[0]
        rolling_value = []
        for i in range(0, total_days):
            if i % interval_days == 0:
                bought_stocks += interval_amount / chart[i]
                
            rolling_value.append(bought_stocks * chart[i])

        return list(rolling_value)
