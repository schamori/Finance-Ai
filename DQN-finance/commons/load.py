import pandas as pd


def load_chart(name: str):
    df: pd.DataFrame = pd.read_csv(f"Data/${name}.csv", sep=",")
    df = df.dropna().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])

    return df
