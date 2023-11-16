import pandas as pd

def add_weekday_features(df: pd.DataFrame, dtcol=None) -> pd.DataFrame:
    """Add day of week dummies.

    If dtcol is not provided then it will be assumed
    that the index is a datetime index.

    df: pd.DataFrame
        Dataframe with datetime.
    dtcol: str
        The datetime column
    """
    if dtcol is None:
        df["weekday"] = df.index.day_name()
    else:
        df["weekday"] = df[dtcol].dt.day_name()

    df = pd.get_dummies(df, columns=["weekday"])

    return df
