import pandas as pd

def add_weekday_features(df: pd.DataFrame, dtcol=None) -> pd.DataFrame:
    """Add day of week dummies.

    If dtcol is not provided then it will be assumed
    that the index is a datetime index.

    PARAMETERS
    ----------
    df: pd.DataFrame
        Dataframe with datetime.
    dtcol: str
        The datetime column

    RETURNS
    -------
    df: pd.DataFrame
        Dataframe with weekday one-hot variables.
    """
    if dtcol is None:
        df["weekday"] = df.index.day_name()
    else:
        df["weekday"] = df[dtcol].dt.day_name()

    df = pd.get_dummies(df, columns=["weekday"])

    return df
