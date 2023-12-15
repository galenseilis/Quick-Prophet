import pandas as pd
import numpy as np

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


def add_day_of_year_features(df, dtcol):
    """
    Add day-of-year features to a DataFrame based on a datetime column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a datetime column.
    dtcol : str or None, optional
        Name of the datetime column. If None, assumes the default column name 'ds'.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with added day-of-year features.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime

    >>> # Sample DataFrame
    >>> data = {'ds': pd.date_range(start='2023-01-01', end='2023-01-05'), 'value': [1, 2, 3, 4, 5]}
    >>> df = pd.DataFrame(data)

    >>> # Add day-of-year features
    >>> result_df = add_day_of_year_features(df, dtcol='ds')
    >>> print(result_df.head())

                      ds  value  dayofyear_1  dayofyear_2  dayofyear_3  dayofyear_4  dayofyear_5
    0 2023-01-01 00:00:00      1            1            0            0            0            0
    1 2023-01-02 00:00:00      2            0            1            0            0            0
    2 2023-01-03 00:00:00      3            0            0            1            0            0
    3 2023-01-04 00:00:00      4            0            0            0            1            0
    4 2023-01-05 00:00:00      5            0            0            0            0            1

    Notes
    -----
    This function creates new binary columns for each day of the year, indicating whether
    the original datetime values fall on that particular day.

    The resulting DataFrame is obtained by concatenating the original DataFrame with
    one-hot encoded day-of-year features.

    If dtcol is None, it uses the default column name 'ds' for the datetime column.
    """
    if dtcol is None:
        day_of_year = df.ds.dt.dayofyear
    else:
        day_of_year = df.ds.dt.dayofyear

    return pd.concat((df, pd.get_dummies(day_of_year, prefix='dayofyear')), axis=1)
