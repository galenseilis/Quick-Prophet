'''Functions for imputing data in timeseries.'''


import pandas as pd

def fill_missing_dates(
    data, date_column="ds", value_column="y", group_column=None, fill_value=0
):
    """
    Fill missing dates between the min and max date within each group
    and assign the value of 0 for the specified column.
    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - date_column (str): Name of the datetime column.
    - value_column (str): Name of the column to fill missing values.
    - group_column (str): Name of the column representing groupings within the data.
    Returns:
    - pd.DataFrame: DataFrame with missing dates filled and values assigned.
    """
    # Ensure the date_column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column])

    # If grouping column is specified, group by it; otherwise, use the entire DataFrame
    groups = data.groupby(group_column) if group_column else [data]

    # Get the min and max date within each group
    min_date = data[date_column].min()
    max_date = data[date_column].max()

    # Iterate through groups
    filled_dfs = []
    for group_name, group_df in groups:

        # Generate a date range between min and max date
        date_range = pd.date_range(min_date, max_date, freq="D")

        # Create a DataFrame with the date range
        date_range_df = pd.DataFrame({date_column: date_range})

        # Merge the original DataFrame with the date range DataFrame, filling missing values with 0
        filled_df = pd.merge(
            date_range_df, group_df, on=date_column, how="left"
        ).fillna({value_column: fill_value})

        # Fill missing values in the group_column
        filled_df[group_column] = filled_df.apply(lambda row: group_name if pd.isnull(row[group_column]) else row[group_column], axis=1)

        filled_dfs.append(filled_df)

    # Concatenate the filled DataFrames
    result_df = pd.concat(filled_dfs, ignore_index=True)

    # Fill remaining NaN values in the value column
    result_df[value_column] = result_df[value_column].fillna(fill_value)

    return result_df
