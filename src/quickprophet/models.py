import pandas as pd

class BatchCOVIDLogisticProphet:
    def __init__(self, group_cols, floor=0, cap=7.5 * 60 / 8, datalag=26):
        '''
        datalag (int): Most recent number of weeks to treat as special dates.
        '''

        if not (isinstance(group_cols, list) and len(group_cols) >= 1):
            raise ValueError(
                "Must specify a list containing at least one column name to group by."
            )

        self.group_cols = group_cols
        self.floor = floor
        self.cap = cap

        # Prepare special dates for COVID 19
        self.covid_block = pd.DataFrame(
            [
                {
                    "holiday": "covid19",
                    "ds": "2020-01-01",
                    "lower_window": 0,
                    "ds_upper": "2021-01-01",
                }
            ]
        )

        for t_col in ["ds", "ds_upper"]:
            self.covid_block[t_col] = pd.to_datetime(self.covid_block[t_col])

        self.covid_block["upper_window"] = (
            self.covid_block["ds_upper"] - self.covid_block["ds"]
        ).dt.days

        # Include data lag if specified.
        if datalag:
            self.data_lag_block = pd.DataFrame(
                [
                    {
                        "holiday": "data_lag",
                        "ds": pd.Timestamp.today() - pd.Timedelta(weeks=datalag),
                        "lower_window": 0,
                        "ds_upper": pd.Timestamp.today(),
                    }
                ]
            )

            self.data_lag_block["upper_window"] = (
                self.data_lag_block["ds_upper"] - self.data_lag_block["ds"]
            ).dt.days

            self.holidays = pd.concat((self.covid_block, self.data_lag_block))
        
        else:
            self.holidays = self.covid_block

    def fit(self, data):
        self.models = {}
        for group, group_df in data.groupby(self.group_cols):
            print(f"Training Prophet model for {group}.")

            # Groups are assumed to not be predictors. Make additional predictor columns with the same info if
            # you need to reuse them.
            group_df.drop(columns=self.group_cols, inplace=True)

            self.models[group] = Prophet(holidays=self.holidays, growth="logistic")

            # Extra predictors
            extra_predictors = [
                col for col in group_df.keys() if col not in ["ds", "y"]
            ]
            for predictor in extra_predictors:
                self.models[group].add_regressor(predictor)

            # Saturation Effects
            group_df["floor"] = self.floor
            group_df["cap"] = self.cap

            # Train
            self.models[group].fit(group_df)

        return self
