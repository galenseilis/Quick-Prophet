"""Model classes based on Facebook's Prophet."""

import itertools

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation

try:
    from . import features
except ImportError:
    import features

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

    def predict(self, periods=365 * 10, weekday=True, dayofyear=True, nonneg=True, noholiday=True):#TODO: Pass in a "future prep" function for preparing features
        """Predict a given number of periods into the future.

        Args:
        periods (int): Number of periods to forecast into the future.
        nonneg (bool): Whether to truncate negative values to zero. (default=True)
        noholiday (bool): Whether to subtract holiday effects from forecast.

        Returns:
        forecasts (pd.DataFrame): Forecast for each facility.
        """

        forecasts = None

        # Prepare future inputs
        for group in self.models:
            print(f"Forecasting group {group}")
            future = self.models[group].make_future_dataframe(periods=periods)
            if weekday:
                future = features.add_weekday_features(future, "ds")

            if dayofyear:
                future = features.add_day_of_year_features(future, "ds")
                
            future["floor"] = self.floor
            future["cap"] = self.cap

            forecast = self.models[group].predict(future)

            forecast["groups"] = (group,) * forecast.shape[0]

            if forecasts is None:
                forecasts = forecast
            else:
                forecasts = pd.concat((forecasts, forecast))

        forecasts.columns = (
            forecasts.columns.str.replace("'", "")
            .str.replace(" ", "_")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.upper()
        )

        if noholiday:
            forecasts["YHAT_LOWER"] = forecasts["YHAT_LOWER"] - forecasts["HOLIDAYS_LOWER"]
            forecasts["YHAT"] = forecasts["YHAT"] - forecasts["HOLIDAYS"]
            forecasts["YHAT_UPPER"] = forecasts["YHAT_UPPER"] - forecasts["HOLIDAYS_UPPER"]

        if nonneg:
            for col in ["YHAT_LOWER", "YHAT", "YHAT_UPPER"]:
                forecasts[col] = forecasts[col].apply(lambda x: max(x, 0))

        return forecasts

    def cv(self, param_grid=None):
        """Cross validate."""
        raise NotImplementedError('Sorry, not ready yet.')

        # TODO: Need to do the following for each group.
        if param_grid is None:
            param_grid = {
                        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params).fit(df)  # Fit model with given params
            df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        print(tuning_results)

        return self


class BatchCOVIDFlatProphet:
    def __init__(self, group_cols, datalag=26):
        '''
        datalag (int): Most recent number of weeks to treat as special dates.
        '''

        if not (isinstance(group_cols, list) and len(group_cols) >= 1):
            raise ValueError(
                "Must specify a list containing at least one column name to group by."
            )

        self.group_cols = group_cols

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

            self.models[group] = Prophet(holidays=self.holidays, growth="flat")

            # Extra predictors
            extra_predictors = [
                col for col in group_df.keys() if col not in ["ds", "y"]
            ]
            for predictor in extra_predictors:
                self.models[group].add_regressor(predictor)

            # Train
            self.models[group].fit(group_df)

        return self

    def predict(self, periods=365 * 10, weekday=True, dayofyear=True, nonneg=True, noholiday=True):#TODO: Pass in a "future prep" function for preparing features
        """Predict a given number of periods into the future.

        Args:
        periods (int): Number of periods to forecast into the future.
        nonneg (bool): Whether to truncate negative values to zero. (default=True)
        noholiday (bool): Whether to subtract holiday effects from forecast.

        Returns:
        forecasts (pd.DataFrame): Forecast for each facility.
        """

        forecasts = None

        # Prepare future inputs
        for group in self.models:
            print(f"Forecasting group {group}")
            future = self.models[group].make_future_dataframe(periods=periods)
            if weekday:
                future = features.add_weekday_features(future, "ds")

            if dayofyear:
                future = features.add_day_of_year_features(future, "ds")

            forecast = self.models[group].predict(future)

            forecast["groups"] = (group,) * forecast.shape[0]

            if forecasts is None:
                forecasts = forecast
            else:
                forecasts = pd.concat((forecasts, forecast))

        forecasts.columns = (
            forecasts.columns.str.replace("'", "")
            .str.replace(" ", "_")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.upper()
        )

        if noholiday:
            forecasts["YHAT_LOWER"] = forecasts["YHAT_LOWER"] - forecasts["HOLIDAYS_LOWER"]
            forecasts["YHAT"] = forecasts["YHAT"] - forecasts["HOLIDAYS"]
            forecasts["YHAT_UPPER"] = forecasts["YHAT_UPPER"] - forecasts["HOLIDAYS_UPPER"]

        if nonneg:
            for col in ["YHAT_LOWER", "YHAT", "YHAT_UPPER"]:
                forecasts[col] = forecasts[col].apply(lambda x: max(x, 0))

        return forecasts

    def cv(self, param_grid=None):
        """Cross validate."""
        raise NotImplementedError('Sorry, not ready yet.')

        # TODO: Need to do the following for each group.
        if param_grid is None:
            param_grid = {
                        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params).fit(df)  # Fit model with given params
            df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        print(tuning_results)

        return self
