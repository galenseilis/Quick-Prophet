from sklearn.base import BaseEstimator, RegressorMixin

# TODO: By default assume missing dates are zeros
# TODO: By default assume COVI-19 holidays are the year of 2020
# TODO: groupby some columns of choice.
class GroupedAutoProphet(BaseEstimator, RegressorMixin):
    '''Provides some easy defaults for an end-to-end
        forecast using different groupings in the data.
    '''

    def __init__(self, groups, grouped_params=None, t_ind=None:Iterable):
        '''
        Args:
            groups (Iterable): Collection of immutable groups.
            grouped_paras (Dict): Group-specific parameters for the Prophet model.
            t_ind (Iterable[str]): Temporal indicator variables. Supported values include `weekday`, `month`, `year`. Default is `('weekday',)`.
        '''
        self.groups = groups        
        
        if grouped_params is not None:

            if set(grouped_params.keys()) - set(groups):
                raise ValueError('All groups in grouped_params must also be in groups.')

            for group in self.groups:
                
                
                
            
        else:

            self.grouped_params = {}

            for group in groups:
                self.grouped_params[group] = {}
                self.grouped_params[group]['country'] = 'CA' # Sorry, I am Canadian ¯\_(ツ)_/¯
                # TODO: Set COVID block here
                self.grouped_params[group] = ('weekday',) # Weekdays almost always explain human behaviour.
                


    def fit(self, X, y=None):
        '''
        '''
        self.models = {}

        for group, group_df in X.groupby(by=self.groups):
            ...

        return self

    def predict(self, X):
        '''
        Args:
            X (Iterable[datetime]): Sequence of dates for prediction.

        Returns:
            pd.DataFrame: Grouped forecasts
        '''

        forecasts = None
        for group, group_df in X.groupby(by=self.groups):
            ...
        return forecasts
        
