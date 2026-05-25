from sklearn.base import BaseEstimator, TransformerMixin

class BookingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rv_median = None
        self.route_mean = None
        self.vehicle_scale = None
        self.v_median = None
        self.global_mean = None

    def fit(self, X, y=None):
        X = X.copy()

        # Create route
        X['route'] = X['Pickup Location'] + "_" + X['Drop Location']

        # Store global mean (final fallback)
        self.global_mean = X['Booking Value'].mean()

        # Route + Vehicle median
        self.rv_median = X.groupby(
            ['Vehicle Type', 'Pickup Location', 'Drop Location']
        )['Booking Value'].median()

        # Route mean
        self.route_mean = X.groupby('route')['Booking Value'].mean()

        # Vehicle median
        self.v_median = X.groupby('Vehicle Type')['Booking Value'].median()

        # Vehicle scale
        self.vehicle_scale = self.v_median / self.global_mean

        return self

    def transform(self, X):
        X = X.copy()

        # Create route
        X['route'] = X['Pickup Location'] + "_" + X['Drop Location']

        # Merge mappings
        X = X.merge(self.rv_median.rename('rv_median'),
                    on=['Vehicle Type', 'Pickup Location', 'Drop Location'],
                    how='left')

        X = X.merge(self.route_mean.rename('route_mean'),
                    on='route',
                    how='left')

        X = X.merge(self.vehicle_scale.rename('vehicle_scale'),
                    on='Vehicle Type',
                    how='left')

        X = X.merge(self.v_median.rename('v_median'),
                    on='Vehicle Type',
                    how='left')

        # Step 1
        X['Booking Value'] = X['Booking Value'].fillna(X['rv_median'])

        # Step 2
        X['scaled_route_price'] = X['route_mean'] * X['vehicle_scale']
        X['Booking Value'] = X['Booking Value'].fillna(X['scaled_route_price'])

        # Step 3
        X['Booking Value'] = X['Booking Value'].fillna(X['v_median'])

        # Final fallback 
        X['Booking Value'] = X['Booking Value'].fillna(self.global_mean)

        # Cleanup
        X.drop(columns=[
            'rv_median', 'route_mean', 'vehicle_scale',
            'v_median', 'scaled_route_price', 'route'
        ], inplace=True)

        return X