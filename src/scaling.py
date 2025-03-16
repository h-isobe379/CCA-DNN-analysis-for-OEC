from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataScaler:
    def __init__(self, scale_method, all_elements=False):
        self.scale_method = scale_method
        self.all_elements = all_elements

    def scale_data(self, X, Y):
        if self.scale_method == 'minmax':
            return self._scale_with_scaler(X, Y, MinMaxScaler())
        elif self.scale_method == 'zscore':
            return self._scale_with_scaler(X, Y, StandardScaler())
        else:
            raise ValueError("Invalid scale method. Choose either 'minmax' or 'zscore'.")

    def _scale_with_scaler(self, X, Y, scaler):
        if self.all_elements:
            X_scaled = scaler.fit_transform(X.values.reshape(-1, 1)).reshape(X.shape)
            Y_scaled = scaler.fit_transform(Y.values.reshape(-1, 1)).reshape(Y.shape)
        else:
            X_scaled = scaler.fit_transform(X)
            Y_scaled = scaler.fit_transform(Y)
        return X_scaled, Y_scaled

