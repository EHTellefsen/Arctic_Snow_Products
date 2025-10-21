from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr
import pickle
from sklearn.metrics import root_mean_squared_error

from src.data_src.gridded_data_sources.base import GriddedDataSource

class PixelPredictionModel(ABC):
    def __init__(self,
                 input_features=None, 
                 target_feature=None, 
                 weight_feature=None, 
                 model_params=None
                 ):
        self.input_features = input_features
        self.target_feature = target_feature
        self.weight_feature = weight_feature
        self.model_params = model_params

        self.model = None


    # %% Representation
    def __repr__(self):
        print(f"PixelPredictionModel with input features: {self.input_features} and target feature: {self.target_feature}")
        return ""

    # %% initialization checks
    def _assert_initialized(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if self.input_features is None:
                raise ValueError("Model features are not defined.")
        if self.target_feature is None:
                raise ValueError("Model target feature is not defined.")

    # %% Save and Load
    def save(self, path: str):
        self._assert_initialized()
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise ValueError(f"Loaded object is not of type {cls.__name__}")
        return model


    # %% Training and Prediction
    @abstractmethod
    def fit(self, data: pd.DataFrame, model_params: dict = None):
        pass

    def predict(self, data: pd.DataFrame | xr.Dataset | GriddedDataSource) -> pd.Series | xr.DataArray | GriddedDataSource:
        self._assert_initialized()

        if isinstance(data, pd.DataFrame):
            return self._predict_pandas_dataframe(data)

        elif isinstance(data, xr.Dataset):
            return self._predict_xarray_dataset(data)

        elif isinstance(data, GriddedDataSource):
            y = self._predict_xarray_dataset(data.data)
            return GriddedDataSource.from_xarray(y, data.grid)

    def _predict_pandas_dataframe(self, X: pd.DataFrame) -> pd.Series:
        missing_features = set(self.input_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Input data is missing required features: {missing_features}")
        X = X[self.input_features]
        y = self.model.predict(X)
        return pd.Series(y, index=X.index, name=self.target_feature)
    

    def _predict_xarray_dataset(self, X: xr.Dataset) -> xr.DataArray:
        missing_features = set(self.input_features) - set(X.data_vars)
        if missing_features:
            raise ValueError(f"Input data is missing required features: {missing_features}")
        
        # Prepare data for prediction
        cX = X.drop_vars([var for var in X.data_vars if var not in self.input_features])
        dims = list(cX.dims)  # use only true dimension names
        coord_order = {d: cX[d].values for d in dims}
        X_df = cX.to_dataframe().reset_index()
        X_df = X_df.dropna()

        # Predict and reconstruct xarray
        X_df[self.target_feature] = self.model.predict(X_df[self.input_features])
        y_df = X_df[dims + [self.target_feature]].set_index(dims)
        y_ds = y_df.to_xarray().reindex(coord_order)
        return y_ds
    

    def evaluate(self, data: pd.DataFrame, metric = root_mean_squared_error) -> float:
        """
        Evaluate model performance on given data using specified metric.
        """
        # ---- Normalize input ----
        X = data[self.input_features]
        y = data[self.target_feature]
        w = data[self.weight_feature] if self.weight_feature is not None else None

        # ---- Evaluate ----
        self._assert_initialized()
        y_pred = self.predict(X)
        return metric(y, y_pred, sample_weight=w)