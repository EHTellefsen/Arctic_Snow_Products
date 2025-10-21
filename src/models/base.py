from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr
import pickle

from src.data_src.gridded_data_sources.base import GriddedDataSource

class PixelPredictionModel(ABC):
    def __init__(self, model_params=None, grid=None):
        self.input_features = None
        self.target_feature = None
        self.model = None
        self.model_params = model_params if model_params is not None else {}
        self.grid = grid

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
    def train(self, X, y, weights=None):
        pass

    def predict(self, X):
        self._assert_initialized()
        
        if isinstance(X, pd.DataFrame):
            return self._predict_pandas_dataframe(X)
        
        elif isinstance(X, xr.Dataset):
            return self._predict_xarray_dataset(X)

        elif isinstance(X, GriddedDataSource):
            # Assert grid compatibility
            if self.grid is not None:
                if X.grid != self.grid:
                    raise ValueError(f"Input data grid {X.grid} does not match model grid {self.grid}. Please regrid the input data.")
            
            y = self._predict_xarray_dataset(X.data)
            return GriddedDataSource.from_xarray(y, X.grid)
        

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