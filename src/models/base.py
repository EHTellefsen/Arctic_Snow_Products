# -- coding: utf-8 --
# base.py
"""Base class for pixel-based prediction models."""

# -- built-in modules --
from abc import ABC, abstractmethod

# -- third-party modules --
import pandas as pd
import xarray as xr
import pickle
from sklearn.metrics import root_mean_squared_error

# -- custom modules --
from src.data_src.gridded_data_sources.base import GriddedDataSource

#######################################################
class PixelPredictionModel(ABC):
    """ Abstract base class for pixel-based prediction models. """
    def __init__(self,
                 input_features=None, 
                 target_feature=None, 
                 weight_feature=None, 
                 ):
        self.input_features = input_features
        self.target_feature = target_feature
        self.weight_feature = weight_feature
        self.is_fitted = False


    # %% Representation
    def __repr__(self):
        """ String representation of the model. """
        print(f"PixelPredictionModel with input features: {self.input_features} and target feature: {self.target_feature}")
        return ""

    # %% initialization checks
    def _assert_initialized(self):
        """ Check if the model has been properly initialized and trained. """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")
        if self.input_features is None:
                raise ValueError("Model features are not defined.")
        if self.target_feature is None:
                raise ValueError("Model target feature is not defined.")

    # %% Save and Load
    def save(self, path: str):
        """ Save the model to the specified path. """
        self._assert_initialized()
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """ Load a saved model from the specified path. """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise ValueError(f"Loaded object is not of type {cls.__name__}")
        return model


    # %% Training and Prediction
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def _predict(self, X):
        pass        

    def predict(self, data: pd.DataFrame | xr.Dataset | GriddedDataSource, mask = None) -> pd.Series | xr.DataArray | GriddedDataSource:
        """
        Predict using the model on given data.
        The data can be a pandas DataFrame, an xarray Dataset, or a GriddedDataSource.
        An optional mask can be provided to specify valid data points for prediction.
        """
        self._assert_initialized()

        if isinstance(data, pd.DataFrame):
            return self._predict_pandas_dataframe(data, mask)

        elif isinstance(data, xr.Dataset):
            return self._predict_xarray_dataset(data, mask)

        elif isinstance(data, GriddedDataSource):
            y = self._predict_xarray_dataset(data.data, mask)
            return GriddedDataSource.from_xarray(y, data.grid)

    def _predict_pandas_dataframe(self, X: pd.DataFrame, mask = None) -> pd.Series:
        """
        Predict using a pandas DataFrame as input.
        If a mask is provided, it should be a boolean Series with the same index as X,
        containing True for valid data points and False for invalid ones.
        """
        missing_features = set(self.input_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Input data is missing required features: {missing_features}")
        X = X[self.input_features]

        if mask is not None:
            X = X[mask]

        y = self._predict(X)

        if mask is not None:
            full_y = pd.Series(index=mask.index, dtype=y.dtype)
            full_y[mask] = y
            return full_y.rename(self.target_feature)
        else:
            return pd.Series(y, index=X.index, name=self.target_feature)

    def _predict_xarray_dataset(self, X: xr.Dataset, mask = None) -> xr.DataArray:
        """ 
        Predict using an xarray Dataset as input.
        If a mask is provided, it should be an xarray DataArray with the same dimensions as X,
        containing boolean values where True indicates valid data points for prediction.
        """

        if 'doy' in self.input_features and 'doy' not in X.data_vars:
            X['doy'] = X['time'].dt.dayofyear

        missing_features = set(self.input_features) - set(X.data_vars)
        if missing_features:
            raise ValueError(f"Input data is missing required features: {missing_features}")
        
        # keep only input vars
        cX = X.drop_vars([var for var in X.data_vars if var not in self.input_features])
        dims = list(cX.dims)  # use only true dimension names
        coord_order = {d: cX[d].values for d in dims}
        
        # flatten
        X_df = cX.to_dataframe().reset_index()

        # apply mask if provided
        if mask is not None:
            # convert mask to Series aligned with X_df index
            mask_df = mask.to_dataframe(name="mask").reset_index()["mask"]
            X_df = X_df[mask_df]  # keep only rows where mask==True            

        # drop NaNs
        X_df = X_df.dropna()

        # Predict and reconstruct xarray
        X_df[self.target_feature] = self._predict(X_df[self.input_features])

        # reconstruct xarray
        y_df = X_df[dims + [self.target_feature]].set_index(dims)
        y_ds = y_df.to_xarray().reindex(coord_order)

        #explicitly put NaN where mask == False (for safety/clarity)
        if mask is not None:
            y_ds = y_ds.where(mask)

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