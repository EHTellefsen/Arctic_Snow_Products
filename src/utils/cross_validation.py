import yaml
import multiprocessing as mp
import pickle

import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

class CrossValidation:
    def __init__(self, model, model_configs, param_grid, cv_folds):
        self.model = model
        self.model_configs = model_configs
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.performance_cube = None
        self.best_mse = None
        self.best_params = None

    # %% Functions for cross-validation
    def _evaluate_param_fold(self, args):
        """Helper function to evaluate a single parameter set on a single fold."""
        (params, fold), splits, train_data, val_data = args
        
        train_idx, _ = splits[fold]
        train = train_data.iloc[train_idx]

        # Train model
        model_instance = self.model(model_params=params, **self.model_configs)
        model_instance.fit(train)

        # Evaluate on validation set
        y_pred = model_instance.predict(val_data)
        mse = mean_squared_error(
            val_data[self.model_configs['target_feature']], 
            y_pred, 
            sample_weight=val_data[self.model_configs['weight_feature']])
        
        return (params, fold, mse)

    
    def perform_grid_search(self, train_data, val_data, nproc=1, random_state=None):
        """Perform grid search with cross-validation."""

        grid = list(ParameterGrid(self.param_grid))
        tests = []
        for g in grid:
            for fold in range(self.cv_folds):
                tests.append((g, fold))

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(train_data))

        if nproc > 1:
            with mp.Pool(processes=nproc) as pool:
                results = list(tqdm(pool.imap(self._evaluate_param_fold, 
                                               [(test, splits, train_data, val_data) for test in tests]),
                                    total=len(tests)))
        else:
            results = list(tqdm(map(self._evaluate_param_fold, 
                                     [(test, splits, train_data, val_data) for test in tests]),
                                total=len(tests)))
        
        df = pd.DataFrame([
            {k: str(v) for k, v in param_dict.items()}  # ← force every param to string
            | dict(cv_fold=cv_fold, mse=mse)           # merge cv + mse
            for param_dict, cv_fold, mse in results
        ])
        
        self.performance_cube = df.set_index(['cv_fold'] + list(self.param_grid.keys()))
        self.performance_cube = self.performance_cube.to_xarray()

        return self
    
    # %% save/load
    def save_cv_results(self, filepath):
        """Save cross-validation results to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_cv_results(cls, filepath):
        """Load cross-validation results from a file."""
        with open(filepath, 'rb') as f:
            cv_instance = pickle.load(f)
        if not isinstance(cv_instance, cls):
            raise ValueError(f"Loaded object is not of type {cls.__name__}")
        return cv_instance


    def get_best_params(self):
        """Get the best parameters based on cross-validation results."""
        if self.performance_cube is None:
            raise ValueError("No performance data available. Please run perform_grid_search first.")
        
        mean = self.performance_cube.mse.mean('cv_fold')
        best_idx = mean.where(mean == mean.min(), drop=True)
        self.best_params, self.best_mse = reverse_xarray_selection_to_params(best_idx)
        return self.best_params, self.best_mse

    def save_performance_cube(self, filepath):
        """Save the performance cube to a NetCDF file."""
        if self.performance_cube is None:
            raise ValueError("No performance data available. Please run perform_grid_search first.")
        self.performance_cube.to_netcdf(filepath)

    def save_best_params(self, filepath):
        """Save the best parameters to a YAML file."""
        if self.best_params is None:
            raise ValueError("Best parameters not found. Please run get_best_params first.")
        
        with open(filepath, 'w') as f:
            yaml.dump(self.best_params, f)

    def make_model_with_best_params(self):
        """Instantiate a model with the best parameters found."""
        if self.best_params is None:
            raise ValueError("Best parameters not found. Please run get_best_params first.")
        
        return self.model(model_params=self.best_params, **self.model_configs)

# %% Helper function
def reverse_xarray_selection_to_params(x):
    """
    Given e.g. x = ds.mse.sel(...)
    return the corresponding Python param dict, restoring types:
      "None" → None
      "10"   → int(10)
      "3.14" → float(3.14)
      everything else stays as string.
    """
    # If user passed Dataset/DataArray — ensure 0-D DataArray

    params = {}
    for dim, raw in x.coords.items():
        v = raw.item()  # Python scalar

        # --- unstringify back to meaningful Python literal ---
        if v == "None":
            params[dim] = None
        else:
            try:
                params[dim] = int(v)
            except (ValueError, TypeError):
                try:
                    params[dim] = float(v)
                except (ValueError, TypeError):
                    params[dim] = v  # leave as string

    return params, float(x.values.squeeze())