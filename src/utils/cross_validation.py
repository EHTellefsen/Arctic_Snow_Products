# -- coding: utf-8 --
# cross_validation.py
"""Module for performing cross-validation with grid search for hyperparameter tuning."""

# -- built-in libraries --
import yaml
import multiprocessing as mp
import pickle

# -- third-party libraries  --
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

#  -- custom modules  --

###########################################################
class CrossValidation:
    """Class to perform cross-validation with grid search for hyperparameter tuning."""
    def __init__(self, model, model_configs, param_grid):
        self.model = model
        self.model_configs = model_configs
        self.param_grid = param_grid
        self.performance_cube = None
        self.best_mse = None
        self.best_params = None

    # %% Functions for cross-validation
    def _prepare_grid(self, n_splits):
        """Prepare list of parameter and fold combinations for evaluation."""
        grid = list(ParameterGrid(self.param_grid))
        tests = []
        for g in grid:
            for fold in range(n_splits):
                tests.append((g, fold))
        return tests

    def _make_random_splits(self, data, n_splits, random_state):
        """Create random K-Fold splits."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(data))
        return splits
    
    def _make_temporal_splits(self, data, time_column, n_splits):
        """Create temporal splits based on time column."""
        data_sorted = data.sort_values(by=time_column)
        n_samples = len(data_sorted)
        fold_size = n_samples // n_splits
        splits = []
        for fold in range(n_splits):
            val_start = fold * fold_size
            if fold == n_splits - 1:
                val_end = n_samples
            else:
                val_end = (fold + 1) * fold_size
            val_indices = data_sorted.index[val_start:val_end].tolist()
            train_indices = data_sorted.index.difference(val_indices).tolist()
            splits.append((train_indices, val_indices))
        return splits

    def _run_grid_search(self, data, tests, splits, nproc):
        """Run grid search with cross-validation."""
        if nproc > 1:
            with mp.Pool(processes=nproc) as pool:
                results = list(tqdm(pool.imap(self._evaluate_param_fold, 
                                               [(test, splits, data) for test in tests]),
                                    total=len(tests)))
        else:
            results = list(tqdm(map(self._evaluate_param_fold, 
                                     [(test, splits, data) for test in tests]),
                                total=len(tests)))
        return results

    def _evaluate_param_fold(self, args):
        """Helper function to evaluate a single parameter set on a single fold."""
        (params, fold), splits, data = args
        
        train_idx, val_idx = splits[fold]
        train = data.iloc[train_idx]
        val = data.iloc[val_idx]

        # Train model
        model_instance = self.model(model_params=params, **self.model_configs)
        model_instance.fit(train)

        # Evaluate on validation set
        y_pred = model_instance.predict(val)
        mse = mean_squared_error(
            val[self.model_configs['target_feature']], 
            y_pred, 
            sample_weight=val[self.model_configs['weight_feature']])
        
        return (params, fold, mse)

    def _prepare_performance_cube(self, results):
        """Prepare performance cube from results."""
        df = pd.DataFrame([
            {k: str(v) for k, v in param_dict.items()}  # ← force every param to string
            | dict(cv_fold=cv_fold, mse=mse)           # merge cv + mse
            for param_dict, cv_fold, mse in results
        ])
        
        self.performance_cube = df.set_index(['cv_fold'] + list(self.param_grid.keys()))
        self.performance_cube = self.performance_cube.to_xarray()
        return self.performance_cube

    def _grid_search(self, data, splits, n_splits, nproc):
        """Perform grid search with cross-validation."""
        tests = self._prepare_grid(n_splits)
        results = self._run_grid_search(data, tests, splits, nproc)
        self._prepare_performance_cube(results)
        return self        

    # %% run methods
    def random_split_grid_search(self, data, n_splits=5, nproc=1, random_state=None):
        """Perform grid search with random split cross-validation."""
        splits = self._make_random_splits(data, n_splits, random_state)
        return self._grid_search(data, splits, n_splits, nproc)
    
    def temporal_split_grid_search(self, data, time_column, n_splits=5, nproc=1):
        """Perform grid search with temporal cross-validation."""
        splits = self._make_temporal_splits(data, time_column, n_splits)
        return self._grid_search(data, splits, n_splits, nproc)

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
    
    x_out = x.values.squeeze()  # 0-D DataArray to scalar
    return params, float(x_out)