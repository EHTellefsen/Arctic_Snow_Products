import yaml

import pandas as pd

from src.utils.cross_validation import CrossValidation
from src.models.random_forest_regression import RandomForestRegression

if __name__ == "__main__":
    # Load configuration
    with open("configs/pipeline_configs/perform_RFR_cross_validation.yaml", 'r') as f:
        config = yaml.safe_load(f)

    cv_folds = config['cv_folds']
    model_configs = config['model']['configs']
    param_grid = config['model']['param_grid']
    for key, values in param_grid.items():
        param_grid[key] = [None if v == 'None' else v for v in values]

    # Initialize CrossValidation
    cross_validator = CrossValidation(
        model=RandomForestRegression,
        model_configs=model_configs,    
        param_grid=param_grid,
        cv_folds=cv_folds
    )

    # Load training and validation data
    train_data = pd.read_parquet(config['input_data']['train_data_path'])  # Load your training data as a pandas DataFrame
    train_data = train_data[train_data['primary_id'].isin(config['input_data']['primary_ids']) & train_data['secondary_id'].isin(config['input_data']['secondary_ids'])]

    # Perform grid search with cross-validation
    cross_validator = cross_validator.perform_grid_search(train_data, nproc=config['n_jobs'], random_state=config['random_state'])

    # Access results
    cross_validator.save_cv_results(config['output']['cv_results_path'])
    cross_validator.save_performance_cube(config['output']['performance_cube_path'])
    best_params, best_mse = cross_validator.get_best_params()
    print("Best Parameters:", best_params)
    print("Best MSE:", best_mse)
    cross_validator.save_best_params(config['output']['best_params_path'])
