import yaml

import pandas as pd

from src.models.random_forest_regression import RandomForestRegression

if __name__ == "__main__":
    # Load configuration
    with open("configs/pipeline_configs/create_RFR_model.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_configs = config['model']['configs']

    with open(config['model']['param_file'], 'r') as f:
        model_params = yaml.safe_load(f)
    
    fixed_params = config['model']['fixed_params']
    for p in fixed_params.keys():
        model_params[p] = fixed_params[p]

    train_data_path = config['input_data']['train_data_paths']
    if isinstance(train_data_path, str):
        train_data_path = [train_data_path]

    train_data = pd.concat([pd.read_parquet(f) for f in train_data_path], ignore_index=True)
    train_data = train_data[train_data['primary_id'].isin(config['input_data']['primary_ids']) & train_data['secondary_id'].isin(config['input_data']['secondary_ids'])]

    # Initialize and create model
    model = RandomForestRegression(model_params=model_params, **model_configs)
    model.fit(train_data)
    model.save(config['output']['model_path'])
