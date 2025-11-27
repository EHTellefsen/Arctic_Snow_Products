# -- coding: utf-8 --
# setup.py
"""Script for setting up data folders for the project."""

# -- built-in libraries --
import os

# -- third party libraries --

# -- custom modules --

########################################################################
if __name__ == "__main__":
    # Ensure the script runs from the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Define required directories
    directories = [
        "./data/fig",
        "./data/raw",
        "./data/intermediate",
        "./data/intermediate/CV_results",
        "./data/intermediate/datasets",
        "./data/processed",
        "./data/processed/models",        
        "./data/processed/predictions",
        "./data/processed/predictions_aggregated"
    ]

    # Create directories if they don't exist
    print("Setting up data folders...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created or already exists: {directory}")
    print("Data folders setup complete.")