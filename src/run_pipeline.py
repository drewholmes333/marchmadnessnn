import sys
import os

# Align python path to ensure cross-directory importing works natively
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models', 'src'))

from data_loader import load_historical_data
from train import train_model

def load_data():
    """
    Wrapper function fulfilling the data loading from the data/raw/ folder.
    """
    print("Step 1: Loading raw Kaggle CSV subsets from data/raw/ ...")
    matchups, feature_cols = load_historical_data()
    return matchups, feature_cols

def run_pipeline():
    # 1. Run the data verifier pipeline
    load_data()
    
    # 2. Trigger the training validation metric evaluation
    print("\nStep 2: Executing Model Training...")
    train_model()
    
    # 3. Final readout
    print("\nPipeline Complete: Weights Updated with Kaggle Data.")

if __name__ == '__main__':
    run_pipeline()
