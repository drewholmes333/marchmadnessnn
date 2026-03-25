# March Madness Prediction Neural Network

This repository contains the neural network architecture, data parsing logic, and execution steps for predicting March Madness outcomes using KenPom and Torvik datasets.

## Data Flow

1.  **Raw Data Collection**: The raw data consists of two comma-separated files: `kenpom.csv` and `torvik.csv`. These files provide robust statistical coverage of each team's performance, ranking, and situational metrics throughout the college basketball season.
2.  **Data Initialization & Reading**: The `data_loader.py` script reads the aforementioned datasets utilizing the `pandas` library. The script successfully interprets the CSV structure and loads the files into memory as structured DataFrames, paving the way for further cleaning operations.
3.  **Data Preprocessing (Future Step)**: Features originating from the Torvik and KenPom DataFrames will be synchronized by team names and year, merged, and properly normalized or encoded. These features will construct the input tensors. 
4.  **Neural Network Inference**: The combined, preprocessed metrics array will be passed into the PyTorch model located in `model.py`. The model is a feed-forward `Sequential` network structured with fully connected `Linear` layers, `ReLU` activations to maintain non-linearity, and intermediate `Dropout` layers to mitigate model overfitting. The final layer activates the predictions into a discrete probability between `0` and `1` utilizing a `Sigmoid` function.

## Setup Instructions

1.  Ensure you have your Python virtual environment activated:
    *   Windows: `.\venv\Scripts\activate`
2.  Install all required dependencies (if you haven't already):
    *   `pip install torch pandas numpy scikit-learn`
3.  Place your `kenpom.csv` and `torvik.csv` tabular files inside the project's root directory.
4.  Run validation tests locally:
    *   `python data_loader.py`
    *   `python model.py`

## Project Structure
├── data_loader.py        # Loads and structures raw datasets
├── feature_engineering/  # Data preprocessing & feature creation
├── model.py              # Neural network architecture
├── train.py              # Model training pipeline
├── predict.py            # Matchup prediction script
├── data/                 # Place datasets here
└── README.md

## Data Pipeline
Data Collection: 
Uses KenPom and Torvik datasets containing team efficiency, rankings, and performance metrics. 

Data Loading: 
- `data_loader.py` reads CSV files into structured pandas DataFrames.

Feature Engineering (in progress / customizable):  
- Merge datasets by team and season.
- Normalize and encode features.
- Construct model-ready input tensors 

Model Inference:
- Feed-forward neural network (PyTorch)
- Linear layers + ReLU activations
- Dropout for regularization
- Sigmoid output for win probability (0–1)

## Make a prediction
`python predict.py "Team A" "Team B"`
