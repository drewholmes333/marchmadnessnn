import torch
import torch.nn as nn

class MarchMadnessNN(nn.Module):
    def __init__(self, input_size=15, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=1):
        """
        A deep feed-forward Sequential neural network for March Madness predictions.
        Added an extra hidden layer for increased complexity fitting 10 features.
        """
        super(MarchMadnessNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3), # Increased dropout slightly to handle larger network capacity
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Dropout(0.2), # Original dropout value for final deep layer
            nn.Linear(hidden_size3, output_size),
            nn.Sigmoid() # Sigmoid for binary probability output (e.g. Win/Loss)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Initializing an example model to verify architecture
    example_input_features = 10
    model = MarchMadnessNN(input_size=example_input_features)
    
    print("Model initialized successfully!")
    print(model)
    
    # Passing a dummy input through the model
    dummy_input = torch.randn(10, example_input_features)
    predictions = model(dummy_input)
    print(f"Mock Output Predictions Shape: {predictions.shape}")
