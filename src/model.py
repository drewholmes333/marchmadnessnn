import torch
import torch.nn as nn

class MarchMadnessNN(nn.Module):
    def __init__(self, input_size=15, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=1):
        """
        Deep feed-forward neural network for March Madness predictions.
        Architecture: BatchNorm + Dropout(0.2) after every hidden layer.
        Output: Raw logits (no Sigmoid) for BCEWithLogitsLoss.
        """
        super(MarchMadnessNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size3, output_size)
            # No Sigmoid — BCEWithLogitsLoss handles it internally
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    example_input_features = 15
    model = MarchMadnessNN(input_size=example_input_features)
    
    print("Model initialized successfully!")
    print(model)
    
    dummy_input = torch.randn(10, example_input_features)
    predictions = model(dummy_input)
    print(f"Mock Output (Raw Logits) Shape: {predictions.shape}")
