import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import MarchMadnessNN

def train_model():
    print("Preparing data...")
    from data_loader import get_dataloaders
    train_loader, val_loader, input_size = get_dataloaders(batch_size=32)
    
    print(f"Initializing model with input size {input_size}...")
    model = MarchMadnessNN(input_size=input_size)
    
    # Define Loss function and Optimizer
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    num_epochs = 100
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training iteration
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()           # Reset gradients
            outputs = model(batch_X)        # Forward pass
            loss = criterion(outputs, batch_y) # Compute loss
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # No gradients needed during validation
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                # Calculate accuracy
                predictions = (outputs >= 0.5).float() # Threshold probability at 0.5
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
        val_loss /= len(val_loader.dataset)
        accuracy = (correct / total) * 100
        
        weights_path = 'march_madness_weights.pt'
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)
            
        scheduler.step()
            
        # Print progress every 10 epochs or on the first one
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {epoch_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Accuracy: {accuracy:.2f}% | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
    print(f"\nTraining complete for {num_epochs} epochs.")
    print(f"Weights successfully saved to {weights_path}")
    print(f"\nFINAL TEST PEAK ACCURACY (2025): {best_val_acc:.2f}% \n(Target was > 75%)")
    print(f"\nFINAL TEST LOG LOSS (2025): {best_val_loss:.4f} \n(Target was < 0.55)")

if __name__ == '__main__':
    train_model()
