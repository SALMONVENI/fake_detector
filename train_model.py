# train_model.py

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim

# Paths - handle nested directory structure
dataset_path = 'dataset'
model_path = 'model/model.pth'

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load dataset
try:
    # Check if we need to use nested paths
    aiart_path = os.path.join(dataset_path, 'aiart', 'AiArtData')
    realart_path = os.path.join(dataset_path, 'realart', 'RealArt', 'RealArt')  # Handle double nesting
    
    if os.path.exists(aiart_path) and os.path.exists(realart_path):
        # Create a temporary dataset structure
        import shutil
        import tempfile
        
        temp_dataset = tempfile.mkdtemp()
        temp_aiart = os.path.join(temp_dataset, 'aiart')
        temp_realart = os.path.join(temp_dataset, 'realart')
        
        # Copy files to temporary structure
        shutil.copytree(aiart_path, temp_aiart)
        shutil.copytree(realart_path, temp_realart)
        
        dataset_path = temp_dataset
        print("Using nested dataset structure")
    else:
        print("Using standard dataset structure")
    
    dataset = ImageFolder(root=dataset_path, transform=transform)
    print(f"Dataset loaded successfully. Found {len(dataset)} images in {len(dataset.classes)} classes.")
    print(f"Classes: {dataset.classes}")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Simple CNN Model - Match the existing model architecture
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
                nn.Linear(64, 2)  # Output: 2 classes (Fake/Real)
            )

        def forward(self, x):
            return self.model(x)

    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training started...")
    for epoch in range(5):  # Try with 5 epochs first
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 0:  # Print every 10 batches
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/5 - Average Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 64, 64).to(device)
        test_output = model(test_input)
        print(f"Test output shape: {test_output.shape}")
        print("Model training completed successfully!")

    # Clean up temporary files if they were created
    if 'temp_dataset' in locals():
        shutil.rmtree(temp_dataset)
        print("Temporary dataset cleaned up")

except Exception as e:
    print(f"Error during training: {e}")
    print("Please check your dataset structure and try again.")
