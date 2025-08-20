import os
import torch
import torch.nn as nn

def load_model():
    # Use the same architecture as the existing trained model
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
    
    model = CNN()

    # Use relative path instead of hardcoded absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', 'model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
