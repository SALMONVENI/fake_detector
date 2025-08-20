# detect_image.py

from PIL import Image
import torchvision.transforms as transforms
import torch
from model_loader import load_model
import os
import sys

def main():
    try:
        # Load model
        print("Loading model...")
        model = load_model()
        print("Model loaded successfully!")
        
        # Setup transforms
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        # Get image path from user
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = input("Enter image path to test: ").strip()
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found.")
            return
        
        # Check if it's an image file
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in allowed_extensions:
            print(f"Error: File '{image_path}' is not a supported image format.")
            print(f"Supported formats: {', '.join(allowed_extensions)}")
            return
        
        # Load and process image
        print(f"Processing image: {image_path}")
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            confidence = probabilities[0][predicted.item()].item()
            label = "AI-Generated (Fake)" if predicted.item() == 0 else "Real"
        
        # Display results
        print("\n" + "="*50)
        print("DEEPFAKE DETECTION RESULTS")
        print("="*50)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        print("="*50)
        
        if predicted.item() == 0:
            print("🚫 This image appears to be AI-generated (FAKE)")
        else:
            print("✅ This image appears to be REAL")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model file exists by running 'python train_model.py' first.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your input and try again.")

if __name__ == "__main__":
    main()
