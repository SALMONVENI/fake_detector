#!/usr/bin/env python3
"""
Startup script for DeepFake Image Detector
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import torch
        import torchvision
        import PIL
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def check_model():
    """Check if the trained model exists"""
    model_path = os.path.join('model', 'model.pth')
    if os.path.exists(model_path):
        print(f"✓ Model found at {model_path}")
        return True
    else:
        print(f"✗ Model not found at {model_path}")
        print("  Please run 'python train_model.py' first to train the model")
        return False

def main():
    print("DeepFake Image Detector - Startup Check")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstalling dependencies...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
            print("✓ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("✗ Failed to install dependencies")
            return False
    
    # Check model
    if not check_model():
        print("\nTraining model...")
        try:
            subprocess.run([sys.executable, 'train_model.py'], check=True)
            print("✓ Model trained successfully")
        except subprocess.CalledProcessError:
            print("✗ Failed to train model")
            return False
    
    print("\nStarting Flask application...")
    print("The application will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
