#!/usr/bin/env python3
"""
Test script for DeepFake Image & Video Detector
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print("✓ TorchVision imported successfully")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False
    
    try:
        import PIL
        print("✓ PIL imported successfully")
    except ImportError as e:
        print(f"✗ PIL import failed: {e}")
        return False
    
    # Test video processing libraries
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        print("  Video processing will not be available")
        return False
    
    try:
        import moviepy
        print("✓ MoviePy imported successfully")
    except ImportError as e:
        print(f"✗ MoviePy import failed: {e}")
        print("  Some video features may not work")
        return False
    
    return True

def test_model_loader():
    """Test if the model loader works"""
    print("\nTesting model loader...")
    
    try:
        from model_loader import load_model
        print("✓ Model loader imported successfully")
        
        # Check if model file exists
        model_path = os.path.join('model', 'model.pth')
        if os.path.exists(model_path):
            print(f"✓ Model file found at {model_path}")
            return True
        else:
            print(f"✗ Model file not found at {model_path}")
            print("  This is expected if you haven't trained the model yet")
            return False
            
    except Exception as e:
        print(f"✗ Model loader test failed: {e}")
        return False

def test_video_processor():
    """Test if the video processor can be imported"""
    print("\nTesting video processor...")
    
    try:
        from video_processor import VideoProcessor, create_video_transform
        print("✓ Video processor imported successfully")
        return True
    except Exception as e:
        print(f"✗ Video processor test failed: {e}")
        return False

def test_flask_app():
    """Test if the Flask app can be created"""
    print("\nTesting Flask app...")
    
    try:
        from app import app
        print("✓ Flask app created successfully")
        return True
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        return False

def test_dataset_structure():
    """Test if the dataset structure is correct"""
    print("\nTesting dataset structure...")
    
    aiart_path = os.path.join('dataset', 'aiart', 'AiArtData')
    realart_path = os.path.join('dataset', 'realart', 'RealArt', 'RealArt')  # Handle double nesting
    
    if os.path.exists(aiart_path):
        aiart_count = len([f for f in os.listdir(aiart_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
        print(f"✓ AI art dataset found with {aiart_count} images")
    else:
        print("✗ AI art dataset not found")
        return False
    
    if os.path.exists(realart_path):
        realart_count = len([f for f in os.listdir(realart_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
        print(f"✓ Real art dataset found with {realart_count} images")
    else:
        print("✗ Real art dataset not found")
        return False
    
    return True

def main():
    print("DeepFake Image & Video Detector - System Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_loader,
        test_video_processor,
        test_flask_app,
        test_dataset_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The system is ready to run.")
        print("\nTo start the application, run:")
        print("  python run.py")
        print("  or")
        print("  python app.py")
        print("\nFor command-line usage:")
        print("  python detect_image.py [image_path]")
        print("  python detect_video.py [video_path]")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        
        if passed < 3:  # If basic imports failed
            print("\nTo install dependencies, run:")
            print("  pip install -r requirements.txt")
        
        if not os.path.exists(os.path.join('model', 'model.pth')):
            print("\nTo train the model, run:")
            print("  python train_model.py")

if __name__ == "__main__":
    main()
