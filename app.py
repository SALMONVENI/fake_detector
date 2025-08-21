from flask import Flask, render_template, request, jsonify, flash
from PIL import Image
import torch
import torchvision.transforms as transforms
from model_loader import load_model
from video_processor import VideoProcessor, create_video_transform
import os
import io
import tempfile
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')  # Use environment variable in production

# Global variables
model = None
transform = None
video_processor = None

def initialize_model():
    global model, transform, video_processor
    try:
        model = load_model()
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Match the training size
            transforms.ToTensor(),
        ])
        video_processor = VideoProcessor(model, transform)
        print("Model and video processor loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        video_processor = None

# Initialize model when app starts
initialize_model()

class_names = ['AI-Generated (Fake)', 'Real']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        file_ext = file.filename.lower().rsplit('.', 1)[1] if '.' in file.filename else ''
        
        # Image processing
        if file_ext in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}:
            return process_image(file)
        
        # Video processing
        elif file_ext in {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}:
            return process_video(file)
        
        else:
            return jsonify({'error': 'Unsupported file type. Please upload an image or video.'}), 400
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

def process_image(file):
    """Process image file for deepfake detection"""
    try:
        # Check file size (max 10MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:
            return jsonify({'error': 'File size too large. Maximum size is 10MB.'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check the backend.'}), 500
        
        # Process image
        image = Image.open(file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            confidence = probabilities[0][predicted.item()].item()
            label = class_names[predicted.item()]
            
        return jsonify({
            'type': 'image',
            'prediction': label,
            'confidence': f"{confidence:.2%}",
            'is_fake': predicted.item() == 0
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': f'Image processing error: {str(e)}'}), 500

def process_video(file):
    """Process video file for deepfake detection"""
    try:
        # Check file size (max 100MB for videos)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 100 * 1024 * 1024:
            return jsonify({'error': 'Video file size too large. Maximum size is 100MB.'}), 400
        
        if video_processor is None:
            return jsonify({'error': 'Video processor not loaded. Please check the backend.'}), 500
        
        # Save video to temporary file
        temp_dir = tempfile.mkdtemp()
        video_filename = f"video_{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1]}"
        video_path = os.path.join(temp_dir, video_filename)
        
        file.save(video_path)
        
        try:
            # Process video
            results = video_processor.process_video(video_path)
            
            if 'error' in results:
                return jsonify({'error': results['error']}), 500
            
            # Add file type indicator
            results['type'] = 'video'
            
            return jsonify(results)
            
        finally:
            # Clean up temporary files
            try:
                os.remove(video_path)
                os.rmdir(temp_dir)
            except:
                pass
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return jsonify({'error': f'Video processing error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'video_processor_loaded': video_processor is not None
    })

if __name__ == '__main__':
    if model is None:
        print("Warning: Model could not be loaded. Please check if model.pth exists.")
    if video_processor is None:
        print("Warning: Video processor could not be initialized.")
    
    # Use environment variables for production deployment
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug, host='0.0.0.0', port=port)
