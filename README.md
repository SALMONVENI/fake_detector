# DeepFake Image & Video Detector

A web application that uses deep learning to detect AI-generated (fake) images and videos from real ones.

## Features

- **Image Detection**: Upload and analyze images for deepfake detection
- **Video Detection**: Upload and analyze videos with frame-by-frame analysis
- **Real-time Prediction**: Get instant results with confidence scores
- **Modern Web Interface**: Beautiful, responsive design
- **RESTful API**: Backend endpoints for both image and video processing
- **Frame Analysis**: Detailed breakdown of video frame predictions
- **Multiple Formats**: Support for various image and video formats

## Supported Formats

### Images
- PNG, JPG, JPEG, GIF, BMP
- Maximum size: 10MB

### Videos
- MP4, AVI, MOV, MKV, WMV, FLV, WEBM
- Maximum size: 100MB
- Frame extraction and analysis

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The video processing libraries (OpenCV, MoviePy) may require additional system dependencies on some platforms.

### 2. Dataset Structure

Ensure your dataset follows this structure:
```
dataset/
├── aiart/
│   └── (AI-generated images)
└── realart/
    └── (Real images)
```

### 3. Train the Model

```bash
python train_model.py
```

This will create a `model/model.pth` file.

### 4. Run the Application

```bash
python app.py
```

Or use the smart startup script:
```bash
python run.py
```

The application will be available at `http://localhost:5000`

## Usage

### Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Upload an image or video file
3. Click "Analyze" to get the prediction
4. View detailed results:
   - **Images**: Single prediction with confidence
   - **Videos**: Overall prediction + frame-by-frame analysis

### Command Line

#### Image Detection
```bash
python detect_image.py [image_path]
```

#### Video Detection
```bash
python detect_video.py [video_path] [options]
```

**Options:**
- `--frames N`: Maximum frames to analyze (default: 50)
- `--interval N`: Frame extraction interval (default: 10)
- `--output file.txt`: Save detailed results to file

**Example:**
```bash
python detect_video.py sample_video.mp4 --frames 100 --interval 5 --output results.txt
```

## API Endpoints

- `GET /` - Main page
- `POST /predict` - File analysis endpoint (supports both images and videos)
- `GET /health` - Health check endpoint

## How Video Detection Works

1. **Frame Extraction**: Extracts frames at regular intervals (every Nth frame)
2. **Frame Analysis**: Each frame is processed through the deepfake detection model
3. **Aggregation**: Results are combined to provide overall video prediction
4. **Detailed Results**: Shows frame-by-frame analysis with timestamps

## File Structure

- `app.py` - Flask backend server with image/video support
- `model_loader.py` - Neural network model loader
- `video_processor.py` - Video processing and frame analysis
- `train_model.py` - Model training script
- `detect_image.py` - Command-line image detection
- `detect_video.py` - Command-line video detection
- `templates/` - HTML templates
- `static/` - Static files
- `model/` - Trained model files
- `dataset/` - Training dataset

## Requirements

- Python 3.7+
- PyTorch
- Flask
- Pillow (PIL)
- NumPy
- OpenCV (for video processing)
- MoviePy (for video handling)

## Performance Notes

- **Image Processing**: Near-instant results
- **Video Processing**: Depends on video length and frame count
- **Memory Usage**: Videos are processed in chunks to manage memory
- **Frame Limit**: Default maximum 50 frames per video (configurable)

## Troubleshooting

- If the model fails to load, ensure `model/model.pth` exists
- Check that your dataset structure matches the expected format
- Ensure all dependencies are installed correctly
- For video processing issues, verify OpenCV and MoviePy installation
- Large videos may take longer to process - check the loading indicator

## Advanced Configuration

### Video Processing Settings

You can modify video processing parameters in `video_processor.py`:
- `frame_interval`: How often to extract frames (default: every 10th frame)
- `max_frames`: Maximum number of frames to analyze (default: 50)

### Model Architecture

The current model is optimized for 64x64 pixel inputs. If you need different dimensions, update both `model_loader.py` and `train_model.py`.
