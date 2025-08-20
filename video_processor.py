import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import List, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, model, transform, frame_interval: int = 10):
        """
        Initialize video processor
        
        Args:
            model: Pre-trained deepfake detection model
            transform: Image transformation pipeline
            frame_interval: Extract every Nth frame for analysis
        """
        self.model = model
        self.transform = transform
        self.frame_interval = frame_interval
        
    def extract_frames(self, video_path: str, max_frames: int = 50) -> List[np.ndarray]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frames as numpy arrays
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {total_frames} total frames")
            
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Extract every Nth frame
                if frame_count % self.frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    
                frame_count += 1
                
                # Progress indicator
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames for analysis")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def analyze_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze extracted frames for deepfake detection
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            Dictionary containing analysis results
        """
        if not frames:
            return {"error": "No frames to analyze"}
        
        predictions = []
        confidences = []
        
        self.model.eval()
        
        with torch.no_grad():
            for i, frame in enumerate(frames):
                try:
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(frame)
                    
                    # Apply transformations
                    image_tensor = self.transform(pil_image).unsqueeze(0)
                    
                    # Get prediction
                    outputs = self.model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    confidence = probabilities[0][predicted.item()].item()
                    prediction = predicted.item()  # 0 for fake, 1 for real
                    
                    predictions.append(prediction)
                    confidences.append(confidence)
                    
                except Exception as e:
                    logger.warning(f"Error processing frame {i}: {e}")
                    continue
        
        if not predictions:
            return {"error": "Failed to process any frames"}
        
        # Calculate overall results
        fake_count = sum(1 for p in predictions if p == 0)
        real_count = sum(1 for p in predictions if p == 1)
        total_frames = len(predictions)
        
        # Overall prediction based on majority
        overall_prediction = 0 if fake_count > real_count else 1
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences)
        
        # Calculate frame-by-frame analysis
        frame_analysis = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            frame_analysis.append({
                "frame": i * self.frame_interval,
                "prediction": "AI-Generated (Fake)" if pred == 0 else "Real",
                "confidence": f"{conf:.2%}",
                "is_fake": pred == 0
            })
        
        return {
            "overall_prediction": "AI-Generated (Fake)" if overall_prediction == 0 else "Real",
            "overall_confidence": f"{avg_confidence:.2%}",
            "is_fake": overall_prediction == 0,
            "frame_count": total_frames,
            "fake_frames": fake_count,
            "real_frames": real_count,
            "fake_percentage": f"{(fake_count/total_frames)*100:.1f}%",
            "real_percentage": f"{(real_count/total_frames)*100:.1f}%",
            "frame_analysis": frame_analysis,
            "analysis_method": f"Analyzed {total_frames} frames (every {self.frame_interval}th frame)"
        }
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process video file for deepfake detection
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video analysis results
        """
        try:
            logger.info(f"Starting video analysis: {video_path}")
            
            # Extract frames
            frames = self.extract_frames(video_path)
            
            # Analyze frames
            results = self.analyze_frames(frames)
            
            # Add video metadata
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            cap.release()
            
            results["video_metadata"] = {
                "fps": round(fps, 2),
                "duration_seconds": round(duration, 2),
                "duration_formatted": f"{int(duration//60)}:{int(duration%60):02d}",
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            }
            
            logger.info(f"Video analysis completed: {results['overall_prediction']}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {"error": f"Video processing failed: {str(e)}"}

def create_video_transform():
    """Create transformation pipeline for video frames"""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
