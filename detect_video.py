#!/usr/bin/env python3
"""
Command-line video deepfake detection script
"""

import os
import sys
import argparse
from video_processor import VideoProcessor, create_video_transform
from model_loader import load_model

def main():
    parser = argparse.ArgumentParser(description='Detect deepfakes in video files')
    parser.add_argument('video_path', help='Path to video file to analyze')
    parser.add_argument('--frames', type=int, default=50, help='Maximum frames to analyze (default: 50)')
    parser.add_argument('--interval', type=int, default=10, help='Frame extraction interval (default: 10)')
    parser.add_argument('--output', help='Output file for detailed results (optional)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        return 1
    
    try:
        print("Loading model...")
        model = load_model()
        transform = create_video_transform()
        
        print("Initializing video processor...")
        processor = VideoProcessor(model, transform, frame_interval=args.interval)
        
        print(f"Analyzing video: {args.video_path}")
        print(f"Settings: Max frames={args.frames}, Interval={args.interval}")
        print("-" * 60)
        
        # Process video
        results = processor.process_video(args.video_path)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return 1
        
        # Display results
        display_results(results)
        
        # Save detailed results if requested
        if args.output:
            save_results(results, args.output)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

def display_results(results):
    """Display analysis results in a formatted way"""
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION RESULTS")
    print("=" * 60)
    
    # Overall prediction
    icon = "🚫" if results['is_fake'] else "✅"
    status = "FAKE" if results['is_fake'] else "REAL"
    print(f"{icon} Overall Prediction: {results['overall_prediction']}")
    print(f"   Confidence: {results['overall_confidence']}")
    print()
    
    # Video metadata
    if 'video_metadata' in results:
        meta = results['video_metadata']
        print("📹 Video Information:")
        print(f"   Duration: {meta['duration_formatted']} ({meta['duration_seconds']}s)")
        print(f"   FPS: {meta['fps']}")
        print(f"   Total Frames: {meta['total_frames']}")
        print()
    
    # Frame analysis summary
    print("🔍 Frame Analysis Summary:")
    print(f"   Frames Analyzed: {results['frame_count']}")
    print(f"   Fake Frames: {results['fake_frames']} ({results['fake_percentage']})")
    print(f"   Real Frames: {results['real_frames']} ({results['real_percentage']})")
    print()
    
    # Analysis method
    print(f"📊 {results['analysis_method']}")
    print()
    
    # Frame-by-frame details (show first 10)
    if results['frame_analysis']:
        print("🎬 Frame-by-Frame Analysis (showing first 10):")
        print("-" * 40)
        for i, frame in enumerate(results['frame_analysis'][:10]):
            icon = "🚫" if frame['is_fake'] else "✅"
            print(f"   Frame {frame['frame']:4d}ms: {icon} {frame['prediction']} ({frame['confidence']})")
        
        if len(results['frame_analysis']) > 10:
            print(f"   ... and {len(results['frame_analysis']) - 10} more frames")
        print()
    
    # Final verdict
    print("=" * 60)
    if results['is_fake']:
        print("🚫 VERDICT: This video appears to contain AI-GENERATED (FAKE) content")
        print("   The analysis detected manipulation in multiple frames")
    else:
        print("✅ VERDICT: This video appears to be REAL")
        print("   No significant signs of AI manipulation detected")
    print("=" * 60)

def save_results(results, output_path):
    """Save detailed results to a file"""
    try:
        with open(output_path, 'w') as f:
            f.write("DeepFake Video Detection Results\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Overall Prediction: {results['overall_prediction']}\n")
            f.write(f"Confidence: {results['overall_confidence']}\n")
            f.write(f"Is Fake: {results['is_fake']}\n\n")
            
            if 'video_metadata' in results:
                meta = results['video_metadata']
                f.write("Video Metadata:\n")
                f.write(f"  Duration: {meta['duration_formatted']}\n")
                f.write(f"  FPS: {meta['fps']}\n")
                f.write(f"  Total Frames: {meta['total_frames']}\n\n")
            
            f.write(f"Frame Analysis Summary:\n")
            f.write(f"  Total Frames Analyzed: {results['frame_count']}\n")
            f.write(f"  Fake Frames: {results['fake_frames']} ({results['fake_percentage']})\n")
            f.write(f"  Real Frames: {results['real_frames']} ({results['real_percentage']})\n\n")
            
            f.write(f"Analysis Method: {results['analysis_method']}\n\n")
            
            if results['frame_analysis']:
                f.write("Frame-by-Frame Analysis:\n")
                f.write("-" * 30 + "\n")
                for frame in results['frame_analysis']:
                    f.write(f"Frame {frame['frame']:4d}ms: {frame['prediction']} ({frame['confidence']})\n")
        
        print(f"Detailed results saved to: {output_path}")
        
    except Exception as e:
        print(f"Warning: Could not save results to {output_path}: {e}")

if __name__ == "__main__":
    sys.exit(main())
