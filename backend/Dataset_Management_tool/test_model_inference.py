"""
Test/Inference Script for Trained YOLOv8 Model
Tests a trained model on images or videos and saves results with detections
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import argparse
import numpy as np


def test_model_on_image(model_path: str, image_path: str, output_dir: str = "inference_results", conf_threshold: float = 0.25):
    """
    Test trained YOLOv8 model on a single image
    
    Args:
        model_path: Path to trained .pt model file
        image_path: Path to image file to test on
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
    """
    
    # Validate inputs
    if not os.path.exists(model_path):
        print(f" Error: Model file not found: {model_path}")
        return False
    
    if not os.path.exists(image_path):
        print(f" Error: Image file not found: {image_path}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print(f" Loading model from: {model_path}")
        model = YOLO(model_path)
        
        print(f"  Running inference on: {image_path}")
        print(f" Using confidence threshold: {conf_threshold}")
        
        # Run inference with lower confidence threshold
        results = model(image_path, conf=conf_threshold, verbose=True)
        
        # Extract results
        result = results[0]
        
        print("\n" + "="*60)
        print(" DETECTION RESULTS")
        print("="*60)
        
        if len(result.boxes) == 0:
            print("  No objects detected in the image")
            print(" Try lowering the confidence threshold (--conf 0.1)")
            detections_found = False
        else:
            print(f" Found {len(result.boxes)} detection(s):\n")
            detections_found = True
            
            for idx, box in enumerate(result.boxes, 1):
                # Get confidence
                conf = box.conf[0].item()
                
                # Get class name
                class_id = int(box.cls[0].item())
                class_name = model.names.get(class_id, f"Class {class_id}")
                
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                width = (x2 - x1).item()
                height = (y2 - y1).item()
                
                print(f"  Detection {idx}:")
                print(f"    - Class: {class_name}")
                print(f"    - Confidence: {conf:.2%}")
                print(f"    - Bounding Box: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
                print(f"    - Size: {width:.0f}x{height:.0f} pixels\n")
        
        # Save annotated image
        annotated_image = result.plot()
        output_path = os.path.join(output_dir, "result.jpg")
        cv2.imwrite(output_path, annotated_image)
        print(f" Annotated image saved to: {output_path}")
        
        # Save detailed results as text
        txt_output = os.path.join(output_dir, "results.txt")
        with open(txt_output, 'w') as f:
            f.write("="*60 + "\n")
            f.write("YOLOv8 INFERENCE RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Confidence Threshold: {conf_threshold}\n")
            f.write(f"Detections: {len(result.boxes)}\n\n")
            
            if detections_found:
                for idx, box in enumerate(result.boxes, 1):
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = model.names.get(class_id, f"Class {class_id}")
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    f.write(f"Detection {idx}: {class_name} ({conf:.2%})\n")
                    f.write(f"  Coordinates: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})\n\n")
        
        print(f" Results summary saved to: {txt_output}")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f" Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_on_video(model_path: str, video_path: str, output_dir: str = "inference_results", conf_threshold: float = 0.25, max_frames: int = 100):
    """
    Test trained YOLOv8 model on a video file
    
    Args:
        model_path: Path to trained .pt model file
        video_path: Path to video file to test on
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        max_frames: Maximum number of frames to process
    """
    
    # Validate inputs
    if not os.path.exists(model_path):
        print(f" Error: Model file not found: {model_path}")
        return False
    
    if not os.path.exists(video_path):
        print(f" Error: Video file not found: {video_path}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print(f" Loading model from: {model_path}")
        model = YOLO(model_path)
        
        print(f" Running inference on video: {video_path}")
        print(f" Using confidence threshold: {conf_threshold}")
        print(f" Processing up to {max_frames} frames")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(" Error: Could not open video file")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f" Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Prepare output video
        output_video_path = os.path.join(output_dir, "result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_num = 0
        total_detections = 0
        
        while frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Run inference on frame
            results = model(frame, conf=conf_threshold, verbose=False)
            result = results[0]
            
            # Count detections in this frame
            frame_detections = len(result.boxes)
            total_detections += frame_detections
            
            # Draw detections on frame
            annotated_frame = result.plot()
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_num}/{min(max_frames, frame_count)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Detections: {frame_detections}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Progress update
            if frame_num % 10 == 0:
                print(f" Processed {frame_num}/{min(max_frames, frame_count)} frames, Total detections: {total_detections}")
        
        cap.release()
        out.release()
        
        print("\n" + "="*60)
        print(" VIDEO DETECTION RESULTS")
        print("="*60)
        print(f" Processed {frame_num} frames")
        print(f" Total detections across all frames: {total_detections}")
        print(f" Average detections per frame: {total_detections/frame_num:.2f}")
        
        if total_detections == 0:
            print("  No objects detected in the video")
            print(" Try lowering the confidence threshold (--conf 0.1)")
        else:
            print(" Detections found! Check the output video.")
        
        print(f" Annotated video saved to: {output_video_path}")
        
        # Save summary
        txt_output = os.path.join(output_dir, "video_results.txt")
        with open(txt_output, 'w') as f:
            f.write("="*60 + "\n")
            f.write("YOLOv8 VIDEO INFERENCE RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Confidence Threshold: {conf_threshold}\n")
            f.write(f"Frames Processed: {frame_num}\n")
            f.write(f"Total Detections: {total_detections}\n")
            f.write(f"Average per Frame: {total_detections/frame_num:.2f}\n")
        
        print(f"📄 Results summary saved to: {txt_output}")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f" Error during video inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_model(model_path: str):
    """
    Diagnose potential issues with the trained model
    """
    print("\n" + "="*60)
    print(" MODEL DIAGNOSIS")
    print("="*60)
    
    try:
        model = YOLO(model_path)
        print(f" Model loaded successfully")
        print(f" Model type: {type(model)}")
        
        # Check model info
        if hasattr(model, 'names'):
            print(f"  Classes: {model.names}")
            print(f" Number of classes: {len(model.names)}")
        
        # Check if model has been trained (look for best.pt vs yolov8n.pt)
        if 'best.pt' in str(model_path):
            print(" This appears to be a trained model (best.pt)")
        elif 'last.pt' in str(model_path):
            print("  This is a checkpoint model (last.pt), try using best.pt instead")
        else:
            print("ℹ  This appears to be a pretrained model")
            
        return True
        
    except Exception as e:
        print(f" Error loading model: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test trained YOLOv8 model on images or videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on image
  python test_model_inference.py --model runs/detect/train/weights/best.pt --image test.jpg
  
  # Test on video with lower confidence
  python test_model_inference.py --model best.pt --video drone_video.mp4 --conf 0.1
  
  # Diagnose model issues
  python test_model_inference.py --model best.pt --diagnose
  
  # Test with custom output directory
  python test_model_inference.py --model yolov8n.pt --image drone.jpg --output my_results/
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained .pt model file"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to test"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to test"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Output directory for results (default: inference_results)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25, try 0.1 if no detections)"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to process for video (default: 100)"
    )
    
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Only diagnose the model, don't run inference"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.video and not args.diagnose:
        print(" Error: Must specify --image, --video, or --diagnose")
        sys.exit(1)
    
    if args.image and args.video:
        print(" Error: Cannot specify both --image and --video")
        sys.exit(1)
    
    # Run diagnosis first
    if not diagnose_model(args.model):
        sys.exit(1)
    
    if args.diagnose:
        sys.exit(0)
    
    # Run inference
    success = False
    if args.image:
        success = test_model_on_image(args.model, args.image, args.output, args.conf)
    elif args.video:
        success = test_model_on_video(args.model, args.video, args.output, args.conf, args.max_frames)
    
    sys.exit(0 if success else 1)
