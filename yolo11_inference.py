#!/usr/bin/env python3
"""
YOLO11 Inference and Validation Script

Run inference and validation on YOLO11 models.

Usage:
    # Validation
    $ python yolo11_inference.py --mode val --weights runs/yolo11_train/infrared_basic/weights/best.pt --data data_infrared.yaml
    
    # Inference on images
    $ python yolo11_inference.py --mode predict --weights runs/yolo11_train/infrared_basic/weights/best.pt --source path/to/images
    
    # Inference on video
    $ python yolo11_inference.py --mode predict --weights runs/yolo11_train/infrared_basic/weights/best.pt --source path/to/video.mp4
"""

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLO11 Inference and Validation')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['predict', 'val', 'export'], 
                       default='predict', help='inference mode')
    
    # Model and data
    parser.add_argument('--weights', type=str, required=True, help='model weights path')
    parser.add_argument('--data', type=str, help='dataset.yaml path (required for validation)')
    parser.add_argument('--source', type=str, help='source for prediction (image/video/directory)')
    
    # Inference parameters
    parser.add_argument('--imgsz', '--img', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    # Output options
    parser.add_argument('--save', action='store_true', default=True, help='save results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/predict', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    # Validation specific
    parser.add_argument('--split', type=str, default='val', help='dataset split to validate on')
    parser.add_argument('--save-json', action='store_true', help='save results to JSON file')
    
    # Export specific
    parser.add_argument('--format', type=str, default='onnx', 
                       choices=['onnx', 'torchscript', 'tensorflow', 'tflite', 'edgetpu', 'tfjs', 'openvino'],
                       help='export format')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision export')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model')
    
    return parser.parse_args()

def validate_model(model, args, logger):
    """Run model validation."""
    logger.info("Running validation...")
    
    if not args.data:
        logger.error("--data argument is required for validation mode")
        return
    
    val_args = {
        'data': args.data,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'iou': args.iou,
        'max_det': args.max_det,
        'device': args.device,
        'split': args.split,
        'save_json': args.save_json,
        'save_txt': args.save_txt,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
    }
    
    try:
        results = model.val(**val_args)
        
        logger.info("Validation completed successfully!")
        logger.info(f"Results: {results}")
        
        # Print key metrics
        if hasattr(results, 'results_dict'):
            logger.info("Validation metrics:")
            for key, value in results.results_dict.items():
                logger.info(f"  {key}: {value}")
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return None

def run_prediction(model, args, logger):
    """Run model prediction."""
    logger.info("Running prediction...")
    
    if not args.source:
        logger.error("--source argument is required for prediction mode")
        return
    
    predict_args = {
        'source': args.source,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'iou': args.iou,
        'max_det': args.max_det,
        'device': args.device,
        'save': args.save and not args.nosave,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_crop': args.save_crop,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
    }
    
    try:
        results = model.predict(**predict_args)
        
        logger.info("Prediction completed successfully!")
        logger.info(f"Results saved to: {Path(args.project) / args.name}")
        
        # Print detection summary
        total_detections = 0
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                total_detections += len(result.boxes)
        
        logger.info(f"Total detections: {total_detections}")
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

def export_model(model, args, logger):
    """Export model to different formats."""
    logger.info(f"Exporting model to {args.format} format...")
    
    export_args = {
        'format': args.format,
        'imgsz': args.imgsz,
        'half': args.half,
        'dynamic': args.dynamic,
        'simplify': args.simplify,
    }
    
    try:
        export_path = model.export(**export_args)
        
        logger.info("Export completed successfully!")
        logger.info(f"Exported model saved to: {export_path}")
        
        return export_path
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return None

def main():
    """Main function."""
    args = parse_args()
    logger = setup_logging()
    
    # Print configuration
    logger.info("YOLO11 Inference Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Load model
    logger.info(f"Loading model: {args.weights}")
    try:
        model = YOLO(args.weights)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Run the specified mode
    if args.mode == 'predict':
        results = run_prediction(model, args, logger)
    elif args.mode == 'val':
        results = validate_model(model, args, logger)
    elif args.mode == 'export':
        results = export_model(model, args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return
    
    if results is not None:
        logger.info("Operation completed successfully!")
    else:
        logger.error("Operation failed!")

if __name__ == '__main__':
    main() 
