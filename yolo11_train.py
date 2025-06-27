#!/usr/bin/env python3
"""
YOLO11 Training Script

Train a YOLO11 model on a custom dataset with comprehensive logging and configuration options.
Similar to YOLOv5 training but using the new Ultralytics YOLO11 architecture.

Usage:
    $ python yolo11_train.py --data data.yaml --weights yolo11n.pt --img 640 --epochs 100
    $ python yolo11_train.py --data data.yaml --weights yolo11n.pt --img 640 --epochs 100 --logfile training.log
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# Setup logging function
def setup_logging(log_file=None, verbose=True):
    """Setup logging configuration with optional file output."""
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in (-1, 0) else logging.WARNING
    
    # Create logger
    logger = logging.getLogger('YOLO11_TRAIN')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file and rank in (-1, 0):
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        logger.info(f'Logging to file: {log_file}')
    
    return logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLO11 model')
    
    # Model and data arguments
    parser.add_argument('--weights', type=str, default='yolo11n.pt', help='initial weights path')
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path (optional)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train image size (pixels)')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    
    # Optimizer and learning rate
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='auto', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate fraction')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='optimizer weight decay')
    
    # Training options
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50 percent')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    
    # Augmentation
    parser.add_argument('--hsv-h', type=float, default=0.015, help='image HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv-s', type=float, default=0.7, help='image HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv-v', type=float, default=0.4, help='image HSV-Value augmentation (fraction)')
    parser.add_argument('--degrees', type=float, default=0.0, help='image rotation (plus/minus deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='image translation (plus/minus fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='image scale (plus/minus gain)')
    parser.add_argument('--shear', type=float, default=0.0, help='image shear (plus/minus deg)')
    parser.add_argument('--perspective', type=float, default=0.0, help='image perspective (plus/minus fraction)')
    parser.add_argument('--flipud', type=float, default=0.0, help='image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5, help='image flip left-right (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.0, help='image mixup (probability)')
    
    # Validation and saving
    parser.add_argument('--val', action='store_true', default=True, help='validate/test during training')
    parser.add_argument('--save-period', type=int, default=-1, help='save checkpoint every x epochs')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='cache images in "ram" (default) or "disk"')
    
    # Project organization
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    # Logging
    parser.add_argument('--logfile', type=str, default='', help='path to log file (logs to console only if empty)')
    parser.add_argument('--verbose', action='store_true', default=True, help='verbose logging')
    
    # Advanced options
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to enable deterministic mode')
    parser.add_argument('--amp', action='store_true', default=True, help='Automatic Mixed Precision training')
    parser.add_argument('--fraction', type=float, default=1.0, help='dataset fraction to train on')
    parser.add_argument('--profile', action='store_true', help='profile ONNX and TensorRT speeds during training')
    
    return parser.parse_args()

def load_hyperparameters(hyp_path):
    """Load hyperparameters from YAML file."""
    if hyp_path and Path(hyp_path).exists():
        with open(hyp_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    log_file = None
    if args.logfile:
        if not os.path.isabs(args.logfile):
            log_file = Path(args.project) / args.name / args.logfile
        else:
            log_file = Path(args.logfile)
    
    logger = setup_logging(log_file, args.verbose)
    
    # Print arguments
    logger.info("YOLO11 Training Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Load model
    logger.info(f"Loading model: {args.weights}")
    try:
        model = YOLO(args.weights)
        logger.info(f"Model loaded successfully: {model.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Load hyperparameters if provided
    hyp = load_hyperparameters(args.hyp)
    if hyp:
        logger.info(f"Loaded hyperparameters from: {args.hyp}")
        for key, value in hyp.items():
            logger.info(f"  {key}: {value}")
    
    # Prepare training arguments
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'pretrained': True,
        'optimizer': args.optimizer,
        'seed': args.seed,
        'deterministic': args.deterministic,
        'single_cls': args.single_cls,
        'rect': args.rect,
        'cos_lr': args.cos_lr,
        'patience': args.patience,
        'resume': args.resume,
        'amp': args.amp,
        'fraction': args.fraction,
        'profile': args.profile,
        'cache': args.cache,
        'save_period': args.save_period,
        'val': args.val,
        'plots': True,
        'verbose': args.verbose,
        # Learning rate parameters
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        # Augmentation parameters
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'perspective': args.perspective,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
    }
    
    # Add hyperparameters if loaded
    if hyp:
        train_args.update(hyp)
    
    # Start training
    logger.info("Starting training...")
    start_time = time.time()
    
    try:
        results = model.train(**train_args)
        
        # Training completed
        end_time = time.time()
        training_time = end_time - start_time
        hours = training_time // 3600
        minutes = (training_time % 3600) // 60
        seconds = training_time % 60
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        logger.info(f"Results: {results}")
        
        # Log final metrics if available
        if hasattr(results, 'results_dict'):
            logger.info("Final metrics:")
            for key, value in results.results_dict.items():
                logger.info(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("Training script completed")

if __name__ == '__main__':
    main() 
