#!/usr/bin/env python3
"""
YOLO11 Training Examples

This script demonstrates different ways to train YOLO11 models.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print its description."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print("✅ Command completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    return True

def main():
    """Run different training examples."""
    
    print("YOLO11 Training Examples")
    print("Make sure you have activated your virtual environment!")
    print("Usage: python yolo11_train_example.py")
    
    # Example 1: Basic training with infrared dataset
    print("\n📋 Example 1: Basic YOLO11n training")
    cmd1 = [
        sys.executable, "yolo11_train.py",
        "--data", "data_infrared.yaml",
        "--weights", "yolo11n.pt",
        "--epochs", "10",
        "--batch-size", "8",
        "--imgsz", "640",
        "--project", "runs/yolo11_train",
        "--name", "infrared_basic",
        "--logfile", "training_basic.log"
    ]
    
    if not run_command(cmd1, "Basic YOLO11n training"):
        print("⚠️  Basic training failed. Check your dataset configuration.")
    
    # Example 2: Training with custom hyperparameters
    print("\n📋 Example 2: YOLO11s with custom settings")
    cmd2 = [
        sys.executable, "yolo11_train.py",
        "--data", "data_infrared.yaml",
        "--weights", "yolo11s.pt",
        "--epochs", "20",
        "--batch-size", "16",
        "--imgsz", "640",
        "--lr0", "0.001",
        "--optimizer", "Adam",
        "--cos-lr",
        "--patience", "50",
        "--project", "runs/yolo11_train",
        "--name", "infrared_custom",
        "--logfile", "training_custom.log",
        "--cache", "ram"
    ]
    
    print("This example uses YOLO11s with Adam optimizer and cosine learning rate scheduler")
    user_input = input("Run this example? (y/n): ")
    if user_input.lower() == 'y':
        run_command(cmd2, "YOLO11s with custom settings")
    
    # Example 3: Training with data augmentation
    print("\n📋 Example 3: YOLO11m with heavy augmentation")
    cmd3 = [
        sys.executable, "yolo11_train.py",
        "--data", "data_infrared.yaml",
        "--weights", "yolo11m.pt",
        "--epochs", "15",
        "--batch-size", "8",
        "--imgsz", "640",
        "--hsv-h", "0.02",
        "--hsv-s", "0.8",
        "--hsv-v", "0.5",
        "--degrees", "10",
        "--translate", "0.2",
        "--scale", "0.8",
        "--mosaic", "0.8",
        "--mixup", "0.1",
        "--project", "runs/yolo11_train",
        "--name", "infrared_augmented",
        "--logfile", "training_augmented.log"
    ]
    
    print("This example uses heavy data augmentation for better generalization")
    user_input = input("Run this example? (y/n): ")
    if user_input.lower() == 'y':
        run_command(cmd3, "YOLO11m with heavy augmentation")
    
    # Example 4: Resume training
    print("\n📋 Example 4: Resume training from checkpoint")
    print("This example shows how to resume training from a previous run")
    print("First, make sure you have a previous training run to resume from")
    
    # List available runs
    runs_dir = Path("runs/yolo11_train")
    if runs_dir.exists():
        print("Available training runs:")
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                weights_dir = run_dir / "weights"
                if weights_dir.exists():
                    last_pt = weights_dir / "last.pt"
                    if last_pt.exists():
                        print(f"  - {run_dir.name}: {last_pt}")
    
    resume_path = input("Enter path to last.pt file (or press Enter to skip): ")
    if resume_path.strip():
        cmd4 = [
            sys.executable, "yolo11_train.py",
            "--data", "data_infrared.yaml",
            "--weights", resume_path,
            "--resume", "True",
            "--logfile", "training_resumed.log"
        ]
        run_command(cmd4, "Resume training from checkpoint")
    
    print("\n🎉 Training examples completed!")
    print("\nTips:")
    print("- Adjust batch size based on your GPU memory")
    print("- Use smaller models (yolo11n, yolo11s) for faster training")
    print("- Check the logs for training progress and metrics")
    print("- Results are saved in runs/yolo11_train/")

if __name__ == "__main__":
    main() 
