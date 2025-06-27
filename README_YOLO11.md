# YOLO11 Training Setup

This repository contains a comprehensive YOLO11 training setup similar to YOLOv5, with logging, configuration options, and inference capabilities.

## Features

- **Complete YOLO11 training script** with extensive configuration options
- **Logging support** with both console and file output
- **Inference and validation scripts** for model evaluation
- **Example scripts** demonstrating different training scenarios
- **Dataset configuration** for infrared object detection

## Files Overview

- `yolo11_train.py` - Main training script with full argument parsing
- `yolo11_inference.py` - Inference, validation, and export script
- `yolo11_train_example.py` - Example usage demonstrations
- `data_infrared.yaml` - Dataset configuration for infrared data
- `README_YOLO11.md` - This documentation

## Installation

1. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install required packages:**
   ```bash
   pip install ultralytics
   ```

## Quick Start

### 1. Basic Training

Train a YOLO11n model on your infrared dataset:

```bash
python yolo11_train.py \
    --data data_infrared.yaml \
    --weights yolo11n.pt \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --project runs/yolo11_train \
    --name my_experiment \
    --logfile training.log
```

### 2. Training with Custom Settings

```bash
python yolo11_train.py \
    --data data_infrared.yaml \
    --weights yolo11s.pt \
    --epochs 200 \
    --batch-size 8 \
    --imgsz 640 \
    --lr0 0.001 \
    --optimizer Adam \
    --cos-lr \
    --patience 50 \
    --cache ram \
    --logfile custom_training.log
```

### 3. Validation

Validate your trained model:

```bash
python yolo11_inference.py \
    --mode val \
    --weights runs/yolo11_train/my_experiment/weights/best.pt \
    --data data_infrared.yaml \
    --imgsz 640
```

### 4. Inference

Run inference on images:

```bash
python yolo11_inference.py \
    --mode predict \
    --weights runs/yolo11_train/my_experiment/weights/best.pt \
    --source path/to/images \
    --conf 0.25 \
    --save
```

### 5. Export Model

Export to different formats:

```bash
python yolo11_inference.py \
    --mode export \
    --weights runs/yolo11_train/my_experiment/weights/best.pt \
    --format onnx \
    --imgsz 640
```

## Available Models

YOLO11 comes in different sizes:

- `yolo11n.pt` - Nano (fastest, smallest)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (most accurate, largest)

## Training Arguments

### Essential Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | Required | Path to dataset YAML file |
| `--weights` | `yolo11n.pt` | Initial weights path |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 16 | Batch size for training |
| `--imgsz` | 640 | Training image size |

### Optimization Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimizer` | `auto` | Optimizer (SGD, Adam, AdamW) |
| `--lr0` | 0.01 | Initial learning rate |
| `--lrf` | 0.01 | Final learning rate fraction |
| `--momentum` | 0.937 | SGD momentum |
| `--weight-decay` | 0.0005 | Optimizer weight decay |
| `--cos-lr` | False | Use cosine learning rate scheduler |

### Augmentation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hsv-h` | 0.015 | HSV-Hue augmentation |
| `--hsv-s` | 0.7 | HSV-Saturation augmentation |
| `--hsv-v` | 0.4 | HSV-Value augmentation |
| `--degrees` | 0.0 | Image rotation degrees |
| `--translate` | 0.1 | Image translation fraction |
| `--scale` | 0.5 | Image scale factor |
| `--mosaic` | 1.0 | Mosaic augmentation probability |
| `--mixup` | 0.0 | Mixup augmentation probability |

### Hardware and Performance

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | auto | CUDA device (0, 0,1,2,3) or cpu |
| `--workers` | 8 | Number of dataloader workers |
| `--cache` | False | Cache images (ram/disk) |
| `--amp` | True | Automatic Mixed Precision |

### Logging and Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--logfile` | '' | Path to log file |
| `--project` | `runs/train` | Save directory |
| `--name` | `exp` | Experiment name |
| `--save-period` | -1 | Save checkpoint every N epochs |

## Dataset Configuration

Update `data_infrared.yaml` for your dataset:

```yaml
# Dataset path
path: datasets/infrared

# Train/val/test sets
train: images/train
val: images/val
test: images/val

# Number of classes
nc: 1

# Class names
names:
  0: infrared_object
```

## Performance Tips

1. **Batch Size**: Start with batch size 16 and adjust based on GPU memory
2. **Image Size**: Use 640 for good balance of speed/accuracy
3. **Model Size**: Start with `yolo11n` for fast training, upgrade to `yolo11s/m` for better accuracy
4. **Caching**: Use `--cache ram` if you have enough memory
5. **Mixed Precision**: Keep `--amp` enabled for faster training

## Monitoring Training

### Log Files

Training logs are saved to the specified log file and include:
- Training configuration
- Epoch progress
- Loss values
- Validation metrics
- Final results

### TensorBoard (Optional)

If you want TensorBoard visualization:

```bash
# Install tensorboard
pip install tensorboard

# View logs
tensorboard --logdir runs/yolo11_train
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Dataset Not Found**: Check paths in YAML file
3. **Model Loading Error**: Ensure weights file exists and is valid

### Performance Issues

1. **Slow Training**: 
   - Reduce workers if CPU limited
   - Use smaller image size
   - Enable caching

2. **Poor Accuracy**:
   - Increase epochs
   - Try different model size
   - Adjust augmentation parameters
   - Check dataset quality

## Advanced Usage

### Resume Training

```bash
python yolo11_train.py \
    --weights runs/yolo11_train/my_experiment/weights/last.pt \
    --resume True
```

### Multi-GPU Training

YOLO11 automatically detects multiple GPUs. For explicit control:

```bash
python yolo11_train.py \
    --data data_infrared.yaml \
    --weights yolo11n.pt \
    --device 0,1,2,3
```

### Custom Hyperparameters

Create a hyperparameters YAML file and use:

```bash
python yolo11_train.py \
    --data data_infrared.yaml \
    --weights yolo11n.pt \
    --hyp custom_hyp.yaml
```

## Comparison with YOLOv5

### Advantages of YOLO11

- **Better Accuracy**: Higher mAP with fewer parameters
- **Faster Training**: Optimized training pipeline
- **Modern Architecture**: Enhanced backbone and neck
- **Unified Interface**: Single package for all YOLO tasks
- **Better Export**: Improved model export capabilities

### Migration from YOLOv5

- Similar command-line interface
- Compatible dataset formats
- Improved performance out of the box
- Better documentation and support

## Support

For issues and questions:
1. Check the [Ultralytics documentation](https://docs.ultralytics.com/)
2. Visit the [GitHub repository](https://github.com/ultralytics/ultralytics)
3. Join the [Discord community](https://discord.gg/ultralytics)

## License

This project follows the Ultralytics license terms. YOLO11 is available under AGPL-3.0 for open source use and commercial licensing for business applications. 
