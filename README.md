# End-to-End Multi-Modal Image Registration CNN
[简体中文](README_zh.md)
## Project Introduction

This project can be used to create pixel-level aligned multi-modal image pairs for training multi-modal models.

The project utilizes an end-to-end multi-modal image registration CNN, employing a Siamese convolutional network to estimate similarity transformation parameters between two modalities (e.g., infrared and visible light) — rotation angle θ, scaling factor s, and translation vector (Δx, Δy), thereby achieving image alignment.

## Model Architecture

The model adopts a Siamese network architecture with shared weights. The specific structure is shown in the figure below:

![Architecture](CNN.png)

## Usage

### Train the Model

```bash
python train.py --vis_dir PATH_TO_VIS --ir_dir PATH_TO_IR --json_path PATH_TO_JSON --model_dir MODEL_SAVE_DIR
```

Main parameters:
- `--vis_dir`: Directory of visible light images
- `--ir_dir`: Directory of infrared images
- `--json_path`: Path to the transformation parameters JSON file
- `--model_dir`: Directory to save the model
- `--train_percentage`: Percentage of data used for training
- `--batch_size`: Batch size
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--resume`: Resume training from a checkpoint (optional)

You can also omit parameters to use the default values defined in `parse_args()`.

### Model Inference

```bash
python inference.py --vis_dir PATH_TO_VIS --ir_dir PATH_TO_IR --model_path PATH_TO_MODEL --output_dir OUTPUT_DIR
```

Main parameters:
- `--vis_dir`: Directory of visible light images
- `--ir_dir`: Directory of infrared images
- `--model_path`: Path to the trained model
- `--output_dir`: Directory for output results
- `--image_id`: Specific image ID to process (if not specified, all images will be processed)
- `--fusion_mode`: Fusion mode (`average`, `weighted`, `false_color`, `layered`)

You can also omit parameters to use the default values defined in `parse_args()`.

### Inference Results

After inference, the system will generate the following files for each processed image:

1. **Original infrared image**: `ir_original.jpg`
2. **Aligned infrared image**: `ir_aligned.jpg`
3. **Visible light image**: `vis.jpg`
4. **Fused image**: `fused.jpg`
5. **Visualization result**: `fusion.jpg` (includes all images and transformation parameters)
6. **Parameter JSON file**: `params.json` (includes predicted transformation parameters and ground truth)

## Contributors

* [T-Auto](https://github.com/T-Auto)
* [DawnMoon2333](https://github.com/DawnMoon2333/)
