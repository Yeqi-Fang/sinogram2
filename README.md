# Incomplete Ring PET Reconstruction

This repository contains a deep learning solution for reconstructing complete sinograms from incomplete ring PET data. The implementation uses U-Net based architectures to process 2D sinogram slices and reconstruct the missing data.

## Overview

Positron Emission Tomography (PET) scanners with incomplete detector rings (missing angles) produce sinograms with missing data, leading to reconstruction artifacts. This project addresses this problem using deep learning to infer the missing sinogram data.

### Key Features

- Various U-Net model variants optimized for sinogram reconstruction
- Support for processing large 3D sinogram volumes (H=182, W=365, D=1764)
- Efficient memory management for large datasets
- Comprehensive visualization tools for sinogram analysis
- Evaluation metrics including PSNR, MAE, and MSE

## File Structure

```
.
├── sinogram_dataset.py      # Dataset classes for handling sinogram data
├── unet_model.py            # U-Net model implementations
├── train_utils.py           # Training and evaluation utilities
├── visualization_utils.py   # Sinogram visualization tools
├── train.py                 # Main training script
├── test.py                  # Testing and evaluation script
└── README.md                # Project documentation
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- h5py
- tqdm

## Dataset Format

The code assumes a directory structure as follows:

```
data/
├── train/
│   ├── complete_000.npy     # Complete sinogram volumes
│   ├── complete_001.npy
│   ├── ...
│   ├── incomplete_000.npy   # Corresponding incomplete sinogram volumes
│   ├── incomplete_001.npy
│   └── ...
└── test/
    ├── complete_000.npy
    ├── complete_001.npy
    ├── ...
    ├── incomplete_000.npy
    ├── incomplete_001.npy
    └── ...
```

Each `.npy` file contains a 3D sinogram volume with shape (182, 365, 1764).

## Usage

### Training

To train a model:

```bash
python train.py --data_dir ./data --model_type residual --batch_size 8 --epochs 100 --output_dir ./output
```

Key arguments:
- `--data_dir`: Directory containing training data
- `--model_type`: Type of U-Net model ("standard", "residual", "small", "simple")
- `--batch_size`: Batch size for training
- `--slice_sampling`: Sample every N slices to reduce dataset size
- `--preload_data`: Preload all data into memory for faster training
- `--mixed_precision`: Use mixed precision training
- `--output_dir`: Directory to save outputs

### Testing

To evaluate a trained model:

```bash
python test.py --data_dir ./data --model_path ./output/models/best_model.pth --model_type residual --save_volumes --output_dir ./results
```

Key arguments:
- `--data_dir`: Directory containing test data
- `--model_path`: Path to trained model checkpoint
- `--model_type`: Type of U-Net model (same as used for training)
- `--save_volumes`: Save reconstructed volumes as HDF5 files
- `--output_dir`: Directory to save test results

## Model Variants

- **Standard**: Full-size U-Net with four levels of encoding/decoding
- **Residual**: U-Net that predicts the residual between incomplete and complete sinograms
- **Small**: U-Net with smaller feature maps for memory efficiency
- **Simple**: Simplified U-Net with fewer layers for faster training

## Performance Metrics

The models are evaluated using:

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **MAE** (Mean Absolute Error): Lower is better
- **MSE** (Mean Squared Error): Lower is better

## Visualization

The visualization utilities provide:

- Detailed single sinogram slice visualizations
- Comparison views of incomplete, predicted, and complete sinograms
- Difference maps to analyze reconstruction errors
- Animated GIFs of 3D sinogram volumes
- Count distributions along different dimensions

## Extending the Code

### Adding New Model Architectures

To add a new model architecture:
1. Implement the model class in `unet_model.py`
2. Add it to the `create_unet_model` factory function
3. Update the model type choices in the argument parsers

### Supporting Different Data Formats

To support different data formats:
1. Modify the `SinogramSliceDataset` class in `sinogram_dataset.py`
2. Adjust the data loading and preprocessing steps

## License

This project is provided for educational and research purposes only.

## Acknowledgments

This implementation draws inspiration from:
- The original U-Net paper by Ronneberger et al.
- PyTorch official examples
- The PyTomography library for PET reconstruction

## Citation

If you use this code in your research, please cite:
