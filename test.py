import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

# Import our modules
from sinogram_dataset import SinogramSliceDataset
from unet_model import create_unet_model
from train_utils import calculate_psnr, calculate_mae, AverageMeter

def evaluate_model(model, dataloader, device):
    """
    Evaluate model performance on a dataset
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with test data
        device: Device to run inference on
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    
    # Metrics
    psnr_meter = AverageMeter()
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            inputs = batch['incomplete'].to(device)
            targets = batch['complete'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate metrics on CPU
            outputs_cpu = outputs.cpu()
            targets_cpu = targets.cpu()
            
            # Update metrics
            batch_size = inputs.size(0)
            psnr_val = calculate_psnr(outputs_cpu, targets_cpu)
            psnr_meter.update(psnr_val, batch_size)
            
            mae_val = calculate_mae(outputs_cpu, targets_cpu)
            mae_meter.update(mae_val, batch_size)
            
            mse_val = torch.mean((outputs_cpu - targets_cpu) ** 2).item()
            mse_meter.update(mse_val, batch_size)
    
    return {
        'psnr': psnr_meter.avg,
        'mae': mae_meter.avg,
        'mse': mse_meter.avg
    }

def reconstruct_sinogram_volume(model, dataset, output_path, device, missing_angle_info=None):
    """
    Reconstruct and save complete sinogram volumes from dataset
    
    Args:
        model: Trained model
        dataset: Dataset with incomplete sinograms
        output_path: Path to save reconstructed volumes
        device: Device to run inference on
        missing_angle_info: Description of missing angles (for visualization)
    """
    model.eval()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Group indices by volume
    volume_indices = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        vol_idx = sample['volume_idx']
        slice_idx = sample['slice_idx']
        
        if vol_idx not in volume_indices:
            volume_indices[vol_idx] = []
        
        volume_indices[vol_idx].append((idx, slice_idx))
    
    # Process each volume separately
    for vol_idx, indices in volume_indices.items():
        print(f"Processing volume {vol_idx}...")
        
        # Sort indices by slice_idx
        indices.sort(key=lambda x: x[1])
        
        # Get volume dimensions from first sample
        first_sample = dataset[indices[0][0]]
        slice_shape = first_sample['incomplete'].shape[1:] # (H, W)
        
        # Get depth from highest slice index
        max_slice_idx = max(indices, key=lambda x: x[1])[1]
        depth = max_slice_idx + 1
        
        # Initialize arrays for storing results
        incomplete_volume = np.zeros((slice_shape[0], slice_shape[1], depth))
        reconstructed_volume = np.zeros((slice_shape[0], slice_shape[1], depth))
        complete_volume = np.zeros((slice_shape[0], slice_shape[1], depth))
        
        # Process slices
        with torch.no_grad():
            for dataset_idx, slice_idx in tqdm(indices, desc=f"Volume {vol_idx}"):
                # Get sample
                sample = dataset[dataset_idx]
                incomplete = sample['incomplete'].unsqueeze(0).to(device)
                complete = sample['complete']
                
                # Forward pass
                reconstructed = model(incomplete).cpu().squeeze(0)
                
                # Store results
                incomplete_volume[:, :, slice_idx] = incomplete.cpu().squeeze(0).squeeze(0).numpy()
                reconstructed_volume[:, :, slice_idx] = reconstructed.squeeze(0).numpy()
                complete_volume[:, :, slice_idx] = complete.squeeze(0).numpy()
        
        # Save volumes
        output_file = os.path.join(output_path, f"volume_{vol_idx}.h5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('incomplete', data=incomplete_volume)
            f.create_dataset('reconstructed', data=reconstructed_volume)
            f.create_dataset('complete', data=complete_volume)
            
            # Store metadata
            if missing_angle_info:
                f.attrs['missing_angle_info'] = missing_angle_info
        
        print(f"Saved volume {vol_idx} to {output_file}")
        
        # Create visualizations
        vis_dir = os.path.join(output_path, f"volume_{vol_idx}_vis")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Visualize slices at different depths
        slice_indices = np.linspace(0, depth-1, 5, dtype=int)
        for i, slice_idx in enumerate(slice_indices):
            plt.figure(figsize=(15, 5))
            
            # Plot incomplete sinogram
            plt.subplot(1, 3, 1)
            plt.imshow(incomplete_volume[:, :, slice_idx], cmap='hot', aspect='auto')
            plt.title(f"Incomplete (Slice {slice_idx})")
            plt.colorbar()
            
            # Plot reconstructed sinogram
            plt.subplot(1, 3, 2)
            plt.imshow(reconstructed_volume[:, :, slice_idx], cmap='hot', aspect='auto')
            plt.title(f"Reconstructed")
            plt.colorbar()
            
            # Plot ground truth
            plt.subplot(1, 3, 3)
            plt.imshow(complete_volume[:, :, slice_idx], cmap='hot', aspect='auto')
            plt.title(f"Ground Truth")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"slice_{slice_idx}.png"))
            plt.close()
            
        # Create difference maps
        for i, slice_idx in enumerate(slice_indices):
            plt.figure(figsize=(15, 5))
            
            # Plot difference map
            plt.subplot(1, 3, 1)
            diff = np.abs(reconstructed_volume[:, :, slice_idx] - complete_volume[:, :, slice_idx])
            plt.imshow(diff, cmap='hot', aspect='auto')
            plt.title(f"Absolute Difference (Slice {slice_idx})")
            plt.colorbar()
            
            # Plot normalized difference
            plt.subplot(1, 3, 2)
            max_val = max(np.max(complete_volume[:, :, slice_idx]), np.max(reconstructed_volume[:, :, slice_idx]))
            if max_val > 0:
                norm_diff = diff / max_val
            else:
                norm_diff = diff
            plt.imshow(norm_diff, cmap='hot', aspect='auto')
            plt.title(f"Normalized Difference")
            plt.colorbar()
            
            # Plot residual predicted by model
            plt.subplot(1, 3, 3)
            residual = reconstructed_volume[:, :, slice_idx] - incomplete_volume[:, :, slice_idx]
            plt.imshow(residual, cmap='hot', aspect='auto')
            plt.title(f"Predicted Residual")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"diff_slice_{slice_idx}.png"))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test U-Net for incomplete ring PET reconstruction")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing test data")
    parser.add_argument("--slice_sampling", type=int, default=1,
                        help="Sample every Nth slice (default=1, use all slices)")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="residual",
                        choices=["standard", "residual", "small", "simple"],
                        help="Type of U-Net model used for training")
    parser.add_argument("--bilinear", action="store_true", default=True,
                        help="Use bilinear upsampling in U-Net")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save test results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    
    # Save reconstructed volumes
    parser.add_argument("--save_volumes", action="store_true", default=False,
                        help="Save reconstructed volumes as HDF5 files")
    
    # Missing angle information
    parser.add_argument("--missing_angle_info", type=str, default=None,
                        help="Description of missing angles (e.g., '30-60 degrees')")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = SinogramSliceDataset(
        args.data_dir,
        is_train=False,
        slice_sampling=args.slice_sampling
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    print(f"Creating {args.model_type} U-Net model...")
    model = create_unet_model(
        model_type=args.model_type,
        in_channels=1,
        out_channels=1,
        bilinear=args.bilinear
    ).to(device)
    
    # Load trained model weights
    print(f"Loading model weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate model
    print("Evaluating model performance...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print and save results
    print("\nTest Results:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Checkpoint: {args.model_path}\n")
        f.write(f"PSNR: {metrics['psnr']:.4f} dB\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"MSE: {metrics['mse']:.6f}\n")
    
    # Reconstruct and save volumes if requested
    if args.save_volumes:
        print("\nReconstructing complete sinogram volumes...")
        reconstruct_sinogram_volume(
            model,
            test_dataset,
            os.path.join(args.output_dir, 'volumes'),
            device,
            missing_angle_info=args.missing_angle_info
        )
    
    print(f"Testing complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()