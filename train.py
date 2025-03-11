import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Import our modules
from sinogram_dataset import SinogramSliceDataset, PreloadedSinogramSliceDataset
from unet_model import create_unet_model
from train_utils import (
    train_one_epoch, validate, save_model, plot_training_curves,
    save_prediction_samples
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train U-Net for incomplete ring PET reconstruction")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the datasets")
    parser.add_argument("--height", type=int, default=182,
                        help="Height of the sinogram slices")
    parser.add_argument("--width", type=int, default=365,
                        help="Width of the sinogram slices")
    parser.add_argument("--depth", type=int, default=1764,
                        help="Depth of the sinogram volumes")
    parser.add_argument("--slice_sampling", type=int, default=8,
                        help="Sample every Nth slice to reduce dataset size (default=8)")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="residual",
                        choices=["standard", "residual", "small", "simple"],
                        help="Type of U-Net model to use")
    parser.add_argument("--bilinear", action="store_true", default=True,
                        help="Use bilinear upsampling in U-Net")
    parser.add_argument("--preload_data", action="store_true", default=False,
                        help="Preload all data into memory for faster training")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (L2 penalty)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of training data to use for validation")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    
    # Mixed precision training
    parser.add_argument("--mixed_precision", action="store_true", default=False,
                        help="Use mixed precision training")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save outputs")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save checkpoints every N epochs")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # Create dataset
    print("Creating dataset...")
    if args.preload_data:
        dataset_class = PreloadedSinogramSliceDataset
        print("Using preloaded dataset (all data will be loaded into memory)")
    else:
        dataset_class = SinogramSliceDataset
        print("Using streaming dataset (data loaded on demand)")
    
    train_dataset = dataset_class(
        args.data_dir, 
        is_train=True,
        slice_sampling=args.slice_sampling
    )
    
    # Create validation split
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print(f"Creating {args.model_type} U-Net model...")
    model = create_unet_model(
        model_type=args.model_type,
        in_channels=1,  # Each slice has 1 channel
        out_channels=1, # Output is also 1 channel
        bilinear=args.bilinear
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Training/validation history
    train_losses = []
    val_losses = []
    train_psnr = []
    val_psnr = []
    best_val_psnr = 0
    patience_counter = 0
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss, train_epoch_psnr = train_one_epoch(
            model, 
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            scaler=scaler
        )
        
        # Validate
        val_loss, val_epoch_psnr, val_mae = validate(
            model,
            val_loader,
            criterion,
            device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_psnr.append(train_epoch_psnr)
        val_psnr.append(val_epoch_psnr)
        
        # Check if this is the best model
        is_best = val_epoch_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_epoch_psnr
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save metrics for current epoch
        val_metrics = {
            'loss': val_loss,
            'psnr': val_epoch_psnr,
            'mae': val_mae
        }
        
        # Save model periodically
        if (epoch + 1) % args.save_interval == 0 or is_best:
            save_model(
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                model_dir,
                is_best=is_best
            )
            
            # Save sample predictions
            save_prediction_samples(
                model,
                val_dataset,
                device,
                os.path.join(samples_dir, f'epoch_{epoch+1}'),
                epoch
            )
            
            # Plot training curves
            plot_training_curves(
                train_losses,
                val_losses,
                train_psnr,
                val_psnr,
                os.path.join(args.output_dir, 'training_curves.png')
            )
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, PSNR: {train_epoch_psnr:.2f} - "
              f"Val Loss: {val_loss:.4f}, PSNR: {val_epoch_psnr:.2f}, MAE: {val_mae:.4f} - "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    # Save final model
    save_model(
        model,
        optimizer,
        scheduler,
        epoch,
        val_metrics,
        model_dir,
        is_best=False
    )
    
    # Final evaluation message
    print(f"Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Training complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()