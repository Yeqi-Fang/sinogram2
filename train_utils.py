import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Metrics for evaluation
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error between two images"""
    return torch.mean(torch.abs(img1 - img2)).item()

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    """
    Train the model for one epoch
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        scaler: Gradient scaler for mixed precision training (optional)
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    
    with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
        for batch in pbar:
            # Get data and move to device
            inputs = batch['incomplete'].to(device)
            targets = batch['complete'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with or without mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward and optimize with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            
            # Calculate PSNR (on CPU to save GPU memory)
            with torch.no_grad():
                psnr_val = calculate_psnr(
                    outputs.detach().cpu(), 
                    targets.detach().cpu()
                )
                psnr_meter.update(psnr_val, batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'psnr': f"{psnr_meter.avg:.2f}"
            })
    
    return losses.avg, psnr_meter.avg

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        avg_loss: Average validation loss
        avg_psnr: Average PSNR
    """
    model.eval()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    mae_meter = AverageMeter()
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for batch in pbar:
                # Get data and move to device
                inputs = batch['incomplete'].to(device)
                targets = batch['complete'].to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                batch_size = inputs.size(0)
                losses.update(loss.item(), batch_size)
                
                # Calculate PSNR and MAE on CPU
                psnr_val = calculate_psnr(outputs.cpu(), targets.cpu())
                mae_val = calculate_mae(outputs.cpu(), targets.cpu())
                
                psnr_meter.update(psnr_val, batch_size)
                mae_meter.update(mae_val, batch_size)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses.avg:.4f}",
                    'psnr': f"{psnr_meter.avg:.2f}",
                    'mae': f"{mae_meter.avg:.4f}"
                })
    
    return losses.avg, psnr_meter.avg, mae_meter.avg

def save_model(model, optimizer, scheduler, epoch, val_metrics, model_dir, is_best=False):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        val_metrics: Validation metrics
        model_dir: Directory to save model
        is_best: Whether this is the best model so far
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_metrics.get('loss', 0),
        'val_psnr': val_metrics.get('psnr', 0),
        'val_mae': val_metrics.get('mae', 0)
    }
    
    # Save periodic checkpoint
    torch.save(checkpoint, os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Overwrite best model if this is the best
    if is_best:
        torch.save(checkpoint, os.path.join(model_dir, 'best_model.pth'))
        print(f"Saved best model with PSNR: {val_metrics.get('psnr', 0):.2f}")

def plot_training_curves(train_losses, val_losses, train_psnr, val_psnr, output_path):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_psnr: List of training PSNR values
        val_psnr: List of validation PSNR values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot PSNR
    plt.subplot(1, 2, 2)
    plt.plot(train_psnr, label='Train PSNR (dB)')
    plt.plot(val_psnr, label='Validation PSNR (dB)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_prediction_samples(model, dataset, device, output_dir, epoch, num_samples=5):
    """
    Save sample predictions for visual evaluation
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        device: Device to run inference on
        output_dir: Directory to save samples
        epoch: Current epoch
        num_samples: Number of samples to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Choose random samples to visualize
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            # Get sample
            sample = dataset[idx]
            incomplete = sample['incomplete'].unsqueeze(0).to(device)
            complete = sample['complete'].unsqueeze(0)
            
            # Make prediction
            prediction = model(incomplete).cpu().squeeze(0)
            
            # Get metadata
            vol_idx = sample['volume_idx']
            slice_idx = sample['slice_idx']
            
            # Convert tensors to numpy for plotting
            incomplete = incomplete.cpu().squeeze(0).squeeze(0).numpy()
            complete = complete.squeeze(0).squeeze(0).numpy()
            prediction = prediction.squeeze(0).numpy()
            
            # Create figure
            plt.figure(figsize=(15, 5))
            
            # Plot incomplete sinogram
            plt.subplot(1, 3, 1)
            plt.imshow(incomplete, cmap='hot', aspect='auto')
            plt.title(f"Incomplete Sinogram (V{vol_idx}, S{slice_idx})")
            plt.colorbar()
            
            # Plot model prediction
            plt.subplot(1, 3, 2)
            plt.imshow(prediction, cmap='hot', aspect='auto')
            plt.title(f"Predicted Complete Sinogram")
            plt.colorbar()
            
            # Plot ground truth
            plt.subplot(1, 3, 3)
            plt.imshow(complete, cmap='hot', aspect='auto')
            plt.title(f"Ground Truth Complete Sinogram")
            plt.colorbar()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i+1}_epoch_{epoch+1}.png'))
            plt.close()
            
            # Additionally save the difference map
            plt.figure(figsize=(15, 5))
            
            # Plot difference between prediction and ground truth
            plt.subplot(1, 3, 1)
            diff = np.abs(prediction - complete)
            plt.imshow(diff, cmap='hot', aspect='auto')
            plt.title(f"Absolute Difference (Pred - GT)")
            plt.colorbar()
            
            # Plot normalized difference
            plt.subplot(1, 3, 2)
            max_val = max(np.max(complete), np.max(prediction))
            if max_val > 0:
                norm_diff = diff / max_val
            else:
                norm_diff = diff
            plt.imshow(norm_diff, cmap='hot', aspect='auto')
            plt.title(f"Normalized Difference")
            plt.colorbar()
            
            # Plot residual predicted by model
            plt.subplot(1, 3, 3)
            residual = prediction - incomplete
            plt.imshow(residual, cmap='hot', aspect='auto')
            plt.title(f"Predicted Residual")
            plt.colorbar()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'diff_{i+1}_epoch_{epoch+1}.png'))
            plt.close()