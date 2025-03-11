import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from tqdm import tqdm

def visualize_sinogram_slice(sinogram, title="Sinogram Slice", cmap="hot", 
                             save_path=None, fig_size=(10, 8), aspect="auto"):
    """
    Visualize a single 2D sinogram slice
    
    Args:
        sinogram: 2D NumPy array or PyTorch tensor
        title: Plot title
        cmap: Colormap to use
        save_path: Path to save the figure (if None, display instead)
        fig_size: Figure size
        aspect: Aspect ratio for imshow
    """
    # Convert PyTorch tensor to NumPy if needed
    if torch.is_tensor(sinogram):
        sinogram = sinogram.detach().cpu().numpy()
    
    # Handle channel dimension if present
    if sinogram.ndim == 3 and sinogram.shape[0] == 1:
        sinogram = sinogram[0]
    
    # Create figure
    plt.figure(figsize=fig_size)
    plt.imshow(sinogram, cmap=cmap, aspect=aspect)
    plt.colorbar()
    plt.title(title)
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compare_sinograms(incomplete, predicted, complete, slice_info="", 
                     save_path=None, fig_size=(15, 5), cmap="hot", aspect="auto"):
    """
    Compare incomplete, predicted, and complete sinograms
    
    Args:
        incomplete: Incomplete sinogram slice
        predicted: Model's prediction
        complete: Ground truth complete sinogram
        slice_info: Additional info to include in titles
        save_path: Path to save the figure
        fig_size: Figure size
        cmap: Colormap to use
        aspect: Aspect ratio for imshow
    """
    # Convert PyTorch tensors to NumPy if needed
    if torch.is_tensor(incomplete):
        incomplete = incomplete.detach().cpu().numpy()
    if torch.is_tensor(predicted):
        predicted = predicted.detach().cpu().numpy()
    if torch.is_tensor(complete):
        complete = complete.detach().cpu().numpy()
    
    # Handle channel dimension if present
    if incomplete.ndim == 3 and incomplete.shape[0] == 1:
        incomplete = incomplete[0]
    if predicted.ndim == 3 and predicted.shape[0] == 1:
        predicted = predicted[0]
    if complete.ndim == 3 and complete.shape[0] == 1:
        complete = complete[0]
    
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Plot incomplete sinogram
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(incomplete, cmap=cmap, aspect=aspect)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    ax1.set_title(f"Incomplete Sinogram {slice_info}")
    
    # Plot predicted sinogram
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(predicted, cmap=cmap, aspect=aspect)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    ax2.set_title(f"Predicted Sinogram {slice_info}")
    
    # Plot complete sinogram
    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(complete, cmap=cmap, aspect=aspect)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)
    ax3.set_title(f"Ground Truth {slice_info}")
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_difference_maps(predicted, complete, incomplete=None, slice_info="",
                             save_path=None, fig_size=(15, 5), cmap="hot", aspect="auto"):
    """
    Visualize difference maps to analyze model performance
    
    Args:
        predicted: Model's prediction
        complete: Ground truth complete sinogram
        incomplete: Original incomplete sinogram (optional)
        slice_info: Additional info to include in titles
        save_path: Path to save the figure
        fig_size: Figure size
        cmap: Colormap to use
        aspect: Aspect ratio for imshow
    """
    # Convert PyTorch tensors to NumPy if needed
    if torch.is_tensor(predicted):
        predicted = predicted.detach().cpu().numpy()
    if torch.is_tensor(complete):
        complete = complete.detach().cpu().numpy()
    if incomplete is not None and torch.is_tensor(incomplete):
        incomplete = incomplete.detach().cpu().numpy()
    
    # Handle channel dimension if present
    if predicted.ndim == 3 and predicted.shape[0] == 1:
        predicted = predicted[0]
    if complete.ndim == 3 and complete.shape[0] == 1:
        complete = complete[0]
    if incomplete is not None and incomplete.ndim == 3 and incomplete.shape[0] == 1:
        incomplete = incomplete[0]
    
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Plot absolute difference
    ax1 = plt.subplot(1, 3, 1)
    diff = np.abs(predicted - complete)
    im1 = ax1.imshow(diff, cmap=cmap, aspect=aspect)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    ax1.set_title(f"Absolute Difference {slice_info}")
    
    # Plot normalized difference
    ax2 = plt.subplot(1, 3, 2)
    max_val = max(np.max(complete), np.max(predicted))
    if max_val > 0:
        norm_diff = diff / max_val
    else:
        norm_diff = diff
    im2 = ax2.imshow(norm_diff, cmap=cmap, aspect=aspect)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    ax2.set_title(f"Normalized Difference {slice_info}")
    
    # Plot residual if incomplete is provided
    ax3 = plt.subplot(1, 3, 3)
    if incomplete is not None:
        residual = predicted - incomplete
        im3 = ax3.imshow(residual, cmap=cmap, aspect=aspect)
        title = f"Predicted Residual {slice_info}"
    else:
        # If incomplete not provided, show error distribution
        error_hist = diff.flatten()
        im3 = ax3.hist(error_hist, bins=50)
        title = f"Error Distribution {slice_info}"
    
    if incomplete is not None:
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax3)
    ax3.set_title(title)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_animated_sinogram_gif(volume, start_idx=0, end_idx=None, step=1, 
                                title="Sinogram Animation", save_path="animation.gif",
                                cmap="hot", aspect="auto", duration=100):
    """
    Create an animated GIF from a 3D sinogram volume
    
    Args:
        volume: 3D NumPy array or PyTorch tensor (H, W, D) or (D, H, W)
        start_idx: Starting slice index
        end_idx: Ending slice index (default: last slice)
        step: Step size between slices
        title: Animation title
        save_path: Path to save the GIF
        cmap: Colormap to use
        aspect: Aspect ratio for imshow
        duration: Duration per frame in milliseconds
    """
    try:
        import imageio
        from PIL import Image
    except ImportError:
        print("Error: This function requires imageio and Pillow. Install with:")
        print("pip install imageio pillow")
        return
    
    # Convert PyTorch tensor to NumPy if needed
    if torch.is_tensor(volume):
        volume = volume.detach().cpu().numpy()
    
    # Ensure 3D format is (H, W, D)
    if volume.ndim == 3:
        if volume.shape[0] < volume.shape[1] and volume.shape[0] < volume.shape[2]:
            # Format is likely (D, H, W)
            volume = volume.transpose(1, 2, 0)
    else:
        raise ValueError("Volume must be 3D with shape (H, W, D) or (D, H, W)")
    
    # Set default end index
    if end_idx is None:
        end_idx = volume.shape[2] - 1
    
    # Ensure valid indices
    start_idx = max(0, start_idx)
    end_idx = min(volume.shape[2] - 1, end_idx)
    
    # Create temporary directory for frames
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Generate frames
    frames = []
    for i in tqdm(range(start_idx, end_idx + 1, step), desc="Generating frames"):
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(volume[:, :, i], cmap=cmap, aspect=aspect)
        plt.colorbar()
        plt.title(f"{title} (Slice {i})")
        
        # Save frame to temporary file
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Add frame to list
        frames.append(imageio.imread(frame_path))
    
    # Save animation
    imageio.mimsave(save_path, frames, duration=duration/1000)
    print(f"Animation saved to {save_path}")
    
    # Clean up temporary files
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

def visualize_reconstructions_from_h5(h5_path, output_dir=None, num_slices=5, cmap="hot", aspect="auto"):
    """
    Visualize reconstructions from an H5 file containing incomplete, predicted, and complete sinograms
    
    Args:
        h5_path: Path to the H5 file
        output_dir: Directory to save visualizations (if None, use same directory as H5 file)
        num_slices: Number of slices to visualize
        cmap: Colormap to use
        aspect: Aspect ratio for imshow
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(h5_path)
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    
    # Load data
    with h5py.File(h5_path, 'r') as f:
        incomplete = f['incomplete'][:]
        reconstructed = f['reconstructed'][:]
        complete = f['complete'][:]
        
        # Get metadata if available
        missing_angle_info = f.attrs.get('missing_angle_info', "")
    
    # Get dimensions
    depth = incomplete.shape[2]
    
    # Select slices to visualize
    if num_slices >= depth:
        slice_indices = np.arange(depth)
    else:
        slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    # Create visualizations
    for slice_idx in slice_indices:
        # Compare sinograms
        compare_sinograms(
            incomplete[:, :, slice_idx],
            reconstructed[:, :, slice_idx],
            complete[:, :, slice_idx],
            slice_info=f"(Slice {slice_idx}, {missing_angle_info})",
            save_path=os.path.join(output_dir, f"{base_name}_slice_{slice_idx}_comparison.png"),
            cmap=cmap,
            aspect=aspect
        )
        
        # Visualize differences
        visualize_difference_maps(
            reconstructed[:, :, slice_idx],
            complete[:, :, slice_idx],
            incomplete[:, :, slice_idx],
            slice_info=f"(Slice {slice_idx}, {missing_angle_info})",
            save_path=os.path.join(output_dir, f"{base_name}_slice_{slice_idx}_differences.png"),
            cmap=cmap,
            aspect=aspect
        )
    
    # Create animated GIFs if depth > 1
    if depth > 1:
        # Animation of incomplete sinograms
        create_animated_sinogram_gif(
            incomplete,
            title=f"Incomplete Sinogram {missing_angle_info}",
            save_path=os.path.join(output_dir, f"{base_name}_incomplete.gif"),
            cmap=cmap,
            aspect=aspect
        )
        
        # Animation of reconstructed sinograms
        create_animated_sinogram_gif(
            reconstructed,
            title=f"Reconstructed Sinogram {missing_angle_info}",
            save_path=os.path.join(output_dir, f"{base_name}_reconstructed.gif"),
            cmap=cmap,
            aspect=aspect
        )
        
        # Animation of difference maps
        diff_volume = np.abs(reconstructed - complete)
        create_animated_sinogram_gif(
            diff_volume,
            title=f"Absolute Difference {missing_angle_info}",
            save_path=os.path.join(output_dir, f"{base_name}_difference.gif"),
            cmap=cmap,
            aspect=aspect
        )

# Function based on the provided sinogram visualization code
def visualize_single_sinogram(sinogram, output_path, title="Sinogram Visualization", image_filename=None):
    """
    Create detailed visualizations for a single sinogram.
    
    Args:
        sinogram: The sinogram data (tensor or numpy array)
        output_path: Path to save the visualization
        title: Title for the visualization
        image_filename: Original image filename for reference
    """
    # Use non-interactive backend
    matplotlib.use('Agg')
    
    try:
        # Ensure sinogram is numpy array
        if torch.is_tensor(sinogram):
            sinogram_np = sinogram.cpu().numpy()
        else:
            sinogram_np = sinogram
        
        sinogram_shape = sinogram_np.shape
        print(f"Sinogram shape: {sinogram_shape}")
        
        # Create multi-page figure - one view per page
        fig = plt.figure(figsize=(12, 8))
        
        # 1. Single slice view - first 42 slices of first angle
        ax1 = plt.subplot(221)
        slice_idx = min(42, sinogram_shape[2])
        im1 = ax1.imshow(sinogram_np[0, :, :slice_idx], cmap='magma')
        ax1.set_title(f'First Angle (First {slice_idx} Slices)')
        ax1.set_xlabel('Ring Index')
        ax1.set_ylabel('Radial Position')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. Multiple slice view - select middle slice
        ax2 = plt.subplot(222)
        middle_slice = min(20, sinogram_shape[2]-1)
        im2 = ax2.imshow(sinogram_np[:, :, middle_slice], cmap='magma', aspect='auto')
        ax2.set_title(f'Detailed View (Ring {middle_slice})')
        ax2.set_xlabel('Radial Position')
        ax2.set_ylabel('Angle')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. Cross-section view - fixed radial position, showing angle vs ring
        ax3 = plt.subplot(223)
        middle_radial = sinogram_shape[1] // 2
        im3 = ax3.imshow(sinogram_np[:, middle_radial, :min(64, sinogram_shape[2])], cmap='magma', aspect='auto')
        ax3.set_title(f'Angle vs Ring (Middle Radial Position)')
        ax3.set_xlabel('Ring Index')
        ax3.set_ylabel('Angle')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Count distribution
        ax4 = plt.subplot(224)
        # Calculate total counts by angle and radial position
        angle_counts = sinogram_np.sum(axis=(1, 2))
        radial_counts = sinogram_np.sum(axis=(0, 2))
        
        # Create dual Y-axis plot
        ax4.set_xlabel('Index')
        ax4.set_ylabel('Angle Total Counts', color='tab:blue')
        ax4.plot(range(len(angle_counts)), angle_counts, 'b-', label='Angle Counts')
        ax4.tick_params(axis='y', labelcolor='tab:blue')
        
        ax4_twin = ax4.twinx()
        ax4_twin.set_ylabel('Radial Total Counts', color='tab:red')
        ax4_twin.plot(range(len(radial_counts)), radial_counts, 'r-', label='Radial Counts')
        ax4_twin.tick_params(axis='y', labelcolor='tab:red')
        
        ax4.set_title('Total Counts Distribution')
        
        # Add overall title
        if image_filename:
            fig.suptitle(f"{title} ({image_filename})", fontsize=16)
        else:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make space for title
        
        # Save image
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"  -> Saved sinogram visualization to {output_path}")
        
    except Exception as e:
        print(f"Warning: Failed to create sinogram visualization: {e}")
        import traceback
        traceback.print_exc()

# Example usage
if __name__ == "__main__":
    # Example: Visualize random sinogram
    test_sinogram = np.random.rand(182, 365, 64)
    visualize_sinogram_slice(test_sinogram[:, :, 0], title="Example Sinogram Slice")
    
    # Example: Compare sinograms
    incomplete = np.random.rand(182, 365) * 0.7
    predicted = np.random.rand(182, 365) * 0.9
    complete = np.random.rand(182, 365)
    compare_sinograms(incomplete, predicted, complete, slice_info="(Example)")
    
    # Example: Visualize differences
    visualize_difference_maps(predicted, complete, incomplete, slice_info="(Example)")