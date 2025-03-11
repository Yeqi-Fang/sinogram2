import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class SinogramSliceDataset(Dataset):
    """
    Dataset for handling 2D sinogram slices from 3D volumes.
    Each 3D sinogram (H, W, D) is split into D slices of (H, W, 1).
    """
    def __init__(self, data_dir, is_train=True, subset_size=None, transform=None, slice_sampling=1):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing complete and incomplete sinogram data
            is_train (bool): Whether to use training or test split
            subset_size (int, optional): Limit to a subset of available volumes (for faster development)
            transform (callable, optional): Optional transform to apply to the samples
            slice_sampling (int): Sample every Nth slice to reduce dataset size (default=1 means use all slices)
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        self.slice_sampling = slice_sampling
        
        # Determine which subdirectory to load from
        subset_dir = "train" if is_train else "test"
        self.base_dir = os.path.join(data_dir, subset_dir)
        
        # Get lists of complete and incomplete sinogram files
        self.complete_files = sorted([f for f in os.listdir(self.base_dir) if f.startswith("complete_")])
        self.incomplete_files = sorted([f for f in os.listdir(self.base_dir) if f.startswith("incomplete_")])
        
        # Limit to subset if specified
        if subset_size is not None and subset_size < len(self.complete_files):
            self.complete_files = self.complete_files[:subset_size]
            self.incomplete_files = self.incomplete_files[:subset_size]
        
        # Ensure matching file counts
        assert len(self.complete_files) == len(self.incomplete_files), \
            f"Mismatch between complete ({len(self.complete_files)}) and incomplete ({len(self.incomplete_files)}) file counts"
            
        print(f"Found {len(self.complete_files)} {'training' if is_train else 'testing'} volume pairs")
        
        # Index mapping from dataset index to (volume_idx, slice_idx)
        self.index_mapping = []
        self._create_index_mapping()
        
    def _create_index_mapping(self):
        """Create mapping from dataset index to (volume_idx, slice_idx)"""
        self.index_mapping = []
        
        # Load one file to get the depth dimension
        sample_file = os.path.join(self.base_dir, self.complete_files[0])
        sample_data = np.load(sample_file)
        depth = sample_data.shape[2]
        
        # Create the mapping for all volumes and slices
        for vol_idx in range(len(self.complete_files)):
            for slice_idx in range(0, depth, self.slice_sampling):
                self.index_mapping.append((vol_idx, slice_idx))
                
        print(f"Created dataset with {len(self.index_mapping)} slices (sampling every {self.slice_sampling} slices)")
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        # Get volume and slice indices
        vol_idx, slice_idx = self.index_mapping[idx]
        
        # Load complete and incomplete data
        complete_path = os.path.join(self.base_dir, self.complete_files[vol_idx])
        incomplete_path = os.path.join(self.base_dir, self.incomplete_files[vol_idx])
        
        complete_volume = np.load(complete_path)
        incomplete_volume = np.load(incomplete_path)
        
        # Extract specific slice
        complete_slice = complete_volume[:, :, slice_idx:slice_idx+1]
        incomplete_slice = incomplete_volume[:, :, slice_idx:slice_idx+1]
        
        # Convert to PyTorch tensors (add channel dimension for 2D convolutions)
        complete_slice = torch.from_numpy(complete_slice).float().permute(2, 0, 1)  # [1, H, W]
        incomplete_slice = torch.from_numpy(incomplete_slice).float().permute(2, 0, 1)  # [1, H, W]
        
        # Apply transforms if any
        if self.transform:
            complete_slice = self.transform(complete_slice)
            incomplete_slice = self.transform(incomplete_slice)
            
        return {'incomplete': incomplete_slice, 'complete': complete_slice, 
                'volume_idx': vol_idx, 'slice_idx': slice_idx}

# Memory-optimized version that preloads all data
class PreloadedSinogramSliceDataset(Dataset):
    """
    Memory-optimized dataset that preloads all sinogram data.
    For faster training at the cost of memory usage.
    """
    def __init__(self, data_dir, is_train=True, subset_size=None, transform=None, slice_sampling=1):
        """
        Initialize the dataset and preload all data into memory.
        
        Args:
            data_dir (str): Directory containing complete and incomplete sinogram data
            is_train (bool): Whether to use training or test split
            subset_size (int, optional): Limit to a subset of available volumes
            transform (callable, optional): Optional transform to apply to the samples
            slice_sampling (int): Sample every Nth slice (default=1 means use all slices)
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        self.slice_sampling = slice_sampling
        
        # Determine which subdirectory to load from
        subset_dir = "train" if is_train else "test"
        self.base_dir = os.path.join(data_dir, subset_dir)
        
        # Get lists of complete and incomplete sinogram files
        self.complete_files = sorted([f for f in os.listdir(self.base_dir) if f.startswith("complete_")])
        self.incomplete_files = sorted([f for f in os.listdir(self.base_dir) if f.startswith("incomplete_")])
        
        # Limit to subset if specified
        if subset_size is not None and subset_size < len(self.complete_files):
            self.complete_files = self.complete_files[:subset_size]
            self.incomplete_files = self.incomplete_files[:subset_size]
        
        # Ensure matching file counts
        assert len(self.complete_files) == len(self.incomplete_files), \
            f"Mismatch between complete ({len(self.complete_files)}) and incomplete ({len(self.incomplete_files)}) file counts"
            
        print(f"Found {len(self.complete_files)} {'training' if is_train else 'testing'} volume pairs")
        print("Preloading all data into memory (this may take a while)...")
        
        # Preload all volumes
        self.complete_slices = []
        self.incomplete_slices = []
        self.vol_slice_indices = []
        
        # Load and process each volume
        for vol_idx in tqdm(range(len(self.complete_files)), desc="Preloading volumes"):
            complete_path = os.path.join(self.base_dir, self.complete_files[vol_idx])
            incomplete_path = os.path.join(self.base_dir, self.incomplete_files[vol_idx])
            
            complete_volume = np.load(complete_path)
            incomplete_volume = np.load(incomplete_path)
            
            depth = complete_volume.shape[2]
            
            # Extract and store each slice
            for slice_idx in range(0, depth, slice_sampling):
                complete_slice = complete_volume[:, :, slice_idx].astype(np.float32)
                incomplete_slice = incomplete_volume[:, :, slice_idx].astype(np.float32)
                
                # Store as NumPy arrays to save memory (convert to tensors in __getitem__)
                self.complete_slices.append(complete_slice)
                self.incomplete_slices.append(incomplete_slice)
                self.vol_slice_indices.append((vol_idx, slice_idx))
        
        print(f"Successfully preloaded {len(self.complete_slices)} slices into memory")
        
    def __len__(self):
        return len(self.complete_slices)
    
    def __getitem__(self, idx):
        # Get the slices and indices
        complete_slice = self.complete_slices[idx]
        incomplete_slice = self.incomplete_slices[idx]
        vol_idx, slice_idx = self.vol_slice_indices[idx]
        
        # Convert to PyTorch tensors with proper channel dimension
        complete_tensor = torch.from_numpy(complete_slice).float().unsqueeze(0)  # [1, H, W]
        incomplete_tensor = torch.from_numpy(incomplete_slice).float().unsqueeze(0)  # [1, H, W]
        
        # Apply transforms if any
        if self.transform:
            complete_tensor = self.transform(complete_tensor)
            incomplete_tensor = self.transform(incomplete_tensor)
            
        return {'incomplete': incomplete_tensor, 'complete': complete_tensor, 
                'volume_idx': vol_idx, 'slice_idx': slice_idx}