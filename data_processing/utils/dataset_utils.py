import torch
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional


def filter_test_folders(available_folders: List[str], test_folders: List[str], dataset_type: str = "") -> tuple[List[str], List[str]]:
    """
    Filter out test folders from available folders.
    
    Args:
        available_folders: List of all available acquisition folders
        test_folders: List of folders designated for testing
        dataset_type: Type of dataset for logging purposes
        
    Returns:
        Tuple of (train_val_folders, actual_test_folders)
        - train_val_folders: Folders available for train/val split
        - actual_test_folders: Test folders that actually exist in available folders
    """
    if not test_folders:
        return available_folders, []
    
    # Find test folders that actually exist in available folders
    actual_test_folders = [folder for folder in test_folders if folder in available_folders]
    
    # Remove test folders from available folders
    train_val_folders = [folder for folder in available_folders if folder not in test_folders]
    
    if actual_test_folders:
        print(f"{dataset_type} dataset: Using {len(actual_test_folders)} folders for testing, {len(train_val_folders)} for train/val")
        if len(actual_test_folders) != len(test_folders):
            missing_folders = [folder for folder in test_folders if folder not in available_folders]
            print(f"WARNING: {len(missing_folders)} test folders not found in {dataset_type} dataset: {missing_folders}")
    
    return train_val_folders, actual_test_folders


def filter_validation_folders(available_folders: List[str], val_folders: List[str], dataset_type: str = "") -> tuple[List[str], List[str]]:
    """
    Filter out validation folders from available folders.
    
    Args:
        available_folders: List of all available acquisition folders
        val_folders: List of folders designated for validation
        dataset_type: Type of dataset for logging purposes
        
    Returns:
        Tuple of (train_test_folders, actual_val_folders)
        - train_test_folders: Folders available for train/test split
        - actual_val_folders: Validation folders that actually exist in available folders
    """
    if not val_folders:
        return available_folders, []
    
    actual_val_folders = [folder for folder in val_folders if folder in available_folders]
    train_test_folders = [folder for folder in available_folders if folder not in val_folders]
    
    if actual_val_folders:
        print(f"{dataset_type} dataset: Using {len(actual_val_folders)} folders for validation, {len(train_test_folders)} for train/test")
        if len(actual_val_folders) != len(val_folders):
            missing_folders = [folder for folder in val_folders if folder not in available_folders]
            print(f"WARNING: {len(missing_folders)} validation folders not found in {dataset_type} dataset: {missing_folders}")
    
    return train_test_folders, actual_val_folders


def split_dataset_pairs(
    data_pairs: List[Dict], 
    split: str = 'all', 
    test_size: Optional[float] = None, 
    val_size: float = 0.2, 
    random_state: int = 42
) -> List[Dict]:
    """
    Split dataset pairs into train/val/test sets.
    
    Args:
        data_pairs: List of data pair dictionaries
        split: Dataset split to use ('train', 'val', 'test', or 'all')
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        List of data pairs for the specified split
    """
    if len(data_pairs) == 0:
        print("WARNING: No data found. Creating empty dataset.")
        return []
        
    if split == 'all':
        print(f"Using all {len(data_pairs)} samples")
        return data_pairs

    if val_size is None:
        val_size = 0.2

    # Scene-level split: split by acquisition folder, NOT by individual frame.
    # Frames from the same acquisition are temporally adjacent and highly
    # correlated; splitting them across train/val/test would leak near-duplicate
    # frames between splits and inflate validation metrics.
    folders = sorted({pair['folder'] for pair in data_pairs})

    if len(folders) < 2:
        raise ValueError(
            f"Cannot create a scene-level '{split}' split from {len(folders)} acquisition folder(s). "
            "Provide explicit split configs or more acquisition folders."
        )

    # Folder-level split
    if test_size is None or test_size == 0.0:
        train_folders, val_folders = train_test_split(
            folders, test_size=val_size, random_state=random_state
        )
        split_folders = {'train': set(train_folders), 'val': set(val_folders)}
    else:
        train_val_folders, test_folders = train_test_split(
            folders, test_size=test_size, random_state=random_state
        )
        train_folders, val_folders = train_test_split(
            train_val_folders, test_size=val_size / (1 - test_size), random_state=random_state
        )
        split_folders = {
            'train': set(train_folders),
            'val': set(val_folders),
            'test': set(test_folders),
        }

    if split not in split_folders:
        raise ValueError(f"Invalid split '{split}' (test_size={test_size})")

    chosen = split_folders[split]
    result_pairs = [pair for pair in data_pairs if pair['folder'] in chosen]

    print(
        f"Using {len(result_pairs)} samples from {len(chosen)} scenes for {split} split "
        f"(scene-level split of {len(folders)} scenes)"
    )
    return result_pairs


def spectral_collate_fn(batch):
    """
    Custom collate function for spectral datasets that handles variable-length keypoints
    and various tensor types.
    """
    if not batch:
        return {}
        
    batch_dict = {}
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'dataset_type':
            # Store dataset types (for mixed datasets)
            batch_dict[key] = [sample[key] for sample in batch]
        elif key == 'pair_type':
            # Keep per-sample pairing type strings
            batch_dict[key] = [sample[key] for sample in batch]
            
        elif key == 'keypoints' or key == 'scores':
            # Handle variable length keypoints and scores
            if any(key in sample for sample in batch):
                # Get max number of keypoints in batch
                max_kpts = max(sample[key].shape[0] for sample in batch if key in sample)
                
                # Create padded tensor and valid mask
                if key == 'keypoints':
                    padded = torch.zeros(len(batch), max_kpts, 2)
                else:  # scores
                    padded = torch.zeros(len(batch), max_kpts)
                valid_mask = torch.zeros(len(batch), max_kpts, dtype=torch.bool)
                
                # Fill padded tensor and mask
                for i, sample in enumerate(batch):
                    if key in sample:
                        num_kpts = sample[key].shape[0]
                        padded[i, :num_kpts] = sample[key]
                        valid_mask[i, :num_kpts] = True
                
                batch_dict[key] = padded
                batch_dict[f'{key}_mask'] = valid_mask
        
        elif key in ['hsi', 'rgb', 'warped_hsi', 'warped_rgb', 'hsi_np', 'rgb_np', 'hsi_np2']:
            # Stack image tensors if they exist
            if any(key in sample for sample in batch):
                tensors = [sample[key] for sample in batch if key in sample]
                if tensors:
                    shapes = [t.shape for t in tensors]
                    if len(set(shapes)) == 1:
                        batch_dict[key] = torch.stack(tensors)
                    else:
                        raise ValueError(
                            f"Cannot collate '{key}' tensors with different shapes: {shapes}. "
                            "Resize inputs or use batch_size=1."
                        )
        
        elif key == 'H_mat':
            # Stack homography matrices if they exist
            if any(key in sample for sample in batch):
                matrices = [sample[key] for sample in batch if key in sample]
                if matrices:
                    batch_dict[key] = torch.stack(matrices)
        elif key == 'overlap_bbox':
            # Valid RGB/HSI overlap region [x0, y0, x1, y1] (per-sample, constant per dataset)
            if any(key in sample for sample in batch):
                boxes = [sample[key] for sample in batch if key in sample]
                if boxes:
                    batch_dict[key] = torch.stack(boxes)
        elif key in ['K0', 'K1', 'R_01', 't_01', 'K2', 'R_02', 't_02']:
            # Stack intrinsics and relative pose if present (non-planar pairs;
            # K2/R_02/t_02 = optional 3rd view for multi-view reprojection, #2)
            if any(key in sample for sample in batch):
                mats = []
                for sample in batch:
                    if key in sample:
                        val = sample[key]
                        val = torch.as_tensor(val) if not torch.is_tensor(val) else val
                        mats.append(val.float())
                if mats:
                    batch_dict[key] = torch.stack(mats)

        elif key in ['hsi_pose', 'rgb_pose']:
            # Stack pose vectors if they exist
            if any(key in sample for sample in batch):
                poses = [sample[key] for sample in batch if key in sample]
                if poses:
                    batch_dict[key] = torch.stack([torch.tensor(pose) for pose in poses])

        elif key in ['hsi_path', 'rgb_path', 'hsi_np_path', 'rgb_np_path', 'hsi_np2_path']:
            # Collect paths as lists
            if any(key in sample for sample in batch):
                paths = [sample[key] for sample in batch if key in sample]
                if paths:
                    batch_dict[key] = paths
    
    has_planar = []
    has_nonplanar = []
    for sample in batch:
        has_planar.append(('warped_hsi' in sample) or ('warped_rgb' in sample) or ('H_mat' in sample))
        has_nonplanar.append(('hsi_np' in sample) and all(k in sample for k in ['K0', 'K1', 'R_01', 't_01']))
    batch_dict['has_planar'] = has_planar
    batch_dict['has_nonplanar'] = has_nonplanar

    return batch_dict
