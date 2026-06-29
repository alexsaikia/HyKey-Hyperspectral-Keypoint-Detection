import torch
import numpy as np
from typing import Optional, Tuple, Union
from skimage import transform
from models.utils.utils import sample_homography
import kornia.augmentation as KA



WARP_PRESETS = {
    "harder": dict(scaling_amplitude=0.15, perspective_amplitude_x=0.05,
                    perspective_amplitude_y=0.05, patch_ratio=0.7, max_angle=np.pi/3),
    "paper":   dict(scaling_amplitude=0.05, perspective_amplitude_x=0.01,
                    perspective_amplitude_y=0.01, patch_ratio=0.8, max_angle=np.pi/6),
}

def resolve_warp_params(spec) -> dict:
    """Resolve a warp-difficulty spec into kwargs for generate_warped_image.

    `spec` may be None (-> {}, i.e. function defaults), a preset name in WARP_PRESETS
    (e.g. 'paper' / 'harder'), or an explicit dict of override kwargs.
    """
    if spec is None:
        return {}
    if isinstance(spec, str):
        if spec not in WARP_PRESETS:
            raise ValueError(f"Unknown warp preset '{spec}'. Options: {sorted(WARP_PRESETS)}")
        return dict(WARP_PRESETS[spec])
    if isinstance(spec, dict):
        return dict(spec)
    raise TypeError(f"warp spec must be None, str, or dict; got {type(spec)}")

def generate_warped_image(
    image_tensor: torch.Tensor,
    existing_H_mat: Optional[torch.Tensor] = None,
    sample_idx: Optional[int] = None,
    scaling_amplitude: float = 0.15,
    perspective_amplitude_x: float = 0.05,
    perspective_amplitude_y: float = 0.05,
    patch_ratio: float = 0.7,
    max_angle: float = np.pi/3,
    max_retries: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a warped copy of the input image tensor using a random homography.

    Returns (warped_tensor, H_mat), both in the same device/dtype as the input.
    """
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    height, width = img_np.shape[:2]

    if existing_H_mat is None:
        valid_warp = False
        retry_count = 0

        while not valid_warp and retry_count < max_retries:
            try:
                if sample_idx is not None:
                    # Ensure seed is in [1, 2**32-1] to satisfy numpy's RNG requirement.
                    base = int(sample_idx) % (2**32 - 1)
                    deterministic_seed = int((base * 1000 + retry_count + 42) % (2**32 - 1))
                    if deterministic_seed == 0:
                        deterministic_seed = 1
                else:
                    deterministic_seed = None

                H_mat_np = np.linalg.inv(sample_homography(
                    height=height,
                    width=width,
                    perspective=True,
                    scaling=True,
                    rotation=True,
                    translation=True,
                    n_scales=5,
                    n_angles=25,
                    scaling_amplitude=scaling_amplitude,
                    perspective_amplitude_x=perspective_amplitude_x,
                    perspective_amplitude_y=perspective_amplitude_y,
                    patch_ratio=patch_ratio,
                    max_angle=max_angle,
                    allow_artifacts=False,
                    translation_overflow=0.0,
                    random_state=deterministic_seed
                ))
                
                warped_np = transform.warp(
                    img_np, 
                    np.linalg.inv(H_mat_np), 
                    output_shape=(height, width),
                    mode='constant', 
                    preserve_range=True
                )
                
                has_nan = np.isnan(warped_np).any()
                is_all_zeros = (warped_np == 0).all()
                
                if not has_nan and not is_all_zeros:
                    valid_warp = True
                    
            except Exception as e:
                print(f"Warning: Homography generation failed (attempt {retry_count + 1}): {e}")
                
            retry_count += 1
        
        if not valid_warp:
            # If all retries failed, return original image and identity homography
            print("Warning: All homography generation attempts failed, returning original image")
            H_mat_np = np.eye(3)
            warped_np = img_np.copy()
            
        # Convert homography to tensor
        H_mat = torch.from_numpy(H_mat_np).float().to(image_tensor.device)
    else:
        # Use existing homography
        H_mat = existing_H_mat
        H_mat_np = H_mat.cpu().numpy()
        
        try:
            warped_np = transform.warp(
                img_np,
                np.linalg.inv(H_mat_np),
                output_shape=(height, width),
                mode='constant',
                preserve_range=True
            )
            if np.isnan(warped_np).any():
                print("Warning: Warping with existing homography produced NaN values, using original image")
                warped_np = img_np.copy()
                
        except Exception as e:
            print(f"Warning: Warping with existing homography failed: {e}, using original image")
            warped_np = img_np.copy()
    
    warped_tensor = torch.from_numpy(warped_np).permute(2, 0, 1).to(image_tensor.device)
    warped_tensor = warped_tensor.to(image_tensor.dtype)
    if torch.isnan(warped_tensor).any():
        print("WARNING: NaN values found in warped tensor, replacing with zeros")
        warped_tensor = torch.nan_to_num(warped_tensor, nan=0.0)
    
    return warped_tensor, H_mat



_AUG_CACHE = {}

def get_aug_pipeline(device, channels, aug_strength):
    key = (device, channels, round(float(aug_strength), 4))
    if key in _AUG_CACHE:
        return _AUG_CACHE[key]

    b = 0.20 * aug_strength
    c = 0.20 * aug_strength
    blur_kernel = 3 if aug_strength < 0.5 else 5
    blur_sigma_low = max(1e-6, 0.1 * aug_strength)
    blur_sigma_high = max(blur_sigma_low + 1e-6, 0.6 * aug_strength)

    pipe = []
    # brightness/contrast always OK for HSI; no hue/sat unless RGB
    pipe.append(KA.ColorJitter(
        brightness=b if b > 0 else 0.0,
        contrast=c if c > 0 else 0.0,
        saturation=(0.2 * aug_strength if channels == 3 else 0.0),
        hue=(0.1 * aug_strength if channels == 3 else 0.0),
        p=0.9
    ))
    if blur_sigma_high > 0:
        pipe.append(KA.RandomGaussianBlur(
            kernel_size=(blur_kernel, blur_kernel),
            sigma=(blur_sigma_low, blur_sigma_high),
            p=0.35
        ))

    # Try Kornia's RandomGaussianNoise if available; otherwise we’ll add noise manually
    has_noise = hasattr(KA, "RandomGaussianNoise")
    if has_noise and aug_strength > 0:
        pipe.append(KA.RandomGaussianNoise(mean=0.0, std=0.05 * aug_strength, p=0.5))

    seq = torch.nn.Sequential(*pipe).to(device)
    _AUG_CACHE[key] = (seq, has_noise)
    return _AUG_CACHE[key]

def apply_photometric_augmentation(
    image_tensor: torch.Tensor,
    aug_strength: float = 0.5,
) -> torch.Tensor:
    """
    Apply photometric augmentation to a CHW image tensor in [0, 1].

    HSI tensors receive spectral gain/tilt before generic jitter. RGB tensors
    skip the spectral tilt and use RGB color jitter from the Kornia pipeline.
    """
    if not torch.is_tensor(image_tensor) or image_tensor.ndim != 3:
        raise TypeError("image_tensor must be a CHW torch.Tensor")

    aug_strength = float(max(0.0, min(1.0, aug_strength)))
    if aug_strength == 0.0:
        return image_tensor

    x = image_tensor.to(torch.float32)
    dev = x.device
    C = x.shape[0]
    x = x.unsqueeze(0)  # 1xCxHxW

    is_hsi = C != 3
    if is_hsi:
        gain = 1.0 + (0.20 * aug_strength) * (torch.rand((), device=dev) * 2 - 1)
        x = torch.clamp(x * gain, 0.0, 1.0)

        if C > 1:
            t = torch.linspace(-1, 1, C, device=dev)
            a1 = 0.10 * aug_strength * (torch.rand((), device=dev) * 2 - 1)
            a2 = 0.05 * aug_strength * (torch.rand((), device=dev) * 2 - 1)
            tilt = 1.0 + a1 * t + a2 * (t**2)
            x = x * tilt.view(1, C, 1, 1)
            x = torch.clamp(x, 0.0, 1.0)

    aug_seq, has_noise = get_aug_pipeline(dev, C, aug_strength)
    x = aug_seq(x)

    if not has_noise and aug_strength > 0:
        k = 0.04 * aug_strength
        read = 0.005 * aug_strength
        shot_std = torch.sqrt(torch.clamp(x, 0.0, 1.0)) * k
        read_std = torch.full_like(x, read)
        noise = torch.randn_like(x) * (shot_std + read_std)
        x = x + noise

    x = torch.clamp(x, 0.0, 1.0)

    return x.squeeze(0)