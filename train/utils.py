import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import yaml
import os
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from models.utils.scheduler import WarmupConstantSchedule, WarmupCosineSchedule, WarmupLinearSchedule
import functools
import math

from data_processing.spectral_dataset import SpectralDataset

# Repo root, used for repo-relative config defaults.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Default location of the precomputed fixed per-band normalization stats.
_DEFAULT_BAND_STATS_PATH = str(_REPO_ROOT / 'configs' / 'band_stats.npz')


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def discover_spectral_folders(data_root: str) -> List[str]:
    """Discover all spectral acquisition folders in the data root."""
    data_root_path = Path(data_root)
    if not data_root_path.exists():
        print(f"WARNING: Spectral data root directory does not exist: {data_root_path}")
        return []
        
    folders = sorted([
        f for f in os.listdir(data_root_path)
        if os.path.isdir(data_root_path / f) and "acquisition" in f and "HE" not in f
    ])
    return folders


def _resolve_repo_path(path: str) -> Path:
    """Resolve config paths from either the current working directory or repo root."""
    config_path = Path(path)
    if config_path.is_absolute() or config_path.exists():
        return config_path
    return _REPO_ROOT / config_path


def _load_required_split_folders(config_path: Optional[str], split_name: str) -> List[str]:
    """Load held-out scene folders and fail closed if the split config is invalid."""
    if not config_path:
        return []

    resolved_path = _resolve_repo_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{split_name} split config not found: {resolved_path}. "
            "Refusing to continue because held-out scenes could leak into training."
        )

    with open(resolved_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    folders = config.get('spectral', {}).get('acquisition_folders')
    if not isinstance(folders, list):
        raise ValueError(
            f"{split_name} split config must define spectral.acquisition_folders: {resolved_path}"
        )

    return folders


def _assert_no_overlap(split_folders: Dict[str, Set[str]]) -> None:
    """Abort when two dataset splits contain the same acquisition folder."""
    names = list(split_folders)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = split_folders[left] & split_folders[right]
            if overlap:
                overlap_list = ', '.join(sorted(overlap))
                raise ValueError(
                    f"Dataset split leakage detected between {left} and {right}: {overlap_list}"
                )


def _folders_from_dataset(dataset) -> Set[str]:
    """Return the acquisition folders represented by a built dataset."""
    if dataset is None:
        return set()
    if hasattr(dataset, 'data_pairs'):
        return {pair['folder'] for pair in dataset.data_pairs if 'folder' in pair}
    return set(getattr(dataset, 'acquisition_folders', []))


def get_spectral_dataset(cfg, split=None, dataset_cache=None):
    """Initialize SpectralDataset based on configuration."""
    start_time = time.time()
    
    if dataset_cache is not None and split in dataset_cache:
        return dataset_cache[split]

    if split is None:
        split = cfg['dataset']['split']
    
    data_root = cfg['dataset']['spectral_data_root']
    acquisition_folders = cfg['dataset'].get('spectral_acquisition_folders')
    if acquisition_folders is None:
        acquisition_folders = discover_spectral_folders(data_root)

    test_set_config_path = cfg['dataset'].get('test_set_config_path')
    test_folders = _load_required_split_folders(test_set_config_path, 'test')
    if test_folders:
        from data_processing.utils.dataset_utils import filter_test_folders
        if split == 'test':
            _, acquisition_folders = filter_test_folders(acquisition_folders, test_folders, "Spectral")
        elif split in ['train', 'val', 'all']:
            acquisition_folders, _ = filter_test_folders(acquisition_folders, test_folders, "Spectral")
    
    val_set_config_path = cfg['dataset'].get('validation_set_config_path')
    val_folders = _load_required_split_folders(val_set_config_path, 'validation')
    config_overlap = set(test_folders) & set(val_folders)
    if config_overlap:
        overlap_list = ', '.join(sorted(config_overlap))
        raise ValueError(
            f"Validation and test split configs contain the same acquisition folders: {overlap_list}"
        )
    if val_folders:
        from data_processing.utils.dataset_utils import filter_validation_folders
        if split == 'val':
            _, acquisition_folders_val = filter_validation_folders(acquisition_folders, val_folders, "Spectral")
            acquisition_folders = acquisition_folders_val
        elif split in ['train', 'test', 'all']:
            acquisition_folders, _ = filter_validation_folders(acquisition_folders, val_folders, "Spectral")
    
    bypass_internal_split = False
    if val_set_config_path and split in ['train', 'val']:
        bypass_internal_split = True
    if test_set_config_path and split == 'test':
        bypass_internal_split = True

    dataset_params = {
        'data_root': data_root,
        'acquisition_folders': acquisition_folders,
        'load_rgb': cfg['dataset'].get('load_rgb', True),
        'load_hsi': cfg['dataset'].get('load_hsi', True),
        'load_poses': cfg['dataset'].get('load_poses', False),
        'aug_strength': cfg['dataset'].get('aug_strength', 0.5),
        'registration': cfg['dataset'].get('registration'),
        'calibration_dir': cfg['dataset'].get('calibration_dir'),
        'target_space': cfg['dataset'].get('target_space', 'rgb'),
        'undistort': cfg['dataset'].get('undistort', False),
        'split': 'all' if bypass_internal_split else split,
        'test_size': None if bypass_internal_split else cfg['dataset'].get('test_size'),
        'val_size': None if bypass_internal_split else cfg['dataset'].get('val_size', 0.2),
        'random_state': cfg['dataset'].get('random_state', 42),
        'radiometric_calibration': cfg['dataset'].get('radiometric_calibration', True),
        'warp_augmentation': cfg['dataset'].get('warp_augmentation', True),
        'warp_difficulty': cfg['dataset'].get('warp_difficulty', 'paper'),
        'warp_params': cfg['dataset'].get('warp_params'),
        'photometric_augmentation': cfg['dataset'].get('photometric_augmentation', False),
        'photometric_augmentation_nonplanar': cfg['dataset'].get('photometric_augmentation_nonplanar', False),
        'always_include_nonplanar': cfg['dataset'].get('always_include_nonplanar', True),
        'max_nonplanar_offset': cfg['dataset'].get('max_nonplanar_offset', 100),
        'nonplanar_stride': cfg['dataset'].get('nonplanar_stride', 1),
        # Fixed per-band normalization stats (defaults to configs/band_stats.npz)
        'band_stats_path': cfg['dataset'].get('band_stats_path', _DEFAULT_BAND_STATS_PATH)
    }

    key = f"nonplanar_stride_{split}"
    if split is not None and key in cfg.get('dataset', {}):
        dataset_params['nonplanar_stride'] = cfg['dataset'][key]
    
    dataset_params = {k: v for k, v in dataset_params.items() if v is not None}

    dataset = SpectralDataset(**dataset_params)

    if dataset_cache is not None:
        dataset_cache[split] = dataset

    return dataset


def worker_init_fn(worker_id):
    """Seed each DataLoader worker for deterministic behaviour."""
    import torch
    import numpy as np
    import random

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_dataloader(dataset, batch_size, num_workers, persistent_workers=False, shuffle=True):
    """Create a DataLoader with custom collate function and seeded worker init."""
    generator = torch.Generator()
    generator.manual_seed(42)

    if num_workers == 0:
        persistent_workers = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=dataset.custom_collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )


def setup_training_environment(seed=42, deterministic=False):
    """Set up the training environment with seeds and (optionally) full determinism.

    Data ordering is always reproducible (seeded sampler + worker_init_fn). cuDNN
    autotuning (``benchmark``) is disabled by default: inputs have variable spatial
    sizes, so autotuning re-benchmarks on every new shape and can hurt rather than help.

    Args:
        seed (int): Global seed.
        deterministic (bool): If True, request fully deterministic GPU algorithms
            (cuDNN deterministic + ``torch.use_deterministic_algorithms``). This is
            slower and some ops fall back via ``warn_only``. If False, allow
            nondeterministic kernels for speed (default).
    """
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('high')
    # Variable input sizes make the cuDNN autotuner churn; keep it off.
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # Match main-branch determinism: deterministic cuDNN convolutions only.
        # We intentionally do NOT call torch.use_deterministic_algorithms(True): it forces
        # slow / non-existent deterministic kernels for HyKey's 3D pooling
        # (adaptive_avg_pool3d / max_pool3d backward), which caused a real per-step slowdown
        # without delivering full determinism anyway (those ops fall back via warn_only).
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False


def get_model_name_from_class(model_class):
    """Extract model name from model class."""
    if hasattr(model_class, '__name__'):
        return model_class.__name__.lower()
    else:
        return str(model_class).lower()


def get_model_name_from_config(cfg):
    """Extract model name from configuration."""
    if 'model_name' in cfg.get('model', {}):
        return cfg['model']['model_name']
    
    input_type = cfg.get('model', {}).get('input_type', '')
    if input_type:
        return f"{input_type.lower()}_model"
    
    return "model"


def generate_checkpoint_filename(cfg, model_class=None):
    """Generate automated checkpoint filename based on model and config."""
    if model_class:
        model_name = get_model_name_from_class(model_class)
    else:
        model_name = get_model_name_from_config(cfg)
    
    monitor = cfg['training'].get('monitor', 'val_loss')
    filename = f"{model_name}-{{epoch:03d}}-{{{monitor}:.4f}}"
    
    return filename


def generate_run_name(cfg, model_class=None):
    """Generate automated run name for WandB based on model and config."""
    # Prefer an explicit model_name from config so wandb run names and checkpoint dirs are unambiguous.
    model_name = cfg.get('model', {}).get('model_name')
    if not model_name:
        if model_class:
            model_name = get_model_name_from_class(model_class)
        else:
            model_name = get_model_name_from_config(cfg)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = cfg.get('dataset', {}).get('name', '')
    if dataset_name:
        run_name = f"{model_name}_{dataset_name}_{timestamp}"
    else:
        run_name = f"{model_name}_{timestamp}"
    
    return run_name


def build_resolved_config(cfg):
    """Return a deep copy of ``cfg`` with ``model:`` expanded to every hyperparameter the
    model is actually built with (defaults resolved via :func:`get_hykey_model_kwargs`).

    This makes the config saved next to a checkpoint self-contained: even if the training
    YAML omitted keys and relied on defaults, the saved copy records the exact loss weights,
    temperatures, detection settings, etc. that produced the run. The non-serialisable
    ``lr_scheduler`` partial is dropped (the scheduler is still described by the preserved
    ``training.lr_scheduler`` / ``training.warmup_steps`` keys).
    """
    import copy
    resolved = copy.deepcopy(cfg)
    model_kwargs = get_hykey_model_kwargs(cfg)
    model_kwargs.pop('lr_scheduler', None)  # functools.partial -> not YAML-serialisable
    model_section = dict(resolved.get('model', {}) or {})
    model_section.update(model_kwargs)
    resolved['model'] = model_section
    return resolved


def create_checkpoint_directory(cfg, config_path, model_class=None):
    """Create checkpoint directory and save the fully-resolved run config into it."""
    if 'model_name' in cfg.get('model', {}):
        model_name = cfg['model']['model_name']
    elif model_class:
        model_name = get_model_name_from_class(model_class)
    else:
        model_name = get_model_name_from_config(cfg)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(cfg['training']['checkpoint_dirpath'],
                                f"{timestamp}_{model_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_filename = os.path.basename(config_path)
    saved_config_path = os.path.join(checkpoint_dir, config_filename)
    # Save the resolved config (model section with all defaults filled in) so the checkpoint
    # is self-documenting. Fall back to a verbatim copy if resolution fails for any reason.
    try:
        with open(saved_config_path, 'w') as f:
            yaml.safe_dump(build_resolved_config(cfg), f, sort_keys=False)
    except Exception as e:
        print(f"Warning: could not write resolved config ({e}); copying raw config instead.")
        shutil.copy(config_path, saved_config_path)

    return checkpoint_dir


def create_checkpoint_callback(cfg, checkpoint_dir, model_class=None):
    """Create model checkpoint callback with automated filename."""
    filename = generate_checkpoint_filename(cfg, model_class)
    
    return ModelCheckpoint(
        monitor=cfg['training']['monitor'],
        dirpath=checkpoint_dir,
        filename=filename,
        save_top_k=cfg['training']['save_top_k'],
        mode=cfg['training']['mode'],
        save_weights_only=cfg['training']['save_weights_only']
    )


def create_early_stopping_callback(cfg):
    """Create early stopping callback."""
    return EarlyStopping(
        monitor=cfg['training']['monitor'],
        patience=cfg['training'].get('early_stopping_patience', 30),
        mode=cfg['training']['mode'],
        min_delta=cfg['training'].get('early_stopping_min_delta', 0.001),
        verbose=True
    )


def create_dataloaders(cfg, dataset_class='spectral', dataset_cache=None):
    """Create train, validation, and test dataloaders from SpectralDataset.

    ``dataset_class`` is accepted for call-site compatibility; only 'spectral' is
    supported in this release.
    """
    if dataset_class != 'spectral':
        raise ValueError(f"Unsupported dataset class: {dataset_class!r} (only 'spectral' is supported)")
    get_dataset_func = get_spectral_dataset

    default_val_set_path = "configs/validation_set.yaml"
    if ('dataset' in cfg) and (cfg['dataset'].get('validation_set_config_path') is None):
        if os.path.exists(default_val_set_path):
            cfg['dataset']['validation_set_config_path'] = default_val_set_path

    train_dataset = get_dataset_func(cfg, split='train', dataset_cache=dataset_cache)
    val_dataset = get_dataset_func(cfg, split='val', dataset_cache=dataset_cache)
    # Test split reuses the same test_set_config_path filtering as get_spectral_dataset.
    test_dataset = get_dataset_func(cfg, split='test', dataset_cache=dataset_cache)

    _assert_no_overlap({
        'train': _folders_from_dataset(train_dataset),
        'validation': _folders_from_dataset(val_dataset),
        'test': _folders_from_dataset(test_dataset),
    })

    # Enforce: no synthetic warping or photometric augmentation for val/test datasets.
    def disable_aug(ds_obj):
        for attr in ('photometric_augmentation', 'photometric_augmentation_nonplanar'):
            if hasattr(ds_obj, attr):
                setattr(ds_obj, attr, False)

    if val_dataset is not None:
        disable_aug(val_dataset)
    if test_dataset is not None:
        disable_aug(test_dataset)
        if hasattr(test_dataset, 'warp_augmentation'):
            test_dataset.warp_augmentation = False

    def apply_nonplanar_toggle(ds_obj, enabled: bool):
        if hasattr(ds_obj, 'always_include_nonplanar'):
            ds_obj.always_include_nonplanar = enabled

    np_val_flag = cfg.get('dataset', {}).get('enable_nonplanar_val', None)
    if np_val_flag is not None and val_dataset is not None:
        apply_nonplanar_toggle(val_dataset, bool(np_val_flag))

    np_test_flag = cfg.get('dataset', {}).get('enable_nonplanar_test', None)
    if np_test_flag is not None and test_dataset is not None:
        apply_nonplanar_toggle(test_dataset, bool(np_test_flag))

    # Create DataLoaders.
    # Only the train loader gets the full worker pool + persistent workers. Val/test get a
    # capped, NON-persistent pool: otherwise each run holds 3x num_workers persistent processes
    # (train+val+test) for the whole of training, which oversubscribes the CPU when several
    # chains are co-located (e.g. 4 chains x 32 x 3 >> 64 cores) and starves the heavier loaders.
    # Worker count does not affect sample order (the seeded sampler fixes it), so this is safe.
    train_workers = cfg['training']['num_workers']
    eval_workers = min(train_workers, cfg['training'].get('eval_num_workers', 8))

    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        num_workers=train_workers,
        persistent_workers=cfg['training'].get('persistent_workers', True),
        shuffle=True
    )

    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=1,  # Usually 1 for validation
        num_workers=eval_workers,
        persistent_workers=False,
        shuffle=False
    )

    test_dataloader = get_dataloader(
        test_dataset,
        batch_size=1,  # Usually 1 for testing
        num_workers=eval_workers,
        persistent_workers=False,
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader


def create_wandb_logger(cfg, model_class=None, project_name=None):
    """Create WandB logger with automated project and run names."""
    # Use hykey_project as default project name
    if project_name is None:
        project_name = 'hykey_project'
    
    # Generate automated run name
    run_name = generate_run_name(cfg, model_class)
    
    return WandbLogger(project=project_name, name=run_name)


def create_trainer(cfg, wandb_logger, checkpoint_callback, early_stop_callback=None, max_steps=None):
    """Create PyTorch Lightning trainer."""
    # Smoke-test mode: SMOKE_FAST_DEV_RUN=N runs N train + N val batches then exits.
    # fast_dev_run disables checkpointing, loggers, and early stopping, so this is a
    # side-effect-free verification that train/val steps work end-to-end for a model.
    smoke_n = os.environ.get('SMOKE_FAST_DEV_RUN')
    if smoke_n:
        print(f"[SMOKE] fast_dev_run={smoke_n}: running {smoke_n} train + {smoke_n} val batches, "
              f"no checkpoints/loggers/test.")
        return pl.Trainer(
            fast_dev_run=int(smoke_n),
            accelerator=cfg['training']['accelerator'],
            devices=cfg['training']['devices'],
            precision=cfg['training'].get('precision', '32-true'),
            logger=False,
        )

    callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval='step')]
    if early_stop_callback:
        callbacks.append(early_stop_callback)
    
    trainer_kwargs = {
        'max_epochs': cfg['training']['num_epochs'],
        'logger': wandb_logger,
        'accelerator': cfg['training']['accelerator'],
        'devices': cfg['training']['devices'],
        'log_every_n_steps': cfg['training']['log_every_n_steps'],
        'val_check_interval': cfg['training']['val_check_interval'],
        'callbacks': callbacks,
        'num_sanity_val_steps': cfg['training'].get('num_sanity_val_steps', 0),
        'accumulate_grad_batches': cfg['training'].get('accumulate_grad_batches', 1),
        # Precision: default to full fp32 ('32-true') for bitwise-reproducible runs.
        # Override per-config with training.precision if needed.
        'precision': cfg['training'].get('precision', '32-true'),
    }
    
    # Add gradient clipping if specified
    if 'gradient_clip_val' in cfg['training']:
        trainer_kwargs['gradient_clip_val'] = cfg['training']['gradient_clip_val']
    if 'gradient_clip_algorithm' in cfg['training']:
        trainer_kwargs['gradient_clip_algorithm'] = cfg['training']['gradient_clip_algorithm']
    
    max_images_per_epoch = cfg.get('training', {}).get('max_images_per_epoch', 10000)
    if max_images_per_epoch is not None:
        batch_size = int(cfg['training']['batch_size'])
        num_batches = max(1, math.ceil(float(max_images_per_epoch) / float(batch_size)))
        trainer_kwargs['limit_train_batches'] = num_batches
    
    # Limit validation batches if specified
    limit_val_batches = cfg.get('training', {}).get('limit_val_batches')
    if limit_val_batches is not None:
        trainer_kwargs['limit_val_batches'] = int(limit_val_batches)

    
    # If max_steps is specified, use it instead of max_epochs
    if max_steps is not None:
        trainer_kwargs['max_steps'] = max_steps
        trainer_kwargs.pop('max_epochs')
    
    return pl.Trainer(**trainer_kwargs)


def load_model_weights(model, checkpoint_path, strict=False):
    """Load model weights from checkpoint."""
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
        
        if not strict:
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                    new_state_dict[k] = v
                else:
                    print(f"Skipping {k} as it doesn't match in the new model")
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        
        print("Successfully loaded previous weights")
        return True
    else:
        print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")
        return False


def test_best_model(best_model_path, model_class, cfg, wandb_logger, test_dataloader):
    """Test the best model from checkpoint."""
    if best_model_path:
        print(f"Testing best model from: {best_model_path}")
        
        if model_class.__name__ != 'HyKey':
            raise ValueError(f"Unknown model class: {model_class.__name__}")
        model_kwargs = get_hykey_model_kwargs(cfg)

        best_model = model_class(**model_kwargs)
        checkpoint = torch.load(best_model_path, map_location=lambda storage, loc: storage, weights_only=False)
        best_model.load_state_dict(checkpoint['state_dict'])
        best_trainer = pl.Trainer(
            logger=wandb_logger,
            accelerator=cfg['training']['accelerator'],
            devices=cfg['training']['devices'],
        )
        best_trainer.test(best_model, test_dataloader)
        return True
    else:
        print("Best model path not found.")
        return False


def create_argument_parser(description, default_config):
    """Create argument parser for training scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to config file')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum number of training steps (overrides max_epochs)')
    return parser


def get_hykey_model_kwargs(cfg):
    """Extract HyKey model parameters from config."""
    training_cfg = cfg.get('training', {})
    warmup_steps = training_cfg.get('warmup_steps', 1000)
    lr_scheduler_name = str(training_cfg.get('lr_scheduler', 'constant')).lower()
    if lr_scheduler_name in ('cosine', 'warmup_cosine'):
        lr_scheduler = functools.partial(
            WarmupCosineSchedule,
            warmup_steps=warmup_steps,
            t_total=training_cfg.get('lr_total_steps', 20000),
            cycles=training_cfg.get('lr_cycles', 0.5),
        )
    elif lr_scheduler_name in ('linear', 'warmup_linear'):
        lr_scheduler = functools.partial(
            WarmupLinearSchedule,
            warmup_steps=warmup_steps,
            t_total=training_cfg.get('lr_total_steps', 20000),
        )
    else:
        lr_scheduler = functools.partial(WarmupConstantSchedule, warmup_steps=warmup_steps)

    return {
        'input_channels': cfg['model']['input_channels'],
        # Training-only; 
        'learning_rate': float(cfg['model'].get('learning_rate', 3e-4)),
        'debug_dir': cfg['model'].get('debug_dir', 'debug'),
        'use_identity_homography': cfg['model'].get('use_identity_homography', False),
        'im_size': cfg['model'].get('im_size', 512),
        # Network parameters
        'c1': cfg['model'].get('c1', 32),
        'c2': cfg['model'].get('c2', 64),
        'c3': cfg['model'].get('c3', 128),
        'dim': cfg['model'].get('dim', 128),
        # Detection parameters
        'radius': cfg['model'].get('radius', 2),
        'top_k': cfg['model'].get('top_k', 400),
        'scores_th': cfg['model'].get('scores_th', 0.0),
        'n_limit': cfg['model'].get('n_limit', 0),
        'scores_th_eval': cfg['model'].get('scores_th_eval', 0.2),
        'n_limit_eval': cfg['model'].get('n_limit_eval', 5000),
        # Loss function weights (ALIKE-style losses only)
        'w_pk': cfg['model'].get('w_pk', 0.5),
        'w_rp': cfg['model'].get('w_rp', 1.0),
        'w_sp': cfg['model'].get('w_sp', 1.0),
        'w_ds': cfg['model'].get('w_ds', 5.0),
        'w_epi': cfg['model'].get('w_epi', 0.1),
        'num_blocks': cfg['model'].get('num_blocks', 2),
        # Ground truth thresholds
        'train_gt_th': cfg['model'].get('train_gt_th', 5),
        'eval_gt_th': cfg['model'].get('eval_gt_th', 3),
        # Spectral encoder spatial max-pooling (kept True for all HyKey models)
        'use_max_pooling': cfg['model'].get('use_max_pooling', True),
        # Input validity mask bounds
        'mask_min_avg': cfg['model'].get('mask_min_avg', 0.01),
        'mask_max_avg': cfg['model'].get('mask_max_avg', 0.95),
        # Loss function parameters
        'sc_th': cfg['model'].get('sc_th', 0.1),
        'temp_det': cfg['model'].get('temp_det', 0.1),
        'temp_des': cfg['model'].get('temp_des', 0.1),
        'temp_rel': cfg['model'].get('temp_rel', 0.1),
        'norm': cfg['model'].get('norm', 1),
        # Debug parameters
        'debug_losses': cfg['model'].get('debug_losses', False),
        # Learning rate scheduler
        'lr_scheduler': lr_scheduler,
        # Enable/disable planar and nonplanar components
        'enable_planar': cfg['model'].get('enable_planar', True),
        'enable_nonplanar': cfg['model'].get('enable_nonplanar', True),
    }
