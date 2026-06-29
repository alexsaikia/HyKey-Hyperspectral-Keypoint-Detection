"""Train HyKey on the HyKey Dataset.

Run with:
    python -m train.train_hykey --config configs/train_hykey.yaml
"""
import os

from models.hykey import HyKey
from train.utils import (
    load_config, create_dataloaders, setup_training_environment,
    create_checkpoint_directory, create_checkpoint_callback, create_early_stopping_callback,
    create_wandb_logger, create_trainer, load_model_weights, test_best_model,
    get_hykey_model_kwargs, create_argument_parser,
)


def main(config_path: str, max_steps: int = None):
    cfg = load_config(config_path)
    setup_training_environment(42, deterministic=cfg.get('training', {}).get('deterministic', False))

    checkpoint_dir = create_checkpoint_directory(cfg, config_path, HyKey)
    checkpoint_callback = create_checkpoint_callback(cfg, checkpoint_dir, HyKey)
    early_stop_callback = create_early_stopping_callback(cfg)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(cfg, dataset_class='spectral')

    wandb_logger = create_wandb_logger(cfg, HyKey)

    model = HyKey(**get_hykey_model_kwargs(cfg))
    model.save_hyperparameters(cfg)

    # Weight initialisation for fine-tuning. Resolution order: INIT_CHECKPOINT env var
    # (used for unattended chaining) overrides the config value. Non-existent paths are
    # treated as "no checkpoint". Loaded non-strict so extra/mismatched keys are skipped.
    init_checkpoint = os.environ.get('INIT_CHECKPOINT') or cfg.get('model', {}).get('init_checkpoint')
    if init_checkpoint and not os.path.exists(init_checkpoint):
        print(f"WARNING: init_checkpoint '{init_checkpoint}' does not exist; training from scratch.")
        init_checkpoint = None
    loaded_pretrained = bool(init_checkpoint) and load_model_weights(model, init_checkpoint, strict=False)
    if loaded_pretrained:
        print(f"Initialized weights from checkpoint: {init_checkpoint}")

    # Guardrail: epipolar loss must not be active when training from scratch unless the config
    # explicitly opts in via model.allow_epi_from_scratch (the epi-from-start ablation).
    # Normal HyKey recipe: warm up noPE checkpoint first, then fine-tune with epi.
    allow_epi_from_scratch = bool(cfg.get('model', {}).get('allow_epi_from_scratch', False))
    if getattr(model, 'w_epi', 0.0) and model.w_epi > 0:
        if loaded_pretrained:
            print(f"Epipolar loss ACTIVE (fine-tuning from checkpoint), w_epi={model.w_epi}")
        elif allow_epi_from_scratch:
            print(f"Epipolar loss ACTIVE FROM SCRATCH (allow_epi_from_scratch=True), w_epi={model.w_epi}")
        else:
            print(f"WARNING: disabling epipolar loss (w_epi={model.w_epi} -> 0.0) for from-scratch "
                  f"training. Set model.init_checkpoint or INIT_CHECKPOINT env var to fine-tune with "
                  f"epi, or set model.allow_epi_from_scratch: true for the epi-from-start ablation.")
            model.w_epi = 0.0

    wandb_logger.log_hyperparams(cfg['model'])

    trainer = create_trainer(cfg, wandb_logger, checkpoint_callback, early_stop_callback, max_steps)
    trainer.fit(model, train_dataloader, val_dataloader)

    best_model_path = checkpoint_callback.best_model_path
    test_best_model(best_model_path, HyKey, cfg, wandb_logger, test_dataloader)


if __name__ == '__main__':
    parser = create_argument_parser('Train HyKey model', "configs/train_hykey.yaml")
    args = parser.parse_args()
    main(args.config, args.max_steps)
