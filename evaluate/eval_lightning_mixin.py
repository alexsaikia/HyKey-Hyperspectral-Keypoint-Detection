"""Shared evaluation/test machinery for all Lightning detector models.

This mixin centralises the validation/test procedure so that the HyKey model
(`models.hykey.HyKey`) and every baseline (`models_baselines.SPN.SPN`,
`models_baselines.ALIKE.ALIKEModel`, ...) run through the *exact same* steps:

  * planar evaluation results carry scene/path/pair_type metadata,
  * nonplanar (pose / epipolar) evaluation is run whenever a camera pack is
    present in the batch,
  * the validation/test epoch ends aggregate the same key metrics, write the
    same per-pair / per-scene / overall CSVs, and plot the same pose-success
    curves.

Models that mix this in are expected to provide:
  * ``self.unified_evaluator`` : an :class:`evaluate.eval_class.Evaluator`,
  * ``self.individual_results`` : a list that the steps append per-sample dicts to.

The nonplanar branch is opt-in per step: a model calls :meth:`run_nonplanar_eval`
with the keypoints/descriptors it produced for a real (nonplanar) image pair.
"""

import os
import csv
import torch


# Key metrics aggregated into ``{mode}/metrics/mean`` (used for checkpoint
# monitoring). Includes planar repeatability/matching plus nonplanar pose/epi
# metrics; missing metrics for a given run are simply skipped.
KEY_METRICS = [
    'repeatability@3', 'matching_score@3', 'MMA@3', 'MHA@3',
    'pose_inlier_ratio', 'pose_S@10', 'epi_AUC@5px',
]


class EvalLightningMixin:
    """Shared validation/test hooks and evaluation helpers."""

    # ------------------------------------------------------------------
    # Checkpoint directory resolution (where CSVs / plots get written)
    # ------------------------------------------------------------------
    def resolve_checkpoint_dir(self):
        import os
        # Prefer explicit checkpoint_dir from trainer callbacks if available
        if hasattr(self, 'trainer') and self.trainer is not None:
            if hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback is not None:
                cp = self.trainer.checkpoint_callback
                if hasattr(cp, 'dirpath') and cp.dirpath:
                    return cp.dirpath
            if hasattr(self.trainer, 'log_dir') and self.trainer.log_dir:
                return self.trainer.log_dir
            if hasattr(self.trainer, 'logger') and self.trainer.logger is not None:
                lg = self.trainer.logger
                if hasattr(lg, 'log_dir') and lg.log_dir:
                    return lg.log_dir
                if hasattr(lg, 'experiment') and hasattr(lg.experiment, 'dir') and lg.experiment.dir:
                    return lg.experiment.dir
                if hasattr(lg, 'save_dir') and hasattr(lg, 'name') and hasattr(lg, 'version'):
                    return os.path.join(lg.save_dir, lg.name, f'version_{lg.version}')
        if hasattr(self, 'logger') and self.logger is not None:
            if hasattr(self.logger, 'log_dir') and self.logger.log_dir:
                return self.logger.log_dir
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'dir') and self.logger.experiment.dir:
                return self.logger.experiment.dir
            if hasattr(self.logger, 'save_dir') and hasattr(self.logger, 'name') and hasattr(self.logger, 'version'):
                return os.path.join(self.logger.save_dir, self.logger.name, f'version_{self.logger.version}')
        return os.path.join(os.getcwd(), 'lightning_logs')

    # ------------------------------------------------------------------
    # Results directory resolution (where eval CSVs / plots get written)
    # ------------------------------------------------------------------
    def resolve_results_dir(self):
        """Directory for eval artifacts (per-pair/scene/overall CSVs + success-curve plots).

        Kept SEPARATE from the checkpoint dir so ``checkpoints/`` stays weights-only.
        Resolution order: an explicit ``self.results_dir`` attribute, then the
        ``HYKEY_RESULTS_DIR`` env var, otherwise ``<cwd>/results/<run-name>`` where
        ``<run-name>`` mirrors the checkpoint dir's basename so artifacts stay grouped
        per run.
        """
        explicit = getattr(self, 'results_dir', None) or os.environ.get('HYKEY_RESULTS_DIR')
        if explicit:
            return str(explicit)
        ckpt_dir = self.resolve_checkpoint_dir()
        run_name = os.path.basename(os.path.normpath(ckpt_dir)) or 'run'
        return os.path.join(os.getcwd(), 'results', run_name)

    # ------------------------------------------------------------------
    # Nonplanar pose success-curve plotting
    # ------------------------------------------------------------------
    def collect_nonplanar_pose_err_deg(self, rows):
        import numpy as np

        def scalar(v):
            if isinstance(v, (int, float)):
                return float(v)
            if torch.is_tensor(v) and v.numel() == 1:
                return float(v.item())
            return None

        pose_errs = []
        for r in rows:
            # _pose_err_deg already folds translation to [0, 90] (the canonical pose error)
            e = scalar(r.get('_pose_err_deg', None))
            if e is None:
                Rv = scalar(r.get('pose_R_deg', None))
                Tv = scalar(r.get('pose_t_deg', None))
                if Rv is not None and Tv is not None:
                    e = max(Rv, min(Tv, 180.0 - Tv))
            if e is not None and np.isfinite(e):
                pose_errs.append(e)
        return pose_errs

    def write_nonplanar_pose_success_curve(self, rows, out_path, title):
        try:
            from models.utils.metrics import success_curve as _success_curve
            from models.utils.plotting import save_success_curve_plot as _save_curve
            pose_errs = self.collect_nonplanar_pose_err_deg(rows)
            if len(pose_errs) == 0:
                return
            device = next(self.parameters()).device
            E = torch.tensor(pose_errs, dtype=torch.float32, device=device)
            taus, suc = _success_curve(E, 20.0)
            _save_curve(taus, suc, title, xlabel='Threshold (deg)', ylabel='Success rate', out_path=out_path)
        except Exception as e:
            print(f"[Nonplanar pose curve] failed ({out_path}): {e}")

    # ------------------------------------------------------------------
    # Metadata builders (scene / image paths) for individual results
    # ------------------------------------------------------------------
    @staticmethod
    def scene_from_path(path):
        """Derive scene name as the parent directory of the HSI/RGB folder."""
        try:
            return os.path.basename(os.path.dirname(os.path.dirname(path)))
        except Exception:
            return None

    def build_planar_meta(self, batch, B):
        """Build per-sample meta (paths/scene/pair_type) for planar evaluation."""
        meta = []
        for bi in range(B):
            mi = {'pair_type': 'planar'}
            p0 = None
            p1 = None
            if 'hsi_path' in batch and isinstance(batch['hsi_path'], list) and bi < len(batch['hsi_path']):
                p0 = batch['hsi_path'][bi]
            if 'rgb_path' in batch and isinstance(batch['rgb_path'], list) and bi < len(batch['rgb_path']):
                p1 = batch['rgb_path'][bi]
            # Fall back to RGB path for image0 when HSI path is absent (RGB models)
            if p0 is None and p1 is not None:
                p0 = p1
            if p0 is not None:
                mi['image0_path'] = p0
            if p1 is not None:
                mi['image1_path'] = p1
            scene = self.scene_from_path(p0) if p0 is not None else None
            if scene is not None:
                mi['scene'] = scene
            meta.append(mi)
        return meta

    def build_nonplanar_meta(self, batch, idx_np):
        """Build per-sample meta for the nonplanar subset given original indices."""
        meta_np = []
        for bi in idx_np:
            mi = {'pair_type': 'nonplanar'}
            path0 = None
            path1 = None
            if 'hsi_path' in batch and isinstance(batch['hsi_path'], list) and bi < len(batch['hsi_path']):
                path0 = batch['hsi_path'][bi]
            if 'hsi_np_path' in batch and isinstance(batch['hsi_np_path'], list) and bi < len(batch['hsi_np_path']):
                path1 = batch['hsi_np_path'][bi]
            if path0 is None and 'rgb_path' in batch and isinstance(batch['rgb_path'], list) and bi < len(batch['rgb_path']):
                path0 = batch['rgb_path'][bi]
            if path1 is None and 'rgb_np_path' in batch and isinstance(batch['rgb_np_path'], list) and bi < len(batch['rgb_np_path']):
                path1 = batch['rgb_np_path'][bi]
            if path0 is not None:
                mi['image0_path'] = path0
            if path1 is not None:
                mi['image1_path'] = path1
            scene = self.scene_from_path(path0) if path0 is not None else None
            if scene is not None:
                mi['scene'] = scene
            meta_np.append(mi)
        return meta_np

    # ------------------------------------------------------------------
    # Nonplanar (pose / epipolar) evaluation runner
    # ------------------------------------------------------------------
    def run_nonplanar_eval(self, kpts0_px, desc0, kpts1_px, desc1, cam_pack, meta, mode):
        """Evaluate a real (nonplanar) image pair and store per-sample results.

        Args:
            kpts0_px, kpts1_px: lists (length B_np) of (N,2) pixel-space keypoints.
            desc0, desc1:       lists (length B_np) of (N,D) descriptors.
            cam_pack:           dict with 'K0','K1','R_01','t_01' (B_np stacked).
            meta:               list (length B_np) of per-sample meta dicts.
            mode:               'val' or 'test'.

        Returns the evaluator's batch metrics, or None on failure / empty input.
        """
        try:
            B_np = len(kpts0_px)
            if B_np == 0:
                return None
            have_cam = all(k in cam_pack for k in ['K0', 'K1', 'R_01', 't_01'])
            if not have_cam:
                return None

            pred_dict_np = {'kpts0': kpts0_px, 'kpts1': kpts1_px, 'desc0': desc0, 'desc1': desc1}

            # Keep the evaluator on the data device
            dev = kpts0_px[0].device
            self.unified_evaluator.device = dev
            self.unified_evaluator.config.device = dev.type

            np_pack = self.unified_evaluator.evaluate_nonplanar_batch_individual(
                pred_dict_np, cam_pack,
                plot_dir=None,
                plot_prefix='nonplanar',
                plot_success_curves=False,
            )
            for mk, mv in np_pack['batch_metrics'].items():
                self.log(f'{mode}/{mk}', mv, sync_dist=True)

            if hasattr(self, 'individual_results'):
                results = np_pack['individual_results']
                for j, res in enumerate(results):
                    if meta is not None and j < len(meta) and isinstance(meta[j], dict):
                        res.update(meta[j])
                    else:
                        res.setdefault('pair_type', 'nonplanar')
                    self.individual_results.append(res)
            return np_pack['batch_metrics']
        except Exception as e:
            print(f"Warning: nonplanar evaluation failed: {e}")
            return None

    @staticmethod
    def nonplanar_indices(batch, base_batch_size):
        """Indices (into the original batch) of samples that have a nonplanar pair."""
        has_np = batch.get('has_nonplanar', None)
        if has_np is None:
            return list(range(base_batch_size))
        return [i for i, f in enumerate(has_np) if bool(f)]

    # ------------------------------------------------------------------
    # Validation epoch hooks
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self):
        self.individual_results = []

    def on_validation_epoch_end(self):
        device = next(self.parameters()).device
        if hasattr(self, 'individual_results') and len(self.individual_results) > 0:
            self.aggregate_and_log(self.individual_results, 'val', device)
            # Aggregated nonplanar pose success curve over the full validation epoch
            try:
                nonplanar_rows = [r for r in self.individual_results if r.get('pair_type') == 'nonplanar']
                if nonplanar_rows:
                    results_dir = self.resolve_results_dir()
                    os.makedirs(results_dir, exist_ok=True)
                    self.write_nonplanar_pose_success_curve(
                        nonplanar_rows,
                        os.path.join(results_dir, 'nonplanar_pose_success.png'),
                        "Pose success curve (nonplanar, val)",
                    )
            except Exception as e:
                print(f"[Val nonplanar pose curve] failed: {e}")
        else:
            print(f"\n{type(self).__name__} Validation Epoch End: No metrics collected!")
            self.log('val/metrics/mean', torch.tensor(0.0, device=device), sync_dist=True)

    # ------------------------------------------------------------------
    # Test epoch hooks
    # ------------------------------------------------------------------
    def on_test_epoch_start(self):
        self.individual_results = []
        results_dir = self.resolve_results_dir()
        print(f"[TEST] CSVs will be written under: {results_dir}")
        print(f"[TEST] Success-curve plots will be written under: {results_dir}")
        print(f"[TEST] Planar per-pair CSV:      {os.path.join(results_dir, 'planar_per_pair.csv')}")
        print(f"[TEST] Nonplanar per-pair CSV:   {os.path.join(results_dir, 'nonplanar_per_pair.csv')}")
        print(f"[TEST] Planar per-scene CSV:     {os.path.join(results_dir, 'planar_per_scene.csv')}")
        print(f"[TEST] Nonplanar per-scene CSV:  {os.path.join(results_dir, 'nonplanar_per_scene.csv')}")
        print(f"[TEST] Planar overall CSV:       {os.path.join(results_dir, 'planar_overall.csv')}")
        print(f"[TEST] Nonplanar overall CSV:    {os.path.join(results_dir, 'nonplanar_overall.csv')}")
        print(f"[TEST] Combined overall CSV:     {os.path.join(results_dir, 'overall_testset.csv')}")

    def on_test_epoch_end(self):
        device = next(self.parameters()).device
        if hasattr(self, 'individual_results') and len(self.individual_results) > 0:
            n_metrics = self.aggregate_and_log(self.individual_results, 'test', device)
            print(f"\n{type(self).__name__} Test Epoch End:")
            print(f"  Number of samples: {len(self.individual_results)}")
            print(f"  Metrics computed: {n_metrics}")
            self.write_test_csvs(self.individual_results)
        else:
            print(f"\n{type(self).__name__} Test Epoch End: No metrics collected!")

    # ------------------------------------------------------------------
    # Aggregation + CSV writing (shared by val/test)
    # ------------------------------------------------------------------
    def aggregate_and_log(self, results, mode, device):
        """Log per-metric mean/std and the combined key-metric mean. Returns key count."""
        def log_key_metric_mean(rows, prefix):
            key_means = []
            for metric_name in KEY_METRICS:
                metric_values = []
                for result in rows:
                    v = result.get(metric_name, None)
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        metric_values.append(v.to(device))
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        metric_values.append(torch.tensor(float(v), device=device))
                if len(metric_values) > 0:
                    key_means.append(torch.stack(metric_values).mean())
            if len(key_means) > 0:
                self.log(f'{mode}/{prefix}/metrics/mean', torch.stack(key_means).mean(), sync_dist=True)

        key_union = set()
        for res in results:
            key_union.update(res.keys())
        for metric_name in sorted(list(key_union)):
            sample_values = []
            for result in results:
                v = result.get(metric_name, None)
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1 and torch.isfinite(v).all():
                        sample_values.append(v.to(device))
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    sample_values.append(torch.tensor(float(v), device=device))
            if len(sample_values) > 0:
                metric_tensor = torch.stack(sample_values)
                metric_mean = metric_tensor.mean()
                metric_std = metric_tensor.std() if metric_tensor.numel() > 1 else torch.tensor(0.0, device=device)
                self.log(f'{mode}/{metric_name}_mean', metric_mean, sync_dist=True)
                self.log(f'{mode}/{metric_name}_std', metric_std, sync_dist=True)

        key_means = []
        for metric_name in KEY_METRICS:
            metric_values = []
            for result in results:
                v = result.get(metric_name, None)
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    metric_values.append(v.to(device))
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    metric_values.append(torch.tensor(float(v), device=device))
            if len(metric_values) > 0:
                key_means.append(torch.stack(metric_values).mean())

        if len(key_means) > 0:
            metrics_mean = torch.stack(key_means).mean()
            self.log(f'{mode}/metrics/mean', metrics_mean, sync_dist=True)
        elif mode == 'val':
            # Always emit the monitored metric during validation
            self.log('val/metrics/mean', torch.tensor(0.0, device=device), sync_dist=True)

        planar_rows = [r for r in results if r.get('pair_type', 'planar') == 'planar']
        nonplanar_rows = [r for r in results if r.get('pair_type') == 'nonplanar']
        if planar_rows:
            log_key_metric_mean(planar_rows, 'planar')
        if nonplanar_rows:
            log_key_metric_mean(nonplanar_rows, 'nonplanar')
        return len(key_union)

    def write_test_csvs(self, results):
        """Write per-pair / per-scene / overall CSVs split by pair type, plus pose curves."""
        try:
            ckpt_dir = self.resolve_results_dir()
            os.makedirs(ckpt_dir, exist_ok=True)

            planar_rows = []
            nonplanar_rows = []
            for row in results:
                d = {k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in row.items()}
                if d.get('pair_type', 'planar') == 'nonplanar':
                    nonplanar_rows.append(d)
                else:
                    planar_rows.append(d)

            def write_csv(path, rows):
                if not rows:
                    return
                keys = sorted({k for r in rows for k in r.keys()})
                with open(path, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    for r in rows:
                        w.writerow({k: r.get(k, '') for k in keys})

            write_csv(os.path.join(ckpt_dir, 'planar_per_pair.csv'), planar_rows)
            write_csv(os.path.join(ckpt_dir, 'nonplanar_per_pair.csv'), nonplanar_rows)

            def aggregate_by_scene(rows):
                agg = {}
                for r in rows:
                    scene = r.get('scene', 'unknown')
                    agg.setdefault(scene, []).append(r)
                out = []
                for scene, items in agg.items():
                    metrics = {}
                    for k in items[0].keys():
                        vals = []
                        for it in items:
                            v = it.get(k, None)
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                vals.append(float(v))
                        if vals:
                            metrics[f'{k}_mean'] = sum(vals) / len(vals)
                    metrics['scene'] = scene
                    out.append(metrics)
                return out

            write_csv(os.path.join(ckpt_dir, 'planar_per_scene.csv'), aggregate_by_scene(planar_rows))
            write_csv(os.path.join(ckpt_dir, 'nonplanar_per_scene.csv'), aggregate_by_scene(nonplanar_rows))

            def overall(rows):
                if not rows:
                    return []
                keys = {k for r in rows for k in r.keys()}
                out = {}
                for k in keys:
                    vals = []
                    for r in rows:
                        v = r.get(k, None)
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            vals.append(float(v))
                    if vals:
                        out[f'{k}_mean'] = sum(vals) / len(vals)
                return [out]

            write_csv(os.path.join(ckpt_dir, 'planar_overall.csv'), overall(planar_rows))
            write_csv(os.path.join(ckpt_dir, 'nonplanar_overall.csv'), overall(nonplanar_rows))
            write_csv(os.path.join(ckpt_dir, 'overall_testset.csv'), overall(planar_rows + nonplanar_rows))

            try:
                if nonplanar_rows:
                    self.write_nonplanar_pose_success_curve(
                        nonplanar_rows,
                        os.path.join(ckpt_dir, 'nonplanar_pose_success.png'),
                        "Pose success curve (nonplanar, test)",
                    )
                    self.write_nonplanar_pose_success_curve(
                        nonplanar_rows,
                        os.path.join(ckpt_dir, 'nonplanar_pose_success_overall.png'),
                        "Pose success curve over test set (nonplanar)",
                    )
            except Exception as e:
                print(f"[Overall nonplanar pose curve] failed: {e}")
        except Exception as e:
            print(f"[CSV write] failed: {e}")
