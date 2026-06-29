import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from models.utils.vis import log_image_and_score
from losses.alike_losses import SoftPeakyLoss as PeakyLoss, HuberReprojectionLocLoss as ReprojectionLocLoss
from models.utils.utils import warp, compute_keypoints_distance, mutual_argmin, EmptyTensorError
from models.utils.utils import *
from losses.epi_regularizer import epi_soft_regularizer
from models.utils.scheduler import WarmupConstantSchedule, WarmupCosineSchedule
from evaluate.eval_class import Evaluator, EvaluationConfig
from evaluate.eval_lightning_mixin import EvalLightningMixin
import functools
import wandb
from models.hykey_model import HyKeyModel
from models.compute_correspondence import CorrespondenceComputer
from models.utils.img_coords import norm_to_px


class HyKey(EvalLightningMixin, HyKeyModel):
    def __init__(self, input_channels=16, learning_rate=1e-3, debug_dir="debug", 
                 use_identity_homography=False, im_size=512,
                 c1=32, c2=64, c3=128, dim=128,
                 radius=2, top_k=400, scores_th=0, n_limit=5000, scores_th_eval=0.2, n_limit_eval=5000,
                 # Loss function weights
                w_pk=0.5, w_rp=1.0, w_sp=1.0, w_ds=5.0,
                w_epi = 0.1,
                num_blocks=2,
                 train_gt_th=5, eval_gt_th=3,
                 sc_th=0.1, temp_det=0.1, temp_des=0.1, temp_rel=1.0, norm=1,
                 debug_losses=False,
                 train_sparse: bool = True,
                 lr_scheduler=functools.partial(WarmupConstantSchedule, warmup_steps=500),
                 enable_planar: bool = True,
                 enable_nonplanar: bool = True,
                 # Nonplanar matching gates
                 np_min_sim: float = 0.1,            # absolute descriptor similarity gate
                 np_ratio: float = 1.2,              # Lowe-style ratio (sim_top1 / sim_top2)
                 np_sampson_px: float = 5.0,         # Sampson residual threshold in pixels
                 # Masking parameters
                 mask_min_avg=0.01, mask_max_avg=0.95,
                 use_max_pooling: bool = True,
                 enable_nonplanar_after_epoch: int = 0):
        
        super().__init__(
            input_channels=input_channels,
            learning_rate=learning_rate,
            debug_dir=debug_dir,
            use_identity_homography=use_identity_homography,
            im_size=im_size,
            c1=c1, c2=c2, c3=c3, dim=dim,
            radius=radius, top_k=top_k, scores_th=scores_th, n_limit=n_limit,
            temp_det=temp_det,
            num_blocks=num_blocks,
            mask_min_avg=mask_min_avg, mask_max_avg=mask_max_avg,
            use_max_pooling=use_max_pooling,
        )
        self.save_hyperparameters()
        self.debug = False
        self.debug_losses = debug_losses
        self.train_sparse = train_sparse
        self.sc_th = sc_th
        self.scores_th_eval = scores_th_eval
        self.n_limit_eval = n_limit_eval
        self.temp_des = temp_des
        self.temp_rel = temp_rel
        self.norm = norm
        self.lr_scheduler = lr_scheduler
        self.np_min_sim = float(np_min_sim)
        self.np_ratio = float(np_ratio)
        self.np_sampson_px = float(np_sampson_px)
        self.w_pk = w_pk
        self.w_rp = w_rp
        self.w_sp = w_sp
        self.w_ds = w_ds
        self.w_epi = w_epi
        self.enable_planar = enable_planar
        self.enable_nonplanar = enable_nonplanar
        
        self.train_gt_th = train_gt_th
        self.eval_gt_th = eval_gt_th
        
        self.init_loss_functions()
        
        eval_config = EvaluationConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            thresholds=[1, 3, 5]
        )
        self.unified_evaluator = Evaluator(eval_config)
        self.individual_results = []
        self.correspondence = CorrespondenceComputer(self)

    def init_loss_functions(self):
        if self.w_pk > 0:
            self.PeakyLoss = PeakyLoss(scores_th=self.sc_th)
        if self.w_rp > 0:
            self.ReprojectionLocLoss = ReprojectionLocLoss(norm=self.norm, scores_th=self.sc_th)
        if self.w_sp > 0:
            from losses.alike_losses import SparseScoreMapReliabilityLoss, ScoreMapReliabilityLoss as DenseScoreMapReliabilityLoss
            self.ScoreMapReliabilityLoss = SparseScoreMapReliabilityLoss(temperature=self.temp_rel) if self.train_sparse else DenseScoreMapReliabilityLoss(temperature=self.temp_rel)
        if self.w_ds > 0:
            from losses.alike_losses import SparseDescReprojectionLoss, DescReprojectionLoss as DenseDescReprojectionLoss
            self.DescReprojectionLoss = SparseDescReprojectionLoss(temperature=self.temp_des) if self.train_sparse else DenseDescReprojectionLoss(temperature=self.temp_des)
        # Always initialize even if w_epi=0, since it may be enabled later via callback.
        self.EpiRegularizer = epi_soft_regularizer
    
    def training_step(self, batch, batch_idx):
        # Prefer HSI path if present
        if 'hsi' in batch:
            x0 = batch['hsi']
            views = []
            pair_types = []
            if self.enable_planar and 'warped_hsi' in batch:
                views.append(batch['warped_hsi'])
                pair_types.append('planar')
            if self.enable_nonplanar and 'hsi_np' in batch:
                views.append(batch['hsi_np'])
                pair_types.append('nonplanar')
            if len(views) == 0:
                raise ValueError("Neither planar nor nonplanar views available in batch")
            return self.compute_loss_multi(x0, views, pair_types, batch, 'train', batch_idx)
        elif 'rgb' in batch:
            x0 = batch['rgb']
            views = []
            pair_types = []
            if self.enable_planar and 'warped_rgb' in batch:
                views.append(batch['warped_rgb'])
                pair_types.append('planar')
            if len(views) == 0:
                raise ValueError("No valid RGB view in batch")
            return self.compute_loss_multi(x0, views, pair_types, batch, 'train', batch_idx)
        else:
            raise ValueError("No valid input tensor found in batch")
    
    def validation_step(self, batch, batch_idx):
        if 'hsi' in batch:
            x0 = batch['hsi']
            views = []
            pair_types = []
            if self.enable_planar and 'warped_hsi' in batch:
                views.append(batch['warped_hsi'])
                pair_types.append('planar')
            if self.enable_nonplanar and 'hsi_np' in batch:
                views.append(batch['hsi_np'])
                pair_types.append('nonplanar')
            if len(views) == 0:
                return torch.tensor(0.0, device=x0.device)
            return self.compute_loss_multi(x0, views, pair_types, batch, 'val', batch_idx)
        elif 'rgb' in batch:
            x0 = batch['rgb']
            views = []
            pair_types = []
            if self.enable_planar and 'warped_rgb' in batch:
                views.append(batch['warped_rgb'])
                pair_types.append('planar')
            if len(views) == 0:
                return torch.tensor(0.0, device=x0.device)
            return self.compute_loss_multi(x0, views, pair_types, batch, 'val', batch_idx)
        else:
            raise ValueError("No valid input tensor found in batch")
    
    def test_step(self, batch, batch_idx):
        if 'hsi' in batch:
            x0 = batch['hsi']
            views = []
            pair_types = []
            if self.enable_planar and 'warped_hsi' in batch:
                views.append(batch['warped_hsi'])
                pair_types.append('planar')
            if self.enable_nonplanar and 'hsi_np' in batch:
                views.append(batch['hsi_np'])
                pair_types.append('nonplanar')
            if len(views) == 0:
                return torch.tensor(0.0, device=x0.device)
            return self.compute_loss_multi(x0, views, pair_types, batch, 'test', batch_idx)
        elif 'rgb' in batch:
            x0 = batch['rgb']
            views = []
            pair_types = []
            if self.enable_planar and 'warped_rgb' in batch:
                views.append(batch['warped_rgb'])
                pair_types.append('planar')
            if len(views) == 0:
                return torch.tensor(0.0, device=x0.device)
            return self.compute_loss_multi(x0, views, pair_types, batch, 'test', batch_idx)
        else:
            raise ValueError("No valid input tensor found in batch")
    
    def compute_loss_multi(self, x0, views, pair_types, batch, mode, batch_idx):
        """Compute losses using a single forward for x0 and one per provided view.
        This avoids repeated forward passes while still computing per-pair correspondences/losses.
        """
        pred0 = self(x0)
        preds_views = [self(v) for v in views]

        total_loss = torch.tensor(0.0, device=x0.device)
        component_sums = {}

        for pred1, ptype, v in zip(preds_views, pair_types, views):
            pred0_used = pred0
            batch_used = batch
            batch_used['pair_type'] = [ptype] * x0.shape[0]
            batch_used['image0'] = x0
            batch_used['image1'] = v

            if ptype == 'nonplanar':
                has_np = batch.get('has_nonplanar', None)
                if has_np is None:
                    idx_np = list(range(x0.shape[0]))
                else:
                    idx_np = [i for i, f in enumerate(has_np) if bool(f)]
                if len(idx_np) == 0 or v.shape[0] == 0:
                    continue
                sel = torch.as_tensor(idx_np, device=x0.device)
                pred0_used = {
                    'scores_map': pred0['scores_map'].index_select(0, sel),
                    'descriptor_map': pred0['descriptor_map'].index_select(0, sel)
                }
                if 'train_mask' in pred0:
                    pred0_used['train_mask'] = pred0['train_mask'].index_select(0, sel)
                batch_used = {
                    'image0': x0.index_select(0, sel),
                    'image1': v,
                    'pair_type': ['nonplanar'] * v.shape[0],
                    'orig_idx': sel
                }
                for k in ['K0', 'K1', 'R_01', 't_01']:
                    if k in batch:
                        batch_used[k] = batch[k]
                idx_list = sel.detach().cpu().tolist()
                for k in ['hsi_path', 'hsi_np_path', 'rgb_path', 'rgb_np_path']:
                    if k in batch and isinstance(batch[k], list) and len(batch[k]) > 0:
                        batch_used[k] = [batch[k][int(i)] for i in idx_list if int(i) < len(batch[k])]
                for k in ['has_nonplanar', 'has_planar']:
                    if k in batch:
                        batch_used[k] = batch[k]

            use_rand = (mode == 'train')
            if ptype == 'nonplanar':
                use_rand = False

            extra_pred = None
            correspondences, pred0_with_rand, pred1_with_rand = self.compute_correspondence(
                pred0_used, pred1, batch_used, rand=use_rand,
                extra_pred=extra_pred, extra_batch=batch_used, pair_type=ptype
            )

            if mode != 'test':
                loss_package = self.compute_losses(pred0_with_rand, pred1_with_rand, correspondences, batch)
                pair_total = 0
                if self.w_pk > 0 and 'peaky_loss' in loss_package:
                    pair_total += self.w_pk * loss_package['peaky_loss']
                if self.w_rp > 0 and 'reprojection_loss' in loss_package:
                    pair_total += self.w_rp * loss_package['reprojection_loss']
                if self.w_sp > 0 and 'score_map_rp_loss' in loss_package:
                    pair_total += self.w_sp * loss_package['score_map_rp_loss']
                if self.w_ds > 0 and 'descriptor_loss' in loss_package:
                    pair_total += self.w_ds * loss_package['descriptor_loss']
                if self.w_epi > 0 and 'epi_loss' in loss_package:
                    pair_total += self.w_epi * loss_package['epi_loss']

                for k, v in loss_package.items():
                    if k not in component_sums:
                        component_sums[k] = v.detach()
                    else:
                        component_sums[k] = component_sums[k] + v.detach()

            if batch_idx % 16 == 0:
                self.log_all_views(batch_used, f'{mode}', pred0_with_rand, pred1_with_rand, correspondences, extra_pred)

            if mode in ['val', 'test'] and ptype == 'planar' and ('H_mat' in batch_used):
                self.val_match(batch_used, pred0_with_rand, pred1_with_rand, mode)

            if mode in ['val', 'test'] and ptype == 'nonplanar':
                have_cam = all(k in batch_used for k in ['K0','K1','R_01','t_01'])
                if have_cam:
                    Bnp = pred0_with_rand['scores_map'].shape[0]
                    pred_dict_np = {
                        'kpts0': [], 'kpts1': [], 'desc0': [], 'desc1': []
                    }
                    for bi in range(Bnp):
                        # Convert normalized [-1,1] keypoints to pixel coordinates using actual input image sizes
                        # View-0 uses batch_used['hsi'] when present; otherwise fallback to scores_map size
                        if ('hsi' in batch_used) and (isinstance(batch_used['hsi'], torch.Tensor)) and (batch_used['hsi'].shape[0] > bi):
                            H0 = int(batch_used['hsi'].shape[-2]); W0 = int(batch_used['hsi'].shape[-1])
                        else:
                            H0 = int(pred0_with_rand['scores_map'].shape[-2]); W0 = int(pred0_with_rand['scores_map'].shape[-1])
                        if ('hsi_np' in batch_used) and (isinstance(batch_used['hsi_np'], torch.Tensor)) and (batch_used['hsi_np'].shape[0] > bi):
                            H1 = int(batch_used['hsi_np'].shape[-2]); W1 = int(batch_used['hsi_np'].shape[-1])
                        else:
                            H1 = int(pred1_with_rand['scores_map'].shape[-2]); W1 = int(pred1_with_rand['scores_map'].shape[-1])

                        if len(pred0_with_rand['keypoints']) > bi and pred0_with_rand['keypoints'][bi].numel() > 0:
                            k0px = (pred0_with_rand['keypoints'][bi] + 1) / 2 * pred0_with_rand['keypoints'][bi].new_tensor([W0-1, H0-1])
                        else:
                            k0px = torch.empty(0, 2, device=pred0_with_rand['scores_map'].device)
                        if len(pred1_with_rand['keypoints']) > bi and pred1_with_rand['keypoints'][bi].numel() > 0:
                            k1px = (pred1_with_rand['keypoints'][bi] + 1) / 2 * pred1_with_rand['keypoints'][bi].new_tensor([W1-1, H1-1])
                        else:
                            k1px = torch.empty(0, 2, device=pred1_with_rand['scores_map'].device)
                        d0 = pred0_with_rand['descriptors'][bi] if len(pred0_with_rand['descriptors']) > bi else torch.empty(0, pred0_with_rand['descriptor_map'].shape[1], device=pred0_with_rand['descriptor_map'].device)
                        d1 = pred1_with_rand['descriptors'][bi] if len(pred1_with_rand['descriptors']) > bi else torch.empty(0, pred1_with_rand['descriptor_map'].shape[1], device=pred1_with_rand['descriptor_map'].device)
                        pred_dict_np['kpts0'].append(k0px)
                        pred_dict_np['kpts1'].append(k1px)
                        pred_dict_np['desc0'].append(d0)
                        pred_dict_np['desc1'].append(d1)

                    cam_pack = {'K0': batch_used['K0'], 'K1': batch_used['K1'], 'R_01': batch_used['R_01'], 't_01': batch_used['t_01']}
                    # Per-batch: accumulate metrics only. Success-curve plots are written once
                    # per epoch (see on_validation_epoch_end / write_test_csvs), so no plot_dir here.
                    np_pack = self.unified_evaluator.evaluate_nonplanar_batch_individual(
                        pred_dict_np, cam_pack,
                        plot_dir=None,
                        plot_prefix='nonplanar',
                        plot_success_curves=False,
                    )
                    for mk, mv in np_pack['batch_metrics'].items():
                        self.log(f'{mode}/{mk}', mv, sync_dist=True)
                    if hasattr(self, 'individual_results'):
                        Bnp = len(np_pack['individual_results'])
                        meta_np = []
                        for bi in range(Bnp):
                            mi = {'pair_type': 'nonplanar'}
                            path0 = None
                            path1 = None
                            if 'hsi_path' in batch_used and isinstance(batch_used['hsi_path'], list) and bi < len(batch_used['hsi_path']):
                                path0 = batch_used['hsi_path'][bi]
                            if 'hsi_np_path' in batch_used and isinstance(batch_used['hsi_np_path'], list) and bi < len(batch_used['hsi_np_path']):
                                path1 = batch_used['hsi_np_path'][bi]
                            if path0 is None and 'rgb_path' in batch_used and isinstance(batch_used['rgb_path'], list) and bi < len(batch_used['rgb_path']):
                                path0 = batch_used['rgb_path'][bi]
                            if path1 is None and 'rgb_np_path' in batch_used and isinstance(batch_used['rgb_np_path'], list) and bi < len(batch_used['rgb_np_path']):
                                path1 = batch_used['rgb_np_path'][bi]
                            if path0 is not None:
                                mi['image0_path'] = path0
                            if path1 is not None:
                                mi['image1_path'] = path1
                            if 'scene' not in mi and path0 is not None:
                                mi['scene'] = os.path.basename(os.path.dirname(os.path.dirname(path0)))
                            meta_np.append(mi)
                        for res, mi in zip(np_pack['individual_results'], meta_np):
                            res.update(mi)
                            self.individual_results.append(res)
            if mode != 'test':
                total_loss = total_loss + pair_total

        if mode != 'test':
            self.log(f'{mode}/loss', total_loss, sync_dist=True)
            for k, v in component_sums.items():
                self.log(f'{mode}/{k}', v, sync_dist=True)

        if mode == 'train':
            if torch.isnan(total_loss):
                total_loss = torch.tensor(1e-6, device=x0.device, requires_grad=True)
            assert not torch.isnan(total_loss)

        return total_loss
    
    def compute_losses(self, pred0_with_rand, pred1_with_rand, correspondences, batch):
        loss_package = {}
        b, _, h, w = pred0_with_rand['scores_map'].shape
        zero_scalar = pred0_with_rand['scores_map'].new_zeros(())
        planar_corr = []
        for i in range(b):
            ci = correspondences[i]
            if ci.get('pair_type', 'planar') == 'planar':
                planar_corr.append(ci)
            else:
                planar_corr.append({'correspondence0': None, 'dist': zero_scalar})
        
        if self.w_pk > 0:
            loss_peaky0 = self.PeakyLoss(pred0_with_rand)
            loss_peaky1 = self.PeakyLoss(pred1_with_rand)
            loss_peaky = (loss_peaky0 + loss_peaky1) / 2.
            loss_package['peaky_loss'] = loss_peaky

        has_planar = self.enable_planar and any([c.get('pair_type', 'planar') == 'planar' for c in correspondences])
        if self.w_rp > 0 and has_planar:
            loss_reprojection = self.ReprojectionLocLoss(pred0_with_rand, pred1_with_rand, planar_corr)
            loss_package['reprojection_loss'] = loss_reprojection

        if self.w_sp > 0 and has_planar:
            loss_score_map_rp = self.ScoreMapReliabilityLoss(pred0_with_rand, pred1_with_rand, planar_corr)
            loss_package['score_map_rp_loss'] = loss_score_map_rp

        if self.w_ds > 0 and has_planar:
            loss_des = self.DescReprojectionLoss(pred0_with_rand, pred1_with_rand, planar_corr)
            loss_package['descriptor_loss'] = loss_des

        has_nonplanar = self.enable_nonplanar and any([c.get('pair_type', 'planar') == 'nonplanar' for c in correspondences])

        if has_nonplanar:
            try:
                d0_list, d1_list = [], []
                k0_list, k1_list = [], []
                K0_list, K1_list, R_list, t_list = [], [], [], []
                device = pred0_with_rand['scores_map'].device

                for bi, ci in enumerate(correspondences):
                    if ci.get('pair_type') != 'nonplanar':
                        continue
                    K0_i = ci.get('K0', None); K1_i = ci.get('K1', None)
                    R_i  = ci.get('R01', None); t_i  = ci.get('t01', None)
                    if any(x is None for x in [K0_i, K1_i, R_i, t_i]):
                        continue
                    if torch.is_tensor(t_i) and t_i.dim() > 1:
                        t_i = t_i.view(-1)
                    if bi < len(pred0_with_rand['descriptors']) and bi < len(pred1_with_rand['descriptors']):
                        d0 = pred0_with_rand['descriptors'][bi]
                        d1 = pred1_with_rand['descriptors'][bi]
                        if bi < len(pred0_with_rand['keypoints']) and bi < len(pred1_with_rand['keypoints']):
                            k0_norm = pred0_with_rand['keypoints'][bi]
                            k1_norm = pred1_with_rand['keypoints'][bi]
                            if k0_norm.numel() > 0 and k1_norm.numel() > 0:
                                H0 = ci.get('H0', pred0_with_rand['scores_map'].shape[-2])
                                W0 = ci.get('W0', pred0_with_rand['scores_map'].shape[-1])
                                H1 = ci.get('H1', pred1_with_rand['scores_map'].shape[-2])
                                W1 = ci.get('W1', pred1_with_rand['scores_map'].shape[-1])
                                k0_px = ((k0_norm + 1) / 2) * k0_norm.new_tensor([W0-1, H0-1])
                                k1_px = ((k1_norm + 1) / 2) * k1_norm.new_tensor([W1-1, H1-1])
                                if d0.requires_grad or pred0_with_rand['descriptor_map'].requires_grad:
                                    d0_list.append(d0)
                                    d1_list.append(d1)
                                    k0_list.append(k0_px.detach())
                                    k1_list.append(k1_px.detach())
                                    K0_list.append(K0_i)
                                    K1_list.append(K1_i)
                                    R_list.append(R_i)
                                    t_list.append(t_i)

                if len(d0_list) > 0:
                    K0_b = torch.stack([k if torch.is_tensor(k) else torch.as_tensor(k, device=device) for k in K0_list]).to(device)
                    K1_b = torch.stack([k if torch.is_tensor(k) else torch.as_tensor(k, device=device) for k in K1_list]).to(device)
                    R_b  = torch.stack([r if torch.is_tensor(r) else torch.as_tensor(r, device=device) for r in R_list]).to(device)
                    t_b  = torch.stack([t if torch.is_tensor(t) else torch.as_tensor(t, device=device) for t in t_list]).to(device)
                    # Sampson / matrix-inverse numerics require fp32; casting is differentiable.
                    with torch.autocast(device_type=device.type, enabled=False):
                        d0_list = [d.float() for d in d0_list]
                        d1_list = [d.float() for d in d1_list]
                        loss_epi = self.EpiRegularizer(
                            d0_list, d1_list,
                            k0_list, k1_list,
                            K0_b.float(), K1_b.float(), R_b.float(), t_b.float(),
                            tau_match=0.1,
                            huber_px=5.0,
                            max_kpts=512,
                            symmetric=True,
                            clip_percentile=0.90,
                            max_px=20.0,
                            trim_ratio=0.1
                        )
                else:
                    loss_epi = pred0_with_rand['scores_map'].sum() * 0.0
                loss_package['epi_loss'] = loss_epi
            except Exception as e:
                if self.debug_losses:
                    import traceback
                    traceback.print_exc()
                loss_epi = pred0_with_rand['scores_map'].sum() * 0.0
                loss_package['epi_loss'] = loss_epi
        dev = pred0_with_rand['scores_map'].device
        for k in list(loss_package.keys()):
            if torch.is_tensor(loss_package[k]) and loss_package[k].device != dev:
                loss_package[k] = loss_package[k].to(dev)
       
        
        return loss_package
    
    def sample_descriptors(self, descriptor_map, keypoints):
        """Original descriptor sampling without optimizations"""
        if isinstance(keypoints, list):
            if len(keypoints) == 0:
                return torch.empty(0, descriptor_map.shape[1], device=descriptor_map.device)
            keypoints = keypoints[0]
        if len(keypoints) == 0:
            return torch.empty(0, descriptor_map.shape[1], device=descriptor_map.device)
        
        eps = 1e-6
        keypoints = keypoints.clamp(min=-1.0 + eps, max=1.0 - eps)
        
        desc = F.grid_sample(
            descriptor_map,
            keypoints.view(1, 1, -1, 2),
            mode='bilinear',
            align_corners=True
        )[0, :, 0, :].t()
        
        desc = F.normalize(desc, p=2, dim=1)
        return desc
    
    def compute_correspondence(self, pred0, pred1, batch, rand=None,
                            extra_pred=None, extra_batch=None, pair_type=None):
        """Compute correspondences for planar and nonplanar pairs + optional 3rd view packs.

        Args:
            pred0, pred1: dicts with 'scores_map' (B,1,H,W) and 'descriptor_map' (B,D,H,W)
            batch: dict with per-pair metadata; may include
                - planar:  H_mat
                - nonplanar: K0,K1,R_01,t_01  (and optionally K2,R_02,t_02 when triplet)
            rand: add random keypoints (training augmentation). Will be disabled for nonplanar
            extra_pred: 3rd-view predictions (scores_map, descriptor_map)
            extra_batch: metadata for view-2 (e.g., H_12/H_02 or K2/R_02/t_02)
            pair_type: override type for this call ('planar' or 'nonplanar')
        Returns:
            correspondences, pred0_with_rand, pred1_with_rand
        """
        return self.correspondence.compute(
            pred0, pred1, batch, rand=rand,
            extra_pred=extra_pred, extra_batch=extra_batch, pair_type=pair_type
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def val_match(self, batch, pred0, pred1, mode):
        """Original validation matching without caching"""
        B, _, h0, w0 = batch['image0'].shape
        device = batch['image0'].device
        
        if B == 0:
            return self.unified_evaluator.get_zero_metrics()
        
        scoredispersitys0 = pred0['score_dispersity']
        scoredispersitys1 = pred1['score_dispersity']
        
        keypoints0 = pred0['keypoints'][:len(scoredispersitys0)]
        keypoints1 = pred1['keypoints'][:len(scoredispersitys1)]
        
        pred_dict = {
            'image0': batch['image0'],
            'image1': batch['image1'],
            'scores_map0': pred0['scores_map'],
            'scores_map1': pred1['scores_map'],
            'kpts0': [],
            'kpts1': [],
            'desc0': [],
            'desc1': [],
            'warp01_params': [],
            'warp10_params': []
        }
        
        for idx in range(B):
            if len(keypoints0) > idx and len(keypoints0[idx]) > 0:
                kpts0_pixel = norm_to_px(keypoints0[idx], w0, h0)
            else:
                kpts0_pixel = torch.empty(0, 2, device=device)
            
            if len(keypoints1) > idx and len(keypoints1[idx]) > 0:
                kpts1_pixel = norm_to_px(keypoints1[idx], w0, h0)
            else:
                kpts1_pixel = torch.empty(0, 2, device=device)
            
            pred_dict['kpts0'].append(kpts0_pixel)
            pred_dict['kpts1'].append(kpts1_pixel)
            
            if len(keypoints0) > idx and len(keypoints0[idx]) > 0:
                desc0 = self.sample_descriptors(pred0['descriptor_map'][idx:idx+1], [keypoints0[idx]])
            else:
                desc0 = torch.empty(0, pred0['descriptor_map'].shape[1], device=device)
            
            if len(keypoints1) > idx and len(keypoints1[idx]) > 0:
                desc1 = self.sample_descriptors(pred1['descriptor_map'][idx:idx+1], [keypoints1[idx]])
            else:
                desc1 = torch.empty(0, pred1['descriptor_map'].shape[1], device=device)
            
            pred_dict['desc0'].append(desc0)
            pred_dict['desc1'].append(desc1)
            
            warp01_params = {
                'mode': 'homo',
                'homography_matrix': batch['H_mat'][idx],
                'width': w0,
                'height': h0
            }
            warp10_params = {
                'mode': 'homo',
                'homography_matrix': torch.inverse(batch['H_mat'][idx]),
                'width': w0,
                'height': h0
            }
            pred_dict['warp01_params'].append(warp01_params)
            pred_dict['warp10_params'].append(warp10_params)
        
        try:
            device = batch['image0'].device
            self.unified_evaluator.device = device
            self.unified_evaluator.config.device = device.type
            
            if mode in ['val', 'test']:
                # Build meta with file paths and derived scene
                meta = []
                for bi in range(B):
                    mi = {'pair_type': 'planar'}
                    p0 = None
                    p1 = None
                    if 'hsi_path' in batch and isinstance(batch['hsi_path'], list) and bi < len(batch['hsi_path']):
                        p0 = batch['hsi_path'][bi]
                    if 'rgb_path' in batch and isinstance(batch['rgb_path'], list) and bi < len(batch['rgb_path']):
                        p1 = batch['rgb_path'][bi]
                    if p0 is not None:
                        mi['image0_path'] = p0
                    if p1 is not None:
                        mi['image1_path'] = p1
                    if p0 is not None:
                        mi['scene'] = os.path.basename(os.path.dirname(os.path.dirname(p0)))
                    meta.append(mi)

                eval_results_individual = self.unified_evaluator.evaluate_batch_individual(
                    pred_dict, (h0, w0), meta=meta, pair_type='planar'
                )
                individual_results = eval_results_individual['individual_results']
                for result in individual_results:
                    self.individual_results.append(result)
                return eval_results_individual['batch_metrics']
            else:
                eval_results = self.unified_evaluator.evaluate_batch(
                    pred_dict, (h0, w0)
                )
                return eval_results
        except EmptyTensorError:
            return self.unified_evaluator.get_zero_metrics()
    
    def log_all_views(self, batch, prefix, pred0_with_rand=None, pred1_with_rand=None, correspondences=None, extra_pred=None):
        """Log concatenated src|tgt and matches using already computed predictions."""
        try:
            if not hasattr(self, 'logger') or self.logger is None or not hasattr(self.logger, 'experiment'):
                return
            if pred0_with_rand is None or pred1_with_rand is None:
                return

            pair_type = None
            if 'pair_type' in batch:
                if isinstance(batch['pair_type'], (list, tuple)) and len(batch['pair_type']) > 0:
                    pair_type = batch['pair_type'][0]
                elif isinstance(batch['pair_type'], str):
                    pair_type = batch['pair_type']
            suffix = 'planar_' if pair_type == 'planar' else ('nonplanar_' if pair_type == 'nonplanar' else '')

            pred_dict = {
                'scores_map0': pred0_with_rand['scores_map'],
                'scores_map1': pred1_with_rand['scores_map'],
                'kpts0': pred0_with_rand['keypoints'],
                'kpts1': pred1_with_rand['keypoints'],
                'desc0': pred0_with_rand['descriptors'],
                'desc1': pred1_with_rand['descriptors']
            }
            batch_for_viz = batch
            if pair_type == 'nonplanar':
                batch_for_viz = dict(batch)
                for k in ['H_mat', 'homography', 'warp01_params', 'warp10_params', 'H']:
                    batch_for_viz.pop(k, None)
            log_image_and_score(self.logger, batch_for_viz, pred_dict, f'{prefix}/{suffix}', 500)

            if correspondences is not None and any('np3info' in ci or 'h_cycle' in ci for ci in correspondences):
                from models.utils.vis import log_triplet_viz
                viz_pack = {'extra_pred': extra_pred, 'correspondences': correspondences}
                log_triplet_viz(self.logger, batch_for_viz, pred_dict, viz_pack, f'{prefix}/{suffix}triplet/')
        except Exception:
            pass
    
    # Validation/test epoch hooks, CSV writing, and pose-curve plotting are
    # provided by EvalLightningMixin so that HyKey and all baselines share the
    # exact same procedure.
