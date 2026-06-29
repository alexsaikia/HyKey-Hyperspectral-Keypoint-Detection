import torch
import numpy as np
from models.utils.utils import compute_keypoints_distance, mutual_argmin, warp
from models.utils.geometry import fundamental_from_pose as _F_from_pose, essential_from_pose as _E_from_pose, geodesic_deg as _geo_deg, angle_between_deg as _angle_deg, triangulate_linear as _triangulate
from models.utils.metrics import auc_success_curve as _auc_curve, success_curve as _success_curve
from models.utils.plotting import save_success_curve_plot as _save_curve
from evaluate.utils import mnn_match_descriptors
from evaluate.pose import estimate_pose
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
import os
import cv2  # type: ignore

class EmptyTensorError(Exception):
    pass



@dataclass
class EvaluationConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    thresholds: List[float] = None
    # New runtime knobs
    epi_thresholds_px: Tuple[float, float, float, float] = (1.0, 2.0, 3.0, 5.0)
    pose_auc_degs: Tuple[float, ...] = (1.0,3.0,5.0, 10.0, 20.0)
    epi_auc_px: Tuple[float, float] = (3.0, 5.0)
    max_se_samples_per_pair: int = 4096
    min_sim_threshold: Optional[float] = None
    lowe_ratio_threshold: Optional[float] = None
    enable_plots: bool = True
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = [1, 3, 5]

class Evaluator:
    """Unified evaluation class for keypoint detection models."""
    
    AVAILABLE_IMAGE_TYPES = [
        'keypoints0', 'keypoints1', 'matches', 'matches_overlay', 'raw_data'
    ]
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.device = torch.device(self.config.device)
    
    def evaluate(self, 
                 kpts0: Union[np.ndarray, torch.Tensor],
                 kpts1: Union[np.ndarray, torch.Tensor], 
                 desc0: Union[np.ndarray, torch.Tensor],
                 desc1: Union[np.ndarray, torch.Tensor],
                 homography: Union[np.ndarray, torch.Tensor],
                 image_shape: Tuple[int, int] = (512, 512),
                 return_images: Union[bool, List[str]] = False,
                 image0: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 image1: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, Any]:
        """Compute planar repeatability, MS, MMA, MHA for a single image pair."""
        try:
            kpts0 = self.to_tensor(kpts0)
            kpts1 = self.to_tensor(kpts1)
            desc0 = self.to_tensor(desc0)
            desc1 = self.to_tensor(desc1)
            homography = self.to_tensor(homography)

            if kpts0.ndim == 3: kpts0 = kpts0[0]
            if kpts1.ndim == 3: kpts1 = kpts1[0]
            if desc0.ndim == 3: desc0 = desc0[0]
            if desc1.ndim == 3: desc1 = desc1[0]
            if homography.ndim == 3: homography = homography[0]

            results = self.compute_all_metrics(kpts0, kpts1, desc0, desc1, homography, image_shape)

            if return_images and (image0 is not None and image1 is not None):
                vis_data = self.generate_visualizations(
                    kpts0, kpts1, desc0, desc1, image0, image1, homography, results, return_images
                )
                results['visualizations'] = vis_data

            return results

        except (RuntimeError, ValueError) as e:
            warnings.warn(f"Evaluation failed: {e}")
            return self.get_zero_metrics()

    def evaluate_batch(self, pred_dict: Dict[str, Any], image_shape: Tuple[int, int] = (512, 512)) -> Dict[str, Any]:
        """Evaluate each item in pred_dict and return aggregated batch statistics."""
        batch_size = len(pred_dict['kpts0'])
        batch_results = []

        for idx in range(batch_size):
            kpts0 = pred_dict['kpts0'][idx]
            kpts1 = pred_dict['kpts1'][idx]
            desc0 = pred_dict['desc0'][idx]
            desc1 = pred_dict['desc1'][idx]

            if len(kpts0) == 0 or len(kpts1) == 0 or len(desc0) == 0 or len(desc1) == 0:
                batch_results.append(self.get_zero_metrics())
                continue

            if 'warp01_params' in pred_dict and idx < len(pred_dict['warp01_params']):
                warp_params = pred_dict['warp01_params'][idx]
                homography = warp_params['homography_matrix']
                if 'width' in warp_params and 'height' in warp_params:
                    image_shape = (warp_params['height'], warp_params['width'])
            else:
                batch_results.append(self.get_zero_metrics())
                continue

            try:
                device = kpts0.device if hasattr(kpts0, 'device') else self.device
                self.device = device
                self.config.device = device.type if hasattr(device, 'type') else str(device)

                result = self.evaluate(
                    kpts0, kpts1, desc0, desc1, homography, image_shape
                )
                batch_results.append(result)

            except Exception as e:
                warnings.warn(f"Batch evaluation failed for item {idx}: {e}")
                batch_results.append(self.get_zero_metrics())

        return self.compute_summary_statistics(batch_results)

    def evaluate_planar_batch(self, pred_dict: Dict[str, Any], image_shape: Tuple[int, int] = (512, 512)) -> Dict[str, Any]:
        """
        Thin wrapper for planar evaluation to keep call sites explicit and future-proof.
        Currently identical to evaluate_batch.
        """
        return self.evaluate_batch(pred_dict, image_shape)

    def evaluate_nonplanar_batch_individual(self, pred_dict: Dict[str, Any], cam_pack: Dict[str, Any], meta: Optional[List[Dict[str, Any]]] = None,
                                            epi_thresholds_px=(1.0, 2.0, 3.0, 5.0),
                                            plot_dir: Optional[str] = None,
                                            plot_prefix: str = "nonplanar_individual",
                                            plot_success_curves: bool = True) -> Dict[str, Any]:
        """Like evaluate_batch but also returns per-sample results in 'individual_results'."""
        B = len(pred_dict['kpts0']) if 'kpts0' in pred_dict else 0
        individual = []
        se_collect: List[torch.Tensor] = []
        pose_err_collect: List[torch.Tensor] = []
        device = self.device
        for i in range(B):
            try:
                k0 = pred_dict['kpts0'][i]; k1 = pred_dict['kpts1'][i]
                d0 = pred_dict['desc0'][i]; d1 = pred_dict['desc1'][i]
            except Exception:
                continue
            res = {}
            for τ in epi_thresholds_px:
                res[f'N_inlier_epi@{int(τ)}'] = torch.tensor(0.0, device=device)
                res[f'MS_epi@{int(τ)}'] = torch.tensor(0.0, device=device)
            res['sampson_mean'] = torch.tensor(0.0, device=device)
            res['sampson_median'] = torch.tensor(0.0, device=device)
            res['pose_R_deg'] = torch.tensor(180.0, device=device)  # IMB: 180° for rotation when failed
            res['pose_t_deg'] = torch.tensor(90.0, device=device)   # IMB: 90° for translation when failed
            res['pose_inlier_ratio'] = torch.tensor(0.0, device=device)
            res['N_M'] = torch.tensor(0.0, device=device)
            res['pair_type'] = 'nonplanar'
            if (k0 is not None) and (k1 is not None) and (d0 is not None) and (d1 is not None) and (len(d0) > 0) and (len(d1) > 0):
                mi_np, _ = mnn_match_descriptors(d0.detach().cpu().numpy(), d1.detach().cpu().numpy())
                idx0 = torch.as_tensor(mi_np[:,0], device=d0.device, dtype=torch.long) if mi_np.size>0 else torch.empty(0, dtype=torch.long, device=d0.device)
                idx1 = torch.as_tensor(mi_np[:,1], device=d1.device, dtype=torch.long) if mi_np.size>0 else torch.empty(0, dtype=torch.long, device=d1.device)
                mk0 = k0[idx0]; mk1 = k1[idx1]
                N_M = float(len(mk0))
                res['N_M'] = torch.tensor(N_M, device=device)
                try:
                    K0 = cam_pack['K0'][i] if torch.is_tensor(cam_pack['K0']) else cam_pack['K0'][i]
                    K1 = cam_pack['K1'][i] if torch.is_tensor(cam_pack['K1']) else cam_pack['K1'][i]
                    R  = cam_pack['R_01'][i] if torch.is_tensor(cam_pack['R_01']) else cam_pack['R_01'][i]
                    t  = cam_pack['t_01'][i] if torch.is_tensor(cam_pack['t_01']) else cam_pack['t_01'][i]
                    # expose GT pose for downstream trajectory accumulation
                    res['R_gt'] = R
                    res['t_gt'] = t
                    # Compute Sampson in normalized coords using E, then scale to px
                    Egt = _E_from_pose(R, t)
                    if mk0.numel() > 0 and mk1.numel() > 0:
                        x0h = torch.cat([mk0, torch.ones(mk0.shape[0], 1, device=mk0.device, dtype=mk0.dtype)], 1)
                        x1h = torch.cat([mk1, torch.ones(mk1.shape[0], 1, device=mk1.device, dtype=mk1.dtype)], 1)
                        K0inv = torch.linalg.inv(K0); K1inv = torch.linalg.inv(K1)
                        x0n = (K0inv @ x0h.t()).t() #convert to normalized coordinates
                        x1n = (K1inv @ x1h.t()).t() #convert to normalized coordinates
                        Ex0  = (Egt  @ x0n.t()).t() # transform points from image 0 to image 1
                        Etx1 = (Egt.t() @ x1n.t()).t() # transform points from image 1 to image 0
                        num  = (x1n * (Egt @ x0n.t()).t()).sum(1).abs() # compute the numerator of the Sampson error
                        den  = Ex0[:,0].pow(2) + Ex0[:,1].pow(2) + Etx1[:,0].pow(2) + Etx1[:,1].pow(2) + 1e-12 # compute the denominator of the Sampson error
                        se_n = num / torch.sqrt(den)
                        fx0, fy0 = K0[0,0], K0[1,1]
                        fx1, fy1 = K1[0,0], K1[1,1]
                        f_rep = torch.tensor(float((fx0 + fy0 + fx1 + fy1) / 4.0), device=mk0.device, dtype=mk0.dtype)
                        se = se_n * f_rep
                    else:
                        se = mk0.new_zeros((0,))
                    if se.numel() > 0:
                        res['sampson_mean'] = se.mean()
                        res['sampson_median'] = se.median()
                        # per-sample epipolar AUCs
                        res['epi_AUC@1px'] = self.auc_success_curve_px(se, 1.0)
                        res['epi_AUC@3px'] = self.auc_success_curve_px(se, 3.0)
                        res['epi_AUC@5px'] = self.auc_success_curve_px(se, 5.0)
                        res['epi_AUC@10px'] = self.auc_success_curve_px(se, 10.0)
                        res['epi_AUC@20px'] = self.auc_success_curve_px(se, 20.0)
                        # collect (optionally downsampled) Sampson errors for scene/global AUC
                        try:
                            maxN = int(getattr(self.config, 'max_se_samples_per_pair', 4096))
                        except Exception:
                            maxN = 4096
                        if se.numel() > maxN > 0:
                            perm = torch.randperm(se.numel(), device=se.device)
                            se_sampled = se[perm[:maxN]]
                        else:
                            se_sampled = se
                        se_collect.append(se_sampled.detach())
                    else:
                        res['epi_AUC@1px'] = torch.tensor(0.0, device=device)
                        res['epi_AUC@3px'] = torch.tensor(0.0, device=device)
                        res['epi_AUC@5px'] = torch.tensor(0.0, device=device)
                        res['epi_AUC@10px'] = torch.tensor(0.0, device=device)
                        res['epi_AUC@20px'] = torch.tensor(0.0, device=device)
                    for τ in epi_thresholds_px:
                        Ninl = float((se <= τ).sum()) if se.numel() > 0 else 0.0
                        res[f'N_inlier_epi@{int(τ)}'] = torch.tensor(Ninl, device=device)
                        res[f'MS_epi@{int(τ)}'] = torch.tensor(Ninl / max(N_M, 1.0), device=device)
                except Exception:
                    pass
                # Triangulation consistency with GT pose
                try:
                    K0t = cam_pack['K0'][i] if torch.is_tensor(cam_pack['K0']) else cam_pack['K0'][i]
                    K1t = cam_pack['K1'][i] if torch.is_tensor(cam_pack['K1']) else cam_pack['K1'][i]
                    Rt  = cam_pack['R_01'][i] if torch.is_tensor(cam_pack['R_01']) else cam_pack['R_01'][i]
                    tt  = cam_pack['t_01'][i] if torch.is_tensor(cam_pack['t_01']) else cam_pack['t_01'][i]
                    X, mask = _triangulate(K0t, K1t, Rt, tt, mk0, mk1)
                    if X.shape[1] > 0 and mask.any():
                        X1 = Rt @ X + tt.view(3,1)
                        proj0 = (K0t @ X) ; proj0 = (proj0[:2] / (proj0[2:3] + 1e-12)).T
                        proj1 = (K1t @ X1); proj1 = (proj1[:2] / (proj1[2:3] + 1e-12)).T
                        d0 = torch.norm(proj0 - mk0[:X.shape[1]], dim=1)
                        d1 = torch.norm(proj1 - mk1[:X.shape[1]], dim=1)
                        rmse = torch.sqrt(0.5*(d0.pow(2)+d1.pow(2)).mean())
                        if rmse > 580:
                            rmse = 580
                        res['tri_rmse_px'] = rmse
                        res['tri_cheirality'] = mask.float().mean()
                    else:
                        res['tri_rmse_px'] = torch.tensor(0.0, device=device)
                        res['tri_cheirality'] = torch.tensor(0.0, device=device)
                except Exception:
                    res['tri_rmse_px'] = torch.tensor(0.0, device=device)
                    res['tri_cheirality'] = torch.tensor(0.0, device=device)
                # Pose from matches using robust (more robust USAC_MAGSAC)
                try:
                    if len(mk0)>=5:
                        K0v = K0.detach().cpu().numpy(); K1v = K1.detach().cpu().numpy()
                        mk0_np = mk0.detach().cpu().numpy()
                        mk1_np = mk1.detach().cpu().numpy()

                        # Check for degenerate case: if translation is too small, skip pose estimation
                        t_norm = float(torch.norm(t))
                        if t_norm < 1e-6:  # Very small baseline
                            res['pose_R_deg'] = torch.tensor(180.0, device=device)
                            res['pose_t_deg'] = torch.tensor(90.0, device=device)
                            res['pose_inlier_ratio'] = torch.tensor(0.0, device=device)
                        else:
                            # Use robust with USAC_MAGSAC
                            # thresh=1.0 in pixels (normalized internally), conf=0.99999
                            pose_result = estimate_pose(mk0_np, mk1_np, K0v, K1v, thresh=1.0, conf=0.99999)
                            
                            if pose_result is not None:
                                R_est_np, t_est_np, inlier_mask, _ = pose_result
                                R_est = torch.from_numpy(R_est_np).to(device=device, dtype=torch.float32)
                                t_est = torch.from_numpy(t_est_np).to(device=device, dtype=torch.float32)

                                # Validate the estimated pose
                                if torch.isnan(R_est).any() or torch.isinf(R_est).any() or torch.isnan(t_est).any() or torch.isinf(t_est).any():
                                    res['pose_R_deg'] = torch.tensor(180.0, device=device)
                                    res['pose_t_deg'] = torch.tensor(90.0, device=device)
                                    res['pose_inlier_ratio'] = torch.tensor(0.0, device=device)
                                else:
                                    res['pose_R_deg'] = _geo_deg(R_est, R)
                                    res['pose_t_deg'] = _angle_deg(t_est, t)
                                    # expose estimated pose for downstream trajectory accumulation
                                    res['R_est'] = R_est
                                    res['t_est'] = t_est
                                    # Count inliers from mask
                                    num_inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
                                    res['pose_inlier_ratio'] = torch.tensor(float(num_inliers)/max(len(mk0),1), device=device)
                            else:
                                # No essential matrix found
                                res['pose_R_deg'] = torch.tensor(180.0, device=device)
                                res['pose_t_deg'] = torch.tensor(90.0, device=device)
                                res['pose_inlier_ratio'] = torch.tensor(0.0, device=device)

                        # Per-sample pose aggregates (only compute if pose estimation succeeded)
                        if res['pose_R_deg'] < 180.0 and res['pose_t_deg'] < 180.0:
                            res['pose_avg_deg'] = 0.5*(res['pose_R_deg'] + res['pose_t_deg'])
                            # Translation direction is sign-ambiguous -> fold its angle into
                            # [0, 90] (== arccos(|dot|)). This is the single pose metric we use.
                            pose_err = torch.max(res['pose_R_deg'], torch.minimum(res['pose_t_deg'], 180.0 - res['pose_t_deg']))
                            try:
                                taus = list(self.config.pose_auc_degs)
                            except Exception:
                                taus = [5.0, 10.0, 20.0]
                            for τ in taus:
                                res[f'pose_S@{int(τ)}'] = (pose_err <= float(τ)).float()
                except Exception:
                    # On any error, use IMB failure values
                    res['pose_R_deg'] = torch.tensor(180.0, device=device)
                    res['pose_t_deg'] = torch.tensor(90.0, device=device)
                    res['pose_inlier_ratio'] = torch.tensor(0.0, device=device)
            # Combined pose error for AUC (translation folded to [0, 90] for sign ambiguity).
            res['_pose_err_deg'] = torch.max(
                res['pose_R_deg'], torch.minimum(res['pose_t_deg'], 180.0 - res['pose_t_deg']))
            pose_err_collect.append(res['_pose_err_deg'].detach())
            # Attach meta
            if meta is not None and i < len(meta) and isinstance(meta[i], dict):
                for k,v in meta[i].items():
                    res[k] = v
            individual.append(res)
        batch_metrics = self.compute_summary_statistics(individual)
        try:
            E = torch.stack([r['_pose_err_deg'] for r in individual if '_pose_err_deg' in r])
            if E.numel() > 0:
                try:
                    taus_deg = list(self.config.pose_auc_degs)
                except Exception:
                    taus_deg = [5.0, 10.0, 20.0]
                for τ in taus_deg:
                    batch_metrics[f'pose_AUC@{int(τ)}'] = _auc_curve(E, float(τ))
                    batch_metrics[f'pose_S@{int(τ)}'] = (E <= float(τ)).float().mean()
                if plot_dir is not None and plot_success_curves:
                    try:
                        max_tau = max(taus_deg) if len(taus_deg) > 0 else 20.0
                    except Exception:
                        max_tau = 20.0
                    taus, suc = _success_curve(E, float(max_tau))
                    title = "Pose success curve"
                    _save_curve(taus, suc, title, xlabel='Threshold (deg)', ylabel='Success rate',
                                out_path=os.path.join(plot_dir, f"{plot_prefix}_pose_success.png"))
        except Exception:
            pass
        try:
            concat_se = torch.cat(se_collect) if len(se_collect) > 0 else torch.tensor([], device=device)
        except Exception:
            concat_se = torch.tensor([], device=device)
        try:
            concat_pose = torch.stack(pose_err_collect) if len(pose_err_collect) > 0 else torch.tensor([], device=device)
        except Exception:
            concat_pose = torch.tensor([], device=device)
        return {
            'individual_results': individual,
            'batch_metrics': batch_metrics,
            'concat_sampson_px': concat_se,
            'concat_pose_err_deg': concat_pose,
        }

    def evaluate_batch_individual(self, pred_dict: Dict[str, Any], image_shape: Tuple[int, int] = (512, 512),
                                  meta: Optional[List[Dict[str, Any]]] = None,
                                  pair_type: Optional[str] = 'planar') -> Dict[str, Any]:
        """
        Evaluate a batch of predictions and return individual sample metrics (not batch-averaged).
        
        Args:
            pred_dict: Dictionary containing:
                - 'kpts0': List of keypoint arrays for image 0 [B x [N0, 2]]
                - 'kpts1': List of keypoint arrays for image 1 [B x [N1, 2]]
                - 'desc0': List of descriptor arrays for image 0 [B x [N0, D]]
                - 'desc1': List of descriptor arrays for image 1 [B x [N1, D]]
                - 'warp01_params': List of warp parameters [B x dict]
            image_shape: Image dimensions (H, W) for evaluation
            
        Returns:
            Dictionary containing:
                - 'individual_results': List of individual sample results
                - 'batch_metrics': Dictionary with batch-averaged metrics (for backward compatibility)
        """
        batch_size = len(pred_dict['kpts0'])
        batch_results = []

        for idx in range(batch_size):
            kpts0 = pred_dict['kpts0'][idx]
            kpts1 = pred_dict['kpts1'][idx]
            desc0 = pred_dict['desc0'][idx]
            desc1 = pred_dict['desc1'][idx]

            if len(kpts0) == 0 or len(kpts1) == 0 or len(desc0) == 0 or len(desc1) == 0:
                batch_results.append(self.get_zero_metrics())
                continue

            if 'warp01_params' in pred_dict and idx < len(pred_dict['warp01_params']):
                warp_params = pred_dict['warp01_params'][idx]
                homography = warp_params['homography_matrix']
                if 'width' in warp_params and 'height' in warp_params:
                    image_shape = (warp_params['height'], warp_params['width'])
            else:
                batch_results.append(self.get_zero_metrics())
                continue

            try:
                device = kpts0.device if hasattr(kpts0, 'device') else self.device
                self.device = device
                self.config.device = device.type if hasattr(device, 'type') else str(device)

                result = self.evaluate(
                    kpts0, kpts1, desc0, desc1, homography, image_shape
                )
                if pair_type is not None:
                    result['pair_type'] = pair_type
                if meta is not None and idx < len(meta) and isinstance(meta[idx], dict):
                    for mk, mv in meta[idx].items():
                        result[mk] = mv
                batch_results.append(result)

            except Exception as e:
                warnings.warn(f"Batch evaluation failed for item {idx}: {e}")
                batch_results.append(self.get_zero_metrics())

        return {
            'individual_results': batch_results,
            'batch_metrics': self.compute_summary_statistics(batch_results)
        }

    def to_tensor(self, data: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """Convert data to PyTorch tensor on the configured device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(self.device)
        elif isinstance(data, list):
            return torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unsupported type: {type(data)}")

    def compute_all_metrics(self, kpts0, kpts1, desc0, desc1, homography, image_shape):
        """Compute comprehensive evaluation metrics with configurable thresholds."""
        h, w = image_shape
        device = kpts0.device if len(kpts0) > 0 else (kpts1.device if len(kpts1) > 0 else self.device)
        
        num_kpts0 = torch.tensor(len(kpts0), dtype=torch.float32, device=device)
        num_kpts1 = torch.tensor(len(kpts1), dtype=torch.float32, device=device)
        
        results = {
            'num_keypoints_0': num_kpts0,
            'num_keypoints_1': num_kpts1,
            'N_cov': torch.tensor(0.0, device=device),
            'N_M': torch.tensor(0.0, device=device),
        }
        
        for thresh in self.config.thresholds:
            results[f'N_gt@{thresh}'] = torch.tensor(0.0, device=device)
            results[f'N_inlier@{thresh}'] = torch.tensor(0.0, device=device)
            results[f'repeatability@{thresh}'] = torch.tensor(0.0, device=device)
            results[f'matching_score@{thresh}'] = torch.tensor(0.0, device=device)
            results[f'MMA@{thresh}'] = torch.tensor(0.0, device=device)
            results[f'MHA@{thresh}'] = torch.tensor(0.0, device=device)
        
        if len(kpts0) == 0 or len(kpts1) == 0:
            return results
        
        try:
            warp01_params = {
                'mode': 'homo',
                'homography_matrix': homography,
                'width': w,
                'height': h
            }
            warp10_params = {
                'mode': 'homo',
                'homography_matrix': torch.inverse(homography),
                'width': w,
                'height': h
            }
            
            kpts0_cov, kpts01_cov, _, _ = warp(kpts0, warp01_params)
            kpts1_cov, kpts10_cov, _, _ = warp(kpts1, warp10_params)
            
            # Check if we have enough covisible keypoints
            if len(kpts0_cov) == 0 or len(kpts1_cov) == 0:
                return results
            
            N_cov = (len(kpts0_cov) + len(kpts1_cov)) / 2.0  # number of covisible keypoints
            results['N_cov'] = torch.tensor(N_cov, dtype=torch.float32, device=device)
            
            dist01 = compute_keypoints_distance(kpts0_cov, kpts10_cov)
            dist10 = compute_keypoints_distance(kpts1_cov, kpts01_cov)
            dist_mutual = (dist01 + dist10.t()) / 2.0
            
            # Check if dist_mutual is valid and doesn't contain NaN/Inf
            if (dist_mutual.numel() == 0 or min(dist_mutual.shape) == 0 or 
                torch.isnan(dist_mutual).any() or torch.isinf(dist_mutual).any()):
                return results
            
            imutual = torch.arange(min(dist_mutual.shape), device=device)
            dist_mutual[imutual, imutual] = 99999
            
            mutual_min_indices = mutual_argmin(dist_mutual)
            
            if len(mutual_min_indices[0]) > 0:
                dist_gt = dist_mutual[mutual_min_indices]
                for thresh in self.config.thresholds:
                    N_gt = (dist_gt <= thresh).sum().float()
                    results[f'N_gt@{thresh}'] = N_gt
                    results[f'repeatability@{thresh}'] = N_gt / max(N_cov, 1)
            
            if len(desc0) > 0 and len(desc1) > 0:
                mi_np, _ = mnn_match_descriptors(desc0.detach().cpu().numpy(), desc1.detach().cpu().numpy())
                matches_est = (torch.as_tensor(mi_np[:,0], device=desc0.device, dtype=torch.long), torch.as_tensor(mi_np[:,1], device=desc1.device, dtype=torch.long)) if mi_np.size>0 else (torch.empty(0, dtype=torch.long, device=desc0.device), torch.empty(0, dtype=torch.long, device=desc1.device))
                
                if len(matches_est[0]) > 0:
                    mkpts0 = kpts0[matches_est[0]]
                    mkpts1 = kpts1[matches_est[1]]
                    N_M = len(mkpts0)  # number of putative matches from MNN
                    results['N_M'] = torch.tensor(N_M, dtype=torch.float32, device=device)
                    
                    # Warp putative matches
                    mkpts0_warped, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
                    
                    if len(ids0) > 0:
                        mkpts1_valid = mkpts1[ids0]
                        dist_reproj = torch.sqrt(((mkpts01 - mkpts1_valid) ** 2).sum(axis=1))
                        if torch.isnan(dist_reproj).any() or torch.isinf(dist_reproj).any():
                            dist_reproj = torch.nan_to_num(dist_reproj, nan=99999.0, posinf=99999.0, neginf=99999.0)
                        for thresh in self.config.thresholds:
                            N_inlier = (dist_reproj <= thresh).sum().float()
                            results[f'N_inlier@{thresh}'] = N_inlier
                            results[f'matching_score@{thresh}'] = N_inlier / max(N_cov, 1)
                            results[f'MMA@{thresh}'] = N_inlier / max(N_M, 1)
                            results[f'MHA@{thresh}'] = self.compute_homography_accuracy(
                                mkpts0_warped, mkpts1_valid, homography, image_shape, thresh
                            )

        except Exception as e:
            warnings.warn(f"Metric computation failed: {e}")
            pass
        
        return results

    def auc_success_curve_px(self, errors: torch.Tensor, threshold_px: float) -> torch.Tensor:
        """Compute AUC of success curve for given error distribution up to threshold_px (in pixels).
        Falls back to 0.0 on empty input or failure.
        """
        try:
            if not torch.is_tensor(errors):
                errors = torch.as_tensor(errors, dtype=torch.float32, device=self.device)
            if errors.numel() == 0:
                return torch.tensor(0.0, device=self.device)
            return _auc_curve(errors, float(threshold_px))
        except Exception:
            return torch.tensor(0.0, device=self.device)

    def compute_homography_accuracy(self, mkpts0, mkpts1, homography_gt, image_shape, threshold):
        """
        Compute Mean Homography Accuracy (MHA): percentage of correct image corners
        with estimated homography under the given threshold.
        """
        device = mkpts0.device
        h, w = image_shape
        
        if len(mkpts0) < 4 or len(mkpts1) < 4:
            return torch.tensor(0.0, device=device)
        
        try:
            # Prefer robust estimation with OpenCV RANSAC if available
            H_est = None
            if len(mkpts0) >= 4 and len(mkpts1) >= 4:
                src_np = mkpts0.detach().float().cpu().numpy()
                dst_np = mkpts1.detach().float().cpu().numpy()
                # Expose RANSAC params via env vars to allow faster tuning without disabling feature
                try:
                    maxIters = int(os.environ.get('EVAL_H_RANSAC_ITERS', '1500'))
                except Exception:
                    maxIters = 1500
                try:
                    conf = float(os.environ.get('EVAL_H_RANSAC_CONF', '0.995'))
                except Exception:
                    conf = 0.995
                try:
                    rth = float(os.environ.get('EVAL_H_RANSAC_THRESH', str(int(threshold))))
                except Exception:
                    rth = float(int(threshold))
                H_np, _ = cv2.findHomography(
                    src_np, dst_np, method=cv2.RANSAC, ransacReprojThreshold=rth, maxIters=maxIters, confidence=conf
                )
                if H_np is not None and np.isfinite(H_np).all():
                    H_est = torch.from_numpy(H_np).to(device=device, dtype=torch.float32)
            if H_est is None:
                return torch.tensor(0.0, device=device)

            if H_est is None or torch.any(torch.isnan(H_est)) or torch.any(torch.isinf(H_est)):
                return torch.tensor(0.0, device=device)

            corners = torch.tensor([
                [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
            ], dtype=torch.float32, device=device)
            
            corners_homo = torch.cat([corners, torch.ones(4, 1, device=device)], dim=1)
            corners_gt_homo = torch.einsum('ij,kj->ki', homography_gt, corners_homo)
            if torch.any(corners_gt_homo[:, 2:3] == 0):
                return torch.tensor(0.0, device=device)
            corners_gt = corners_gt_homo[:, :2] / corners_gt_homo[:, 2:3]

            corners_est_homo = torch.einsum('ij,kj->ki', H_est, corners_homo)
            if torch.any(corners_est_homo[:, 2:3] == 0):
                return torch.tensor(0.0, device=device)
            corners_est = corners_est_homo[:, :2] / corners_est_homo[:, 2:3]

            corner_errors = torch.norm(corners_gt - corners_est, dim=1)
            correct_corners = (corner_errors <= threshold).sum().float()
            mha = correct_corners / 4.0
            # Additional sanity: reject implausible DLT if error is extreme
            # If RANSAC produced an implausible H (very high corner error), treat as failure
            if corner_errors.mean() > max(5*threshold, 50.0):
                return torch.tensor(0.0, device=device)
            return mha
            
        except Exception as e:
            warnings.warn(f"MHA computation failed: {e}")
            return torch.tensor(0.0, device=device)

    def get_zero_metrics(self) -> Dict[str, Any]:
        """Return zero metrics when evaluation fails."""
        metrics = {
            'num_keypoints_0': torch.tensor(0.0, device=self.device), 
            'num_keypoints_1': torch.tensor(0.0, device=self.device),
            'N_cov': torch.tensor(0.0, device=self.device),
            'N_M': torch.tensor(0.0, device=self.device),
        }
        
        for thresh in self.config.thresholds:
            metrics[f'N_gt@{thresh}'] = torch.tensor(0.0, device=self.device)
            metrics[f'N_inlier@{thresh}'] = torch.tensor(0.0, device=self.device)
            metrics[f'repeatability@{thresh}'] = torch.tensor(0.0, device=self.device)
            metrics[f'matching_score@{thresh}'] = torch.tensor(0.0, device=self.device)
            metrics[f'MMA@{thresh}'] = torch.tensor(0.0, device=self.device)
            metrics[f'MHA@{thresh}'] = torch.tensor(0.0, device=self.device)
        
        return metrics

    def compute_simple_matching_score(self, pred_dict: Dict[str, Any], image_shape: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """Lightweight evaluation for training - only computes matching score at 3px threshold."""
        try:
            batch_size = len(pred_dict['kpts0'])
            if batch_size == 0:
                return torch.tensor(0.0, device=self.device)
            
            total_score = torch.tensor(0.0, device=self.device)
            valid_samples = 0
            
            for idx in range(batch_size):
                kpts0 = pred_dict['kpts0'][idx]
                kpts1 = pred_dict['kpts1'][idx] 
                desc0 = pred_dict['desc0'][idx]
                desc1 = pred_dict['desc1'][idx]
                
                if len(kpts0) == 0 or len(kpts1) == 0 or len(desc0) == 0 or len(desc1) == 0:
                    continue
                    
                if 'warp01_params' in pred_dict and idx < len(pred_dict['warp01_params']):
                    homography = pred_dict['warp01_params'][idx]['homography_matrix']
                    h = pred_dict['warp01_params'][idx].get('height', image_shape[0])
                    w = pred_dict['warp01_params'][idx].get('width', image_shape[1])
                else:
                    continue
                
                score = self.compute_simple_accuracy(kpts0, kpts1, desc0, desc1, homography, (h, w), threshold=3.0)
                
                if score > 0:
                    total_score += score
                    valid_samples += 1
            
            if valid_samples > 0:
                return total_score / valid_samples
            else:
                return torch.tensor(0.0, device=self.device)
                
        except Exception as e:
            warnings.warn(f"Simple matching score computation failed: {e}")
            return torch.tensor(0.0, device=self.device)

    def compute_simple_nonplanar_score(self, pred_dict: Dict[str, Any], cam_pack: Dict[str, Any], epi_T_px: float = 5.0,
                                       max_se_samples_per_pair: int = 2000) -> torch.Tensor:
        """Lightweight nonplanar training metric: epipolar AUC over [0, epi_T_px] px.
        Uses per-pair Sampson errors from MNN matches and averages AUC across the batch.
        """
        try:
            B = len(pred_dict['kpts0']) if 'kpts0' in pred_dict else 0
            if B == 0:
                return torch.tensor(0.0, device=self.device)
            auc_total = torch.tensor(0.0, device=self.device)
            valid = 0
            for i in range(B):
                k0 = pred_dict['kpts0'][i]; k1 = pred_dict['kpts1'][i]
                d0 = pred_dict['desc0'][i]; d1 = pred_dict['desc1'][i]
                if len(k0)==0 or len(k1)==0 or len(d0)==0 or len(d1)==0:
                    continue
                mi_np, _ = mnn_match_descriptors(d0.detach().cpu().numpy(), d1.detach().cpu().numpy())
                idx0 = torch.as_tensor(mi_np[:,0], device=d0.device, dtype=torch.long) if mi_np.size>0 else torch.empty(0, dtype=torch.long, device=d0.device)
                idx1 = torch.as_tensor(mi_np[:,1], device=d1.device, dtype=torch.long) if mi_np.size>0 else torch.empty(0, dtype=torch.long, device=d1.device)
                mk0 = k0[idx0]; mk1 = k1[idx1]
                if mk0.numel() == 0 or mk1.numel() == 0:
                    continue
                K0 = cam_pack['K0'][i]; K1 = cam_pack['K1'][i]
                R  = cam_pack['R_01'][i]; t = cam_pack['t_01'][i]
                # Use normalized-space Sampson via E, then scale to px
                # Check for degenerate case: very small baseline
                t_norm = float(torch.norm(t))
                if t_norm < 1e-6:
                    # Degenerate case: no inliers
                    se = mk0.new_ones((max(1, len(mk0)),)) * 999.0
                else:
                    Egt = _E_from_pose(R, t)
                    if mk0.numel() > 0 and mk1.numel() > 0:
                        x0h = torch.cat([mk0, torch.ones(mk0.shape[0], 1, device=mk0.device, dtype=mk0.dtype)], 1)
                        x1h = torch.cat([mk1, torch.ones(mk1.shape[0], 1, device=mk1.device, dtype=mk1.dtype)], 1)
                        K0inv = torch.linalg.inv(K0); K1inv = torch.linalg.inv(K1)
                        x0n = (K0inv @ x0h.t()).t()
                        x1n = (K1inv @ x1h.t()).t()
                        Ex0  = (Egt  @ x0n.t()).t()
                        Etx1 = (Egt.t() @ x1n.t()).t()
                        num  = (x1n * (Egt @ x0n.t()).t()).sum(1).abs()
                        # Use adaptive regularization for small baselines
                        baseline_factor = max(1.0, 1e-6 / max(t_norm, 1e-12))
                        reg_term = 1e-8 * baseline_factor
                        den  = Ex0[:,0].pow(2) + Ex0[:,1].pow(2) + Etx1[:,0].pow(2) + Etx1[:,1].pow(2) + reg_term
                        se_n = num / torch.sqrt(den)
                        fx0, fy0 = K0[0,0], K0[1,1]
                        fx1, fy1 = K1[0,0], K1[1,1]
                        f_rep = torch.tensor(float((fx0 + fy0 + fx1 + fy1) / 4.0), device=mk0.device, dtype=mk0.dtype)
                        se = se_n * f_rep
                    else:
                        se = mk0.new_zeros((0,))
                if se.numel() == 0:
                    continue
                if se.numel() > max_se_samples_per_pair > 0:
                    perm = torch.randperm(se.numel(), device=se.device)
                    se = se[perm[:max_se_samples_per_pair]]
                auc = _auc_curve(se, float(epi_T_px))
                auc_total = auc_total + auc
                valid += 1
            return (auc_total / valid) if valid > 0 else torch.tensor(0.0, device=self.device)
        except Exception as e:
            warnings.warn(f"Simple nonplanar score computation failed: {e}")
            return torch.tensor(0.0, device=self.device)

    def compute_simple_accuracy(self, kpts0, kpts1, desc0, desc1, homography, image_shape, threshold=3.0):
        """Compute simple matching score metric for training evaluation."""
        h, w = image_shape
        device = kpts0.device if len(kpts0) > 0 else (kpts1.device if len(kpts1) > 0 else self.device)
        
        if len(kpts0) == 0 or len(kpts1) == 0 or len(desc0) == 0 or len(desc1) == 0:
            return torch.tensor(0.0, device=device)
        
        try:
            mi_np, _ = mnn_match_descriptors(desc0.detach().cpu().numpy(), desc1.detach().cpu().numpy())
            matches_est = (torch.as_tensor(mi_np[:,0], device=desc0.device, dtype=torch.long), torch.as_tensor(mi_np[:,1], device=desc1.device, dtype=torch.long)) if mi_np.size>0 else (torch.empty(0, dtype=torch.long, device=desc0.device), torch.empty(0, dtype=torch.long, device=desc1.device))
            
            if len(matches_est[0]) == 0:
                return torch.tensor(0.0, device=device)
            
            mkpts0 = kpts0[matches_est[0]]
            mkpts1 = kpts1[matches_est[1]]
            
            warp01_params = {
                'mode': 'homo',
                'homography_matrix': homography,
                'width': w,
                'height': h
            }
            
            mkpts0_warped, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
            
            if len(ids0) == 0:
                return torch.tensor(0.0, device=device)
            
            mkpts1_valid = mkpts1[ids0]
            
            dist = torch.sqrt(((mkpts01 - mkpts1_valid) ** 2).sum(axis=1))
            if dist.shape[0] == 0:
                return torch.tensor(0.0, device=device)
            
            correct = dist < threshold
            matching_score = correct.float().mean()
            
            return matching_score
            
        except Exception as e:
            warnings.warn(f"Simple accuracy computation failed: {e}")
            return torch.tensor(0.0, device=device)

    def generate_visualizations(self, kpts0, kpts1, desc0, desc1, image0, image1, homography, results, return_images):
        """Generate visualization data."""
        if isinstance(return_images, bool):
            requested_images = ['keypoints0', 'keypoints1', 'matches'] if return_images else []
        else:
            requested_images = return_images
        
        vis_data = {}
        
        if 'raw_data' in requested_images:
            vis_data['raw_data'] = {
                'image0': image0, 'image1': image1,
                'keypoints0': kpts0, 'keypoints1': kpts1,
                'descriptors0': desc0, 'descriptors1': desc1,
                'homography': homography
            }
        
        return vis_data

    def batch_evaluate(self,
                      kpts0_list, kpts1_list, desc0_list, desc1_list, homography_list,
                      image_shape=(512, 512), **kwargs) -> List[Dict[str, Any]]:
        results = []
        for i in range(len(kpts0_list)):
            try:
                result = self.evaluate(
                    kpts0_list[i], kpts1_list[i], desc0_list[i], desc1_list[i], 
                    homography_list[i], image_shape, **kwargs
                )
                results.append(result)
            except Exception as e:
                warnings.warn(f"Batch evaluation failed for item {i}: {e}")
                results.append(self.get_zero_metrics())
        return results

    def compute_summary_statistics(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from evaluation results."""
        if not results_list:
            return {}
        
        device = self.device
        all_metrics = {}
        
        for result in results_list:
            for key, value in result.items():
                if key != 'visualizations':
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            val = value.item()
                        else:
                            val = value.mean().item()
                    elif isinstance(value, (int, float)):
                        val = value
                    else:
                        continue
                    
                    if np.isnan(val) or np.isinf(val):
                        continue
                    
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(val)
        
        summary = {}
        for metric, values in all_metrics.items():
            if values and len(values) > 0:
                clean_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                if len(clean_values) > 0:
                    values_tensor = torch.tensor(clean_values, dtype=torch.float32, device=device)
                    summary[f'{metric}_mean'] = torch.mean(values_tensor)
                    if len(clean_values) == 1:
                        summary[f'{metric}_std'] = torch.tensor(0.0, device=device)
                    else:
                        std_val = torch.std(values_tensor)
                        if torch.isnan(std_val):
                            summary[f'{metric}_std'] = torch.tensor(0.0, device=device)
                        else:
                            summary[f'{metric}_std'] = std_val
                else:
                    summary[f'{metric}_mean'] = torch.tensor(0.0, device=device)
                    summary[f'{metric}_std'] = torch.tensor(0.0, device=device)
        
        return summary

    # Trimmed: local helpers moved to models.utils.(geometry|metrics|plotting)

    def evaluate_nonplanar_batch(self, pred_dict: Dict[str, Any], cam_pack: Dict[str, Any],
                             epi_thresholds_px=(1.0, 2.0, 3.0, 5.0),
                             plot_dir: Optional[str] = None,
                             plot_prefix: str = "nonplanar",
                             plot_success_curves: bool = True,
                             max_se_samples_per_pair: int = 4096,
                             min_sim_threshold: Optional[float] = None,
                             lowe_ratio_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        pred_dict: like your planar eval (kpts/desc lists).
        cam_pack:  lists/tensors per item: K0, K1, R_01, t_01
        Returns dict of aggregated metrics over the batch.
        """
        B = len(pred_dict['kpts0'])
        device = self.device
        agg = {}
        def acc(name, val):
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=device)
            agg.setdefault(name, []).append(val)

        for i in range(B):
            k0 = pred_dict['kpts0'][i]; k1 = pred_dict['kpts1'][i]
            d0 = pred_dict['desc0'][i]; d1 = pred_dict['desc1'][i]
            if len(k0)==0 or len(k1)==0 or len(d0)==0 or len(d1)==0:
                for τ in epi_thresholds_px:
                    acc(f'N_inlier_epi@{int(τ)}', 0.0); acc(f'MS_epi@{int(τ)}', 0.0)
                acc('sampson_mean', 0.0); acc('sampson_median', 0.0)
                acc('N_M', 0.0); acc('pose_R_deg', 180.0); acc('pose_t_deg', 90.0)  # IMB: 90° for translation when failed
                acc('pose_inlier_ratio', 0.0); acc('tri_rmse_px', 0.0); acc('tri_cheirality', 0.0)
                continue

            try:
                mi_np, _ = mnn_match_descriptors(d0.detach().cpu().numpy(), d1.detach().cpu().numpy(),
                                sim_threshold=self.config.min_sim_threshold,
                                ratio_threshold=self.config.lowe_ratio_threshold)
                idx0 = torch.as_tensor(mi_np[:,0], device=d0.device, dtype=torch.long) if mi_np.size>0 else torch.empty(0, dtype=torch.long, device=d0.device)
                idx1 = torch.as_tensor(mi_np[:,1], device=d1.device, dtype=torch.long) if mi_np.size>0 else torch.empty(0, dtype=torch.long, device=d1.device)
            except Exception:
                idx0 = torch.empty(0, dtype=torch.long, device=d0.device)
                idx1 = torch.empty(0, dtype=torch.long, device=d1.device)
            mk0 = k0[idx0]; mk1 = k1[idx1]
            N_M = float(len(mk0))
            acc('N_M', N_M)

            K0 = cam_pack['K0'][i]; K1 = cam_pack['K1'][i]
            R  = cam_pack['R_01'][i]; t = cam_pack['t_01'][i]

            # Very small baseline makes epipolar geometry unreliable
            t_norm = float(torch.norm(t))
            if t_norm < 1e-6:
                se = mk0.new_ones((max(1, len(mk0)),)) * 999.0
            else:
                Egt = _E_from_pose(R, t)
                if mk0.numel() > 0 and mk1.numel() > 0:
                    fx0, fy0 = K0[0,0], K0[1,1]
                    fx1, fy1 = K1[0,0], K1[1,1]
                    x0h = torch.cat([mk0, torch.ones(mk0.shape[0], 1, device=mk0.device, dtype=mk0.dtype)], 1)
                    x1h = torch.cat([mk1, torch.ones(mk1.shape[0], 1, device=mk1.device, dtype=mk1.dtype)], 1)
                    K0inv = torch.linalg.inv(K0); K1inv = torch.linalg.inv(K1)
                    x0n = (K0inv @ x0h.t()).t()
                    x1n = (K1inv @ x1h.t()).t()
                    Ex0  = (Egt  @ x0n.t()).t()
                    Etx1 = (Egt.t() @ x1n.t()).t()
                    num  = (x1n * (Egt @ x0n.t()).t()).sum(1).abs()
                    # Use larger regularization for small baselines
                    baseline_factor = max(1.0, 1e-6 / max(t_norm, 1e-12))
                    reg_term = 1e-8 * baseline_factor  # Adaptive regularization
                    den  = Ex0[:,0].pow(2) + Ex0[:,1].pow(2) + Etx1[:,0].pow(2) + Etx1[:,1].pow(2) + reg_term
                    se_n = num / torch.sqrt(den)  # normalized Sampson; scale to px below
                    f_rep = torch.tensor(float((fx0 + fy0 + fx1 + fy1) / 4.0), device=mk0.device, dtype=mk0.dtype)
                    se = se_n * f_rep  # convert normalized Sampson to pixel units
                else:
                    se = mk0.new_zeros((0,))
            # Optional cross-check: Sampson via F (pixel-domain), direct threshold counts
            try:
                Fgt = _F_from_pose(K0, K1, R, t)
                if mk0.numel() > 0 and mk1.numel() > 0:
                    x0h_px = torch.cat([mk0, torch.ones(mk0.shape[0], 1, device=mk0.device, dtype=mk0.dtype)], 1)
                    x1h_px = torch.cat([mk1, torch.ones(mk1.shape[0], 1, device=mk1.device, dtype=mk1.dtype)], 1)
                    Ex0_px  = (Fgt  @ x0h_px.t()).t()
                    Etx1_px = (Fgt.t() @ x1h_px.t()).t()
                    num_px  = (x1h_px * (Fgt @ x0h_px.t()).t()).sum(1).abs()
                    den_px  = Ex0_px[:,0].pow(2) + Ex0_px[:,1].pow(2) + Etx1_px[:,0].pow(2) + Etx1_px[:,1].pow(2) + 1e-12
                    se_px = num_px / torch.sqrt(den_px)
                else:
                    se_px = mk0.new_zeros((0,))
            except Exception:
                se_px = mk0.new_zeros((0,))
            # Downsample Sampson errors per pair for memory control
            if se.numel() > max_se_samples_per_pair > 0:
                perm = torch.randperm(se.numel(), device=se.device)
                se = se[perm[:max_se_samples_per_pair]]
            if se.numel() > 0:
                acc('sampson_mean', se.mean())
                acc('sampson_median', se.median())
            else:
                acc('sampson_mean', 0.0); acc('sampson_median', 0.0)
            ninls = []
            for τ in self.config.epi_thresholds_px:
                Ninl = float((se <= τ).sum()) if se.numel() > 0 else 0.0
                ninls.append(int(Ninl))
                acc(f'N_inlier_epi@{int(τ)}', Ninl)
                acc(f'MS_epi@{int(τ)}', (Ninl / max(N_M, 1.0)))
            acc('_sampson_all', se if se.numel()>0 else torch.tensor([], device=device))

            # Pose-from-matches using robust (more robust USAC_MAGSAC)
            R_err = torch.tensor(180.0, device=device); t_err = torch.tensor(90.0, device=device); inl_ratio = torch.tensor(0.0, device=device)
            try:
                if len(mk0) >= 5:
                    K0v = K0.detach().cpu().numpy(); K1v = K1.detach().cpu().numpy()
                    mk0_np = mk0.detach().cpu().numpy()
                    mk1_np = mk1.detach().cpu().numpy()

                    # Check for degenerate case: if translation is too small, skip pose estimation
                    t_norm_val = float(torch.norm(t))
                    if t_norm_val < 1e-6:  # Very small baseline
                        R_err = torch.tensor(180.0, device=device)
                        t_err = torch.tensor(90.0, device=device)
                        inl_ratio = torch.tensor(0.0, device=device)
                    else:
                        # Use robust with USAC_MAGSAC
                        # thresh=1.0 in pixels (normalized internally), conf=0.99999
                        pose_result = estimate_pose(mk0_np, mk1_np, K0v, K1v, thresh=1.0, conf=0.99999)
                        
                        if pose_result is not None:
                            R_est_np, t_est_np, inlier_mask, _ = pose_result
                            R_est = torch.from_numpy(R_est_np).to(device=device, dtype=torch.float32)
                            t_est = torch.from_numpy(t_est_np).to(device=device, dtype=torch.float32)

                            # Validate the estimated pose
                            if torch.isnan(R_est).any() or torch.isinf(R_est).any() or torch.isnan(t_est).any() or torch.isinf(t_est).any():
                                R_err = torch.tensor(180.0, device=device)
                                t_err = torch.tensor(90.0, device=device)
                                inl_ratio = torch.tensor(0.0, device=device)
                            else:
                                R_err = _geo_deg(R_est, R)
                                t_err = _angle_deg(t_est, t)
                                num_inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
                                inl_ratio = torch.tensor(float(num_inliers)/max(len(mk0),1), device=device)
                        else:
                            R_err = torch.tensor(180.0, device=device)
                            t_err = torch.tensor(90.0, device=device)
                            inl_ratio = torch.tensor(0.0, device=device)
            except Exception:
                # IMB failure values (180° R, 90° t)
                R_err = torch.tensor(180.0, device=device)
                t_err = torch.tensor(90.0, device=device)
                inl_ratio = torch.tensor(0.0, device=device)
            acc('pose_R_deg', R_err); acc('pose_t_deg', t_err); acc('pose_inlier_ratio', inl_ratio)
            acc('pose_avg_deg', 0.5*(R_err + t_err))
            # Collect per-item combined pose error (deg); translation folded to [0,90]
            acc('_pose_err_deg', torch.max(R_err, torch.minimum(t_err, 180.0 - t_err)))

            # Triangulation consistency with GT pose
            X, mask = _triangulate(K0, K1, R, t, mk0, mk1)
            if X.shape[1] > 0 and mask.any():
                X1 = R @ X + t.view(3,1)
                proj0 = (K0 @ X) ; proj0 = (proj0[:2] / (proj0[2:3] + 1e-12)).T
                proj1 = (K1 @ X1); proj1 = (proj1[:2] / (proj1[2:3] + 1e-12)).T
                d0 = torch.norm(proj0 - mk0[:X.shape[1]], dim=1)
                d1 = torch.norm(proj1 - mk1[:X.shape[1]], dim=1)
                rmse = torch.sqrt(0.5*(d0.pow(2)+d1.pow(2)).mean())
                rmse = torch.clamp(rmse, max=580)
                acc('tri_rmse_px', rmse)
                acc('tri_cheirality', mask.float().mean())
            else:
                acc('tri_rmse_px', 0.0); acc('tri_cheirality', 0.0)

        out = {}
        for k, vals in agg.items():
            V = torch.stack([v if torch.is_tensor(v) else torch.tensor(v, device=device) for v in vals])
            out[f'{k}_mean'] = V.mean()
            out[f'{k}_std']  = V.std() if V.numel()>1 else torch.tensor(0.0, device=device)
        # Pose AUCs (deg)
        if '_pose_err_deg' in agg and len(agg['_pose_err_deg'])>0:
            E = torch.stack([v if torch.is_tensor(v) else torch.tensor(v, device=device) for v in agg['_pose_err_deg']])
            try:
                taus_deg = list(self.config.pose_auc_degs)
            except Exception:
                taus_deg = [5.0, 10.0, 20.0]
            for τ in taus_deg:
                out[f'pose_AUC@{int(τ)}'] = _auc_curve(E, float(τ))
                out[f'pose_S@{int(τ)}'] = (E <= float(τ)).float().mean()
            # Plot pose success curve up to max configured deg
            if plot_dir is not None and plot_success_curves and self.config.enable_plots:
                try:
                    max_tau = max(taus_deg) if len(taus_deg) > 0 else 20.0
                except Exception:
                    max_tau = 20.0
                taus, suc = _success_curve(E, float(max_tau))
                title = "Pose success curve"
                _save_curve(taus, suc, title, xlabel='Threshold (deg)', ylabel='Success rate',
                            out_path=os.path.join(plot_dir, f"{plot_prefix}_pose_success.png"))
        if '_sampson_all' in agg and len(agg['_sampson_all'])>0:
            se_list = []
            for v in agg['_sampson_all']:
                if v.numel() > 0:
                    se_list.append(v)
            if len(se_list) > 0:
                SE = torch.cat(se_list)
                out['epi_AUC@3px'] = _auc_curve(SE, self.config.epi_auc_px[0])
                out['epi_AUC@5px'] = _auc_curve(SE, self.config.epi_auc_px[1])
                # Plot epipolar success curve up to 5 px
                if plot_dir is not None and plot_success_curves and self.config.enable_plots:
                    taus_px, suc_px = _success_curve(SE, max(self.config.epi_auc_px))
                    title = f"Epipolar (Sampson) success curve (AUC@3px={float(out['epi_AUC@3px']):.3f}, @5px={float(out['epi_AUC@5px']):.3f})"
                    _save_curve(taus_px, suc_px, title, xlabel='Threshold (px)', ylabel='Success rate',
                                out_path=os.path.join(plot_dir, f"{plot_prefix}_epi_success.png"))
        return out
