import torch
from losses.utils import weighted_mean, huber
import torch.nn.functional as F

class SoftPeakyLoss(object):
    def __init__(self, scores_th: float = 0.1, debug: bool = False, soft: bool = True):
        super().__init__()
        self.scores_th = scores_th
        self.debug = debug
        self.soft = soft  # use soft weighting instead of hard mask

    def __call__(self, pred):
        b, c, h, w = pred['scores_map'].shape
        loss = pred['scores_map'].new_zeros(())

        for i in range(b):
            scores_i = pred['scores'][i]              # (Ni,)
            disp_i   = pred['score_dispersity'][i]    # (Ni,)
            n = min(len(scores_i), len(disp_i))
            if n == 0:
                continue
            scores_i = scores_i[:n]
            disp_i   = disp_i[:n]

            if self.soft:
                w = torch.sigmoid(10.0 * (scores_i - self.scores_th))
            else:
                w = (scores_i > self.scores_th).float()

            # encourage peaky responses => minimise dispersity with confidence weights
            if w.sum() > 0:
                loss = loss + weighted_mean(disp_i, w)

        return loss / max(1, b)
    
class HuberReprojectionLocLoss(object):
    def __init__(self, norm: int = 1, scores_th: float = 0.1, debug: bool = False, huber_delta: float = 1.0):
        super().__init__()
        self.norm = norm
        self.scores_th = scores_th
        self.debug = debug
        self.huber_delta = huber_delta

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        loss = pred0['scores_map'].new_zeros(())

        for i in range(b):
            corr = correspondences[i]
            if corr.get('correspondence0', None) is None:
                continue

            dist = corr['dist'] if self.norm == 2 else corr['dist_l1']
            ids0_d, ids1_d = corr['ids0_d'], corr['ids1_d']
            if len(ids0_d) == 0:
                continue

            s0 = corr['scores0'].detach()[ids0_d]
            s1 = corr['scores1'].detach()[ids1_d]
            w_ = s0 * s1
            w_mask = torch.sigmoid(10.0 * (s0 - self.scores_th)) * torch.sigmoid(10.0 * (s1 - self.scores_th))
            w_ = w_ * w_mask

            errs = dist[ids0_d, ids1_d]
            errs = huber(errs, self.huber_delta)
            if w_.sum() > 0:
                loss = loss + weighted_mean(errs, w_)

        return loss / max(1, b)
    
class DescReprojectionLoss(object):
    """Dense descriptor reprojection loss: NLL over softmax similarities with explicit no-match class for outliers."""

    def __init__(self, temperature=0.02, debug: bool = False):
        super().__init__()
        self.inv_temp = 1. / temperature
        self.debug = debug

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        device = pred0['scores_map'].device
        wh = pred0['keypoints'][0].new_tensor([[w - 1, h - 1]])
        loss_mean = pred0['scores_map'].new_zeros(())
        CNT = 0

        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            kpts01, kpts10 = correspondences[idx]['kpts01'], correspondences[idx]['kpts10']
            similarity_map_01 = correspondences[idx]['similarity_map_01']
            similarity_map_10 = correspondences[idx]['similarity_map_10']
            ids0, ids1 = correspondences[idx]['ids0'], correspondences[idx]['ids1']
            ids0_out, ids1_out = correspondences[idx]['ids0_out'], correspondences[idx]['ids1_out']

            similarity_map_01_valid, similarity_map_10_valid = similarity_map_01[ids0], similarity_map_10[ids1]
            similarity_map_01_valid = (similarity_map_01_valid - 1) * self.inv_temp
            similarity_map_10_valid = (similarity_map_10_valid - 1) * self.inv_temp

            if similarity_map_01_valid.numel() > 0 and similarity_map_01_valid.shape[-2:] == (h, w):
                pmf01_valid = torch.softmax(similarity_map_01_valid.view(-1, h * w), dim=1).view(-1, h, w)
            else:
                pmf01_valid = torch.zeros(0, h, w, device=device)

            if similarity_map_10_valid.numel() > 0 and similarity_map_10_valid.shape[-2:] == (h, w):
                pmf10_valid = torch.softmax(similarity_map_10_valid.view(-1, h * w), dim=1).view(-1, h, w)
            else:
                pmf10_valid = torch.zeros(0, h, w, device=device)

            if len(kpts01) > 0 and pmf01_valid.numel() > 0:
                pmf01_kpts_valid = torch.nn.functional.grid_sample(pmf01_valid.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                                   mode='bilinear', align_corners=True)[0, :, 0, :]
                C01 = torch.diag(pmf01_kpts_valid)
            else:
                C01 = torch.tensor([], device=device)
            if len(kpts10) > 0 and pmf10_valid.numel() > 0:
                pmf10_kpts_valid = torch.nn.functional.grid_sample(pmf10_valid.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                                   mode='bilinear', align_corners=True)[0, :, 0, :]
                C10 = torch.diag(pmf10_kpts_valid)
            else:
                C10 = torch.tensor([], device=device)

            valid_ids0_out = ids0_out[ids0_out < len(similarity_map_01)] if len(ids0_out) > 0 else ids0_out
            valid_ids1_out = ids1_out[ids1_out < len(similarity_map_10)] if len(ids1_out) > 0 else ids1_out
            if len(valid_ids0_out) > 0:
                similarity_map_01_out = similarity_map_01[valid_ids0_out]
                out0 = torch.ones(len(similarity_map_01_out), device=device)
                similarity_map_01_out = torch.cat([similarity_map_01_out.reshape(-1, h * w), out0[:, None]], dim=1)
            else:
                similarity_map_01_out = torch.empty(0, h * w + 1, device=device)
            if len(valid_ids1_out) > 0:
                similarity_map_10_out = similarity_map_10[valid_ids1_out]
                out1 = torch.ones(len(similarity_map_10_out), device=device)
                similarity_map_10_out = torch.cat([similarity_map_10_out.reshape(-1, h * w), out1[:, None]], dim=1)
            else:
                similarity_map_10_out = torch.empty(0, h * w + 1, device=device)
            similarity_map_01_out = (similarity_map_01_out - 1) * self.inv_temp
            similarity_map_10_out = (similarity_map_10_out - 1) * self.inv_temp
            pmf01_out = torch.softmax(similarity_map_01_out, dim=1)
            pmf10_out = torch.softmax(similarity_map_10_out, dim=1)
            C01_out = pmf01_out[:, -1] if len(pmf01_out) > 0 else C01.new_tensor([])
            C10_out = pmf10_out[:, -1] if len(pmf10_out) > 0 else C10.new_tensor([])

            all_tensors = [C for C in [C01, C10, C01_out, C10_out] if len(C) > 0]
            if all_tensors:
                C = torch.cat(all_tensors)
                C_widetilde = -C.log()
                loss_mean = loss_mean + C_widetilde.sum()
                CNT = CNT + len(C_widetilde)

        loss_mean = loss_mean / CNT if CNT != 0 else wh.new_zeros(())
        assert not torch.isnan(loss_mean)
        return loss_mean

class ScoreMapReliabilityLoss(object):
    """Score map reliability loss: high scores should correlate with good matches across viewpoints."""

    def __init__(self, temperature: float = 0.1, debug: bool = False):
        super().__init__()
        self.temperature = temperature
        self.radius = 2
        self.debug = debug

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        wh = pred0['keypoints'][0].new_tensor([[w - 1, h - 1]])
        loss_mean = pred0['scores_map'].new_zeros(())
        CNT = 0

        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            scores_map0 = pred0['scores_map'][idx]
            scores_map1 = pred1['scores_map'][idx]
            kpts01 = correspondences[idx]['kpts01']
            kpts10 = correspondences[idx]['kpts10']

            valid_mask01 = torch.isfinite(kpts01).all(dim=1)
            valid_mask10 = torch.isfinite(kpts10).all(dim=1)

            if valid_mask01.sum() == 0 or valid_mask10.sum() == 0:
                continue

            kpts01 = kpts01[valid_mask01]
            kpts10 = kpts10[valid_mask10]

            scores_kpts10 = torch.nn.functional.grid_sample(scores_map0.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]
            scores_kpts01 = torch.nn.functional.grid_sample(scores_map1.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]

            s0 = scores_kpts01 * correspondences[idx]['scores0']
            s1 = scores_kpts10 * correspondences[idx]['scores1']

            similarity_map_01 = correspondences[idx]['similarity_map_01_valid']
            similarity_map_10 = correspondences[idx]['similarity_map_10_valid']

            pmf01 = ((similarity_map_01.detach() - 1) / self.temperature).exp()
            pmf10 = ((similarity_map_10.detach() - 1) / self.temperature).exp()
            kpts01 = kpts01.detach()
            kpts10 = kpts10.detach()

            pmf01_kpts = torch.nn.functional.grid_sample(pmf01.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            pmf10_kpts = torch.nn.functional.grid_sample(pmf10.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            fs0 = torch.diag(pmf01_kpts)
            fs1 = torch.diag(pmf10_kpts)

            if s0.sum() != 0:
                loss01 = (1 - fs0) * s0 * len(s0) / s0.sum()
                if not torch.isnan(loss01.sum()):
                    loss_mean = loss_mean + loss01.sum()
                    CNT = CNT + len(loss01)
            if s1.sum() != 0:
                loss10 = (1 - fs1) * s1 * len(s1) / s1.sum()
                if not torch.isnan(loss10.sum()):
                    loss_mean = loss_mean + loss10.sum()
                    CNT = CNT + len(loss10)
                    
        loss_mean = loss_mean / CNT if CNT != 0 else pred0['scores_map'].new_zeros(())
        assert not torch.isnan(loss_mean)
        return loss_mean

# ===================== Sparse variants =====================

class SparseDescReprojectionLoss(object):
    def __init__(self, temperature=0.02, debug: bool = False):
        super().__init__()
        self.inv_temp = 1. / temperature
        self.debug = debug

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        loss_sum = pred0['scores_map'].new_zeros(())
        CNT = 0

        for idx in range(b):
            if 'pair_type' in correspondences[idx] and correspondences[idx]['pair_type'] != 'planar':
                continue
            if 'correspondence0' not in correspondences[idx] or correspondences[idx]['correspondence0'] is None:
                continue
            S01 = correspondences[idx]['S01']
            S10 = correspondences[idx]['S10']
            ids0_d = correspondences[idx]['ids0_d']
            ids1_d = correspondences[idx]['ids1_d']
            if len(ids0_d) == 0:
                continue
            logits01 = (S01 - 1) * self.inv_temp
            logits10 = (S10 - 1) * self.inv_temp
            loss01 = torch.nn.functional.cross_entropy(logits01[ids0_d], ids1_d, reduction='sum')
            loss10 = torch.nn.functional.cross_entropy(logits10[ids1_d], ids0_d, reduction='sum')
            loss_sum = loss_sum + loss01 + loss10
            CNT += len(ids0_d) + len(ids1_d)

        loss_mean = loss_sum / CNT if CNT != 0 else pred0['scores_map'].new_zeros(())
        assert not torch.isnan(loss_mean)
        return loss_mean

class SparseScoreMapReliabilityLoss(object):
    def __init__(self, temperature: float = 0.1, debug: bool = False):
        super().__init__()
        self.temperature = temperature
        self.inv_temp = 1.0 / temperature
        self.debug = debug

    def pick_weights(self, scores, ids, expected_len, device):
        """
        Make scores length == expected_len.
        If `scores` already matches, return as-is.
        Else, try indexing by `ids`. Else, truncate/pad as last resort.
        """
        if scores is None or expected_len == 0:
            return torch.ones(expected_len, device=device)

        scores = scores.to(device)

        if scores.numel() == expected_len:
            return scores

        if ids is not None and ids.numel() == expected_len:
            # only index if ids are valid for the scores tensor
            if scores.numel() > 0 and ids.max().item() < scores.numel():
                return scores[ids.to(device=device, dtype=torch.long)]

        # fallback: truncate or pad with ones
        if scores.numel() >= expected_len:
            return scores[:expected_len]
        else:
            pad = torch.ones(expected_len - scores.numel(), device=device, dtype=scores.dtype)
            return torch.cat([scores, pad], dim=0)

    def __call__(self, pred0, pred1, correspondences):
        b, c, H, W = pred0['scores_map'].shape
        device = pred0['scores_map'].device
        loss_sum = pred0['scores_map'].new_zeros(())
        CNT = 0
        eps = 1e-6

        for i in range(b):
            corr = correspondences[i]
            if corr.get('correspondence0', None) is None:
                continue

            kpts01 = corr['kpts01']
            kpts10 = corr['kpts10']
            if kpts01.numel() == 0 or kpts10.numel() == 0:
                continue

            kpts01 = kpts01.clamp(min=-1.0 + eps, max=1.0 - eps)
            kpts10 = kpts10.clamp(min=-1.0 + eps, max=1.0 - eps)

            scores_map0 = pred0['scores_map'][i]
            scores_map1 = pred1['scores_map'][i]

            # --- weights (handle pre-indexed vs. full tensors) ---
            scores0 = corr.get('scores0', None)
            scores1 = corr.get('scores1', None)
            ids0    = corr.get('ids0', None)
            ids1    = corr.get('ids1', None)

            w0 = self.pick_weights(scores0, ids0, kpts01.shape[0], device)
            w1 = self.pick_weights(scores1, ids1, kpts10.shape[0], device)

            # --- repeatability terms sampled at warped locations ---
            s0 = F.grid_sample(scores_map1.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                               mode='bilinear', align_corners=True)[0, 0, 0, :] * w0
            s1 = F.grid_sample(scores_map0.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                               mode='bilinear', align_corners=True)[0, 0, 0, :] * w1

            # --- similarity → PMF (sparse) ---
            S01 = corr['S01']  # (N0,N1)
            S10 = corr['S10']  # (N1,N0)
            pmf01 = ((S01.detach() - 1.0) * self.inv_temp).exp()
            pmf10 = ((S10.detach() - 1.0) * self.inv_temp).exp()

            nn01 = corr['nn01']
            nn10 = corr['nn10']

            if nn01.numel() > 0:
                nn01 = nn01.to(device=device, dtype=torch.long).clamp(0, pmf01.shape[1] - 1)
                fs0 = pmf01[torch.arange(pmf01.shape[0], device=device), nn01]
            else:
                fs0 = s0.new_zeros((0,))

            if nn10.numel() > 0:
                nn10 = nn10.to(device=device, dtype=torch.long).clamp(0, pmf10.shape[1] - 1)
                fs1 = pmf10[torch.arange(pmf10.shape[0], device=device), nn10]
            else:
                fs1 = s1.new_zeros((0,))

            # Align lengths to be safe (planar+sparse paths should already match)
            if fs0.numel() != s0.numel():
                m = min(fs0.numel(), s0.numel())
                fs0, s0 = fs0[:m], s0[:m]
            if fs1.numel() != s1.numel():
                m = min(fs1.numel(), s1.numel())
                fs1, s1 = fs1[:m], s1[:m]

            # --- reliability-weighted penalties ---
            if s0.numel() > 0 and s0.sum() > 0:
                loss_sum = loss_sum + ((1 - fs0) * s0 * (len(s0) / s0.sum())).sum()
                CNT += len(s0)
            if s1.numel() > 0 and s1.sum() > 0:
                loss_sum = loss_sum + ((1 - fs1) * s1 * (len(s1) / s1.sum())).sum()
                CNT += len(s1)

        return loss_sum / max(CNT, 1)

