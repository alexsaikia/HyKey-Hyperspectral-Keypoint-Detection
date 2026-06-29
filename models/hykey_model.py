import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

from models.kp_head import DKD
from models.blocks import Spectral3DCNN


class HyKeyModel(pl.LightningModule):
    """Base model containing the core HyKey architecture and extraction utilities.
    """

    def __init__(
        self,
        input_channels: int = 16,
        learning_rate: float = 1e-3,
        debug_dir: str = "debug",
        use_identity_homography: bool = False,
        im_size: int = 512,
        c1: int = 16,
        c2: int = 32,
        c3: int = 64,
        dim: int = 128,
        radius: int = 2,
        top_k: int = 400,
        scores_th: float = 0.0,
        n_limit: int = 400,
        temp_det: float = 0.1,
        num_blocks: int = 2,
        # Masking parameters (used in forward)
        mask_min_avg: float = 0.01,
        mask_max_avg: float = 0.95,
        use_max_pooling: bool = True,
    ) -> None:
        super().__init__()

        # Core hyperparameters 
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.debug_dir = debug_dir
        self.use_identity_homography = use_identity_homography
        self.im_size = im_size
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.temp_det = temp_det
        self.num_blocks = num_blocks
        self.use_max_pooling = use_max_pooling

        # Masking parameters
        self.mask_min_avg = mask_min_avg
        self.mask_max_avg = mask_max_avg

        # Initialize architecture and weights
        self.init_architecture(c1, c2, c3, dim)
        self.init_weights()

    def init_architecture(self, c1, c2, c3, dim):
        # 3D CNN architecture
        self.encoder_blocks = nn.ModuleList([
            Spectral3DCNN(1, c1, num_blocks=self.num_blocks, kernel_size=3, stride=(2,1,1)),
            Spectral3DCNN(c1, c2, num_blocks=self.num_blocks, kernel_size=3, stride=(2,1,1)),
            Spectral3DCNN(c2, c3, num_blocks=self.num_blocks, kernel_size=3, stride=(2,1,1)),
        ])

        self.max_pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        # Spectral pooling
        self.spec_pool = nn.AdaptiveAvgPool3d((1, None, None))

        # 2D convolution head
        self.conv_head = nn.Sequential(
            nn.Conv2d(c3+c2+c1, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim+1, kernel_size=3, padding=1),
        )

        # Initialize DKD for keypoint detection
        self.dkd = DKD(radius=self.radius, top_k=self.top_k,
                       scores_th=self.scores_th, n_limit=self.n_limit,
                       temp_det=self.temp_det)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        descriptor_map, score_map = self.extract_dense_map(x)

        # Base validity mask from intensity range
        avg_pixel_values = x.mean(dim=1, keepdim=True)
        valid_mask = (avg_pixel_values >= self.mask_min_avg) & (avg_pixel_values <= self.mask_max_avg)

        score_map = score_map * valid_mask
        descriptor_map = descriptor_map * valid_mask
        ret = {'descriptor_map': descriptor_map,
               'scores_map': score_map}
        if self.training:
            ret['train_mask'] = valid_mask
        return ret

    def extract_dense_map(self, image, ret_dict=False):
        device = image.device
        b, c, h, w = image.shape
        avg_pixel_values = image.mean(dim=1, keepdim=True)
        # Original padding approach using manual concatenation
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
        if h_ != h:
            h_padding = torch.zeros(b, c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)

        x_3d = image.unsqueeze(1)
        feats3d = []

        # Process 3D features
        for block in self.encoder_blocks:
            x_3d = block(x_3d)
            if self.use_max_pooling:
                x_3d = self.max_pool(x_3d)
            feats3d.append(x_3d)

        # 2D conversion
        feats2d = []
        target_size = (image.shape[2], image.shape[3])

        for feat3d in feats3d:
            x_2d = self.spec_pool(feat3d).squeeze(2)
            if x_2d.shape[2:] != target_size:
                x_2d = F.interpolate(x_2d, size=target_size, mode='bilinear', align_corners=True)
            feats2d.append(x_2d)

        # Concatenate and process
        x_2d = torch.cat(feats2d, dim=1)
        x_all = self.conv_head(x_2d)

        score_map = torch.sigmoid(x_all[:, -1, :, :]).unsqueeze(1)
        descriptor_map = x_all[:, :-1, :, :]

        # Unpadding
        if h_ != h or w_ != w:
            descriptor_map = descriptor_map[:, :, :h, :w]
            score_map = score_map[:, :, :h, :w]

        # Normalize descriptors
        descriptor_map = F.normalize(descriptor_map, p=2, dim=1)

        # Base validity mask from intensity range
        valid_mask = (avg_pixel_values >= self.mask_min_avg) & (avg_pixel_values <= self.mask_max_avg)

        score_map = score_map * valid_mask
        descriptor_map = descriptor_map * valid_mask

        if ret_dict:
            return {'descriptors': descriptor_map, 'score_map': score_map}
        else:
            return descriptor_map, score_map

    def extract(self, image):
        descriptor_map, scores_map = self.extract_dense_map(image)
        keypoints, descriptors, kptscores, scoredispersitys = self.dkd(scores_map, descriptor_map)
        return {'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': kptscores,
                'score_dispersity': scoredispersitys,
                'descriptor_map': descriptor_map,
                'scores_map': scores_map}


