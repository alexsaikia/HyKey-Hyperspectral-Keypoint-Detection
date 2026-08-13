

# IPCAI 2026: HyKey - Hyperspectral Keypoint Detection and Matching in Minimally Invasive Surgery

**Alexander Saikia**  ·  **Chiara Di Vece**  ·  **Zhehua Mao**  ·  **Sierra Bonilla**  ·  **Chloe He**  ·  **Joao Ramalhinho**  ·  **Tobias Czempiel**  ·  **Sophia Bano**  ·  **Danail Stoyanov**

### [Paper](https://link.springer.com/article/10.1007/s11548-026-03633-z) | [Dataset & Weights](https://doi.org/10.5522/04/32793294)

---

## 📝 Citation

If you use our data for your publication, please cite our work.

The hyperspectral data and release paper:

```bibtex
@article{saikia2026hykey,
  title={HyKey: hyperspectral keypoint detection and matching in minimally invasive surgery},
  author={Saikia, Alexander and Di Vece, Chiara and Mao, Zhehua and Bonilla, Sierra and He, Chloe and Ramalhinho, Joao and Czempiel, Tobias and Bano, Sophia and Stoyanov, Danail},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--9},
  year={2026},
  publisher={Springer}
}
```

The robotic arm acquisition platform for acquiring the data:

```bibtex
@article{saikia2025robotic,
  title={Robotic arm platform for multi-view image acquisition and 3d reconstruction in minimally invasive surgery},
  author={Saikia, Alexander and Di Vece, Chiara and Bonilla, Sierra and He, Chloe and Magbagbeola, Morenike and Mennillo, Laurent and Czempiel, Tobias and Bano, Sophia and Stoyanov, Danail},
  journal={IEEE Robotics and Automation Letters},
  volume={10},
  number={4},
  pages={3174--3181},
  year={2025},
  publisher={IEEE}
}
```

---

## ⚙️ Installation

```bash
git clone https://github.com/alexsaikia/HyKey-Hyperspectral-Keypoint-Detection
cd HyKey-Hyperspectral-Keypoint-Detection
python -m venv .venv && source .venv/bin/activate

# Install PyTorch FIRST, matching your GPU driver's CUDA version. A bare `pip install torch`
# pulls the newest CUDA build, which is often too new for an existing driver (you'll see a
# "CUDA driver too old" error at runtime). Pick the wheel for your CUDA from pytorch.org;
# e.g. for a CUDA 12.1 driver:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Tested with Python 3.11 and PyTorch 2.4 (CUDA 12.1). A GPU is recommended for training; inference runs on CPU. To check your driver's CUDA version run `nvidia-smi` (top-right "CUDA Version").

---

## 💾 Dataset

**The dataset repository was updated on 08/08/26 to make it easier to download and fix a scene labelling issue. If you downloaded the data before this date, the lighting conditions for the scenes are swapped and lap actually means surg and vice versa**

Download the HyKey Dataset and extract it to `data/spectral/`. No preprocessing step is needed; the dataset ships with calibration and poses pre-computed.

> **Download:** [https://doi.org/10.5522/04/32793294](https://doi.org/10.5522/04/32793294) (DOI: `10.5522/04/32793294`)

*Ex-vivo ovine kidneys and livers, laparoscopic and surgical lighting, hemispherical-sweep and trocar trajectories.*

### Data structure

```
data/spectral/
├── calibration/                    # shared camera calibration (ships with the dataset)
│   ├── RGB/calibration_data.xml    # RGB intrinsics + hand-eye extrinsics
│   ├── HSI/calibration_data.xml    # HSI intrinsics + hand-eye extrinsics
│   ├── stereo_calibration.xml      # RGB<->HSI stereo extrinsics
│   └── homographies/               # RGB<->HSI registration homographies
│       ├── H_hsi2rgb.npy
│       └── H_rgb2hsi.npy
├── K4_surg_close_acquisition_2024-08-28_22-48-52/
│   ├── HSI/00000000.npy ...        # raw XIMEA hyperspectral mosaics
│   ├── RGB/00000000.npy ...        # raw RGB frames
│   ├── hsi_poses.csv               # hand-eye-calibrated HSI camera poses
│   ├── rgb_poses.csv               # hand-eye-calibrated RGB camera poses
│   ├── frame_log.csv               # exposure times + raw robot poses
│   ├── white_*.npy                 # radiometric white reference
│   └── dark_*.npy                  # radiometric dark reference
└── HE_calib_*_acquisition_<timestamp>/   # raw ChArUco calibration captures (only needed to recalibrate; see docs/RECALIBRATION.md)
```

### Naming conventions

Folder names: `{organ}_{lighting}_{trajectory}_acquisition_<timestamp>`


| Field          | Tag       | Meaning                                         |
| -------------- | --------- | ----------------------------------------------- |
| **Organ**      | `K1`-`K4` | Ovine kidney, 2024-08-28 (train/val/test)       |
|                | `K5`-`K7` | Ovine kidney, 2024-08-16 (train only)           |
|                | `L1`-`L2` | Ovine liver, 2024-08-28 (train/test)            |
|                | `L3`      | Ovine liver, 2024-08-16 (train only)            |
| **Lighting**   | `lap`     | Laparoscopic light only                         |
|                | `surg`    | Surgical overhead lights                        |
| **Trajectory** | `close`   | Hemispherical sweep, close range (~10-15 cm)    |
|                | `far`     | Hemispherical sweep, extended range (~20-30 cm) |
|                | `trocar`  | Trocar (RCM) trajectory                         |


**Splits** (`configs/test_set.yaml`, `configs/validation_set.yaml`): test = `K4_`* + `L1_*`; val = `K2_*`; K5-K7 and L3 are training only.

---



## 🏋️ Training

```bash
python -m train.train_hykey --config configs/train_hykey.yaml
```

Key options in `configs/train_hykey.yaml`: `spectral_data_root`, epipolar loss weight `w_epi`, `radiometric_calibration`, batch size, epochs.

**Synthetic-warp difficulty.** The planar-homography augmentation strength is controlled by `warp_difficulty` in the `dataset:` block (defaults to `paper`, the setting the released checkpoints were trained/evaluated under):

```yaml
dataset:
  warp_difficulty: paper      # 'paper' (default) | 'harder' | or inline overrides below
```

- Switch presets by name: `warp_difficulty: harder` for the harder post-paper settings.
- Override individual parameters from the config without editing code, e.g.:
  ```yaml
  dataset:
    warp_difficulty: paper
    warp_params: { max_angle: 0.35, scaling_amplitude: 0.08 }   # merged on top of the preset
  ```
- The presets themselves live in `data_processing/utils/augmentation_utils.py` (`WARP_PRESETS`); edit there to redefine `paper`/`current` or add your own. Parameters: `scaling_amplitude`, `perspective_amplitude_x`, `perspective_amplitude_y`, `patch_ratio`, `max_angle`.

---



## 📊 Evaluation

```bash
python evaluate_hykey.py --config configs/evaluate_hykey.yaml
```

Reports planar MMA (pixel thresholds) and relative-pose mAA (degree thresholds). Set checkpoint dir under `evaluation.model_checkpoint_dirs`.



---

## 🔍 Inference

```bash
python infer_pair.py \
    --acquisition ./data/spectral/K4_surg_close_acquisition_2024-08-28_22-48-52 \
    --checkpoint ./checkpoints/hykey \
    --idx 0 --pair planar --output matches.png
```

`--pair planar` uses a synthetic homography warp; `--pair nonplanar` uses a real subsequent frame.



---

## 🔧 Calibration

Pre-computed calibration ships inside the dataset under `data/spectral/calibration/`. **No recalibration step is needed to run the dataset.** It is loaded automatically as `<spectral_data_root>/calibration` (override via the `calibration_dir` config key).

If you want to recalibrate for your own hardware (or reproduce `calibration/` from the raw `HE_calib_*` ChArUco captures shipped with the dataset), see **[docs/RECALIBRATION.md](docs/RECALIBRATION.md)**.

## License

Code, HyKey Dataset, and weights are released under **CC BY-NC 4.0**. See [LICENSE](LICENSE). For commercial use, contact the authors.