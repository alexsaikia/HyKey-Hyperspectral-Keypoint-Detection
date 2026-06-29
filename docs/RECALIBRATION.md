# Recalibration Guide

**You do not need this to use HyKey.** The HyKey Dataset already ships with a
pre-computed `calibration/` folder (intrinsics, hand-eye extrinsics, stereo
extrinsics, and RGB↔HSI homographies) that is loaded automatically as
`<spectral_data_root>/calibration`. Training, evaluation, and inference work
out of the box.

This guide is only for users who want to **reproduce** `calibration/` from the raw ChArUco captures

All code referenced lives in `data_processing/utils/calibrate_utils.py` and
`data_processing/utils/registration_utils.py`. Run snippets from the repo root.

---

## 1. The raw calibration captures (`HE_calib_*`)

The dataset ships the raw ChArUco capture sequences used to produce
`calibration/`. They are **not** used by the data loader (folders starting with
`HE` are skipped) and are only needed for recalibration.

The ChArUco board geometry is encoded in the folder name:

```
HE_calib_<rows>_<cols>_<square_mm>_<marker_mm>_<dict>_acquisition_<timestamp>
```

e.g. `HE_calib_37_37_5_3_4X4_acquisition_2024-08-28_19-18-59` → a 37×37 board,
5 mm squares, 3 mm markers, `DICT_4X4_*`.

The inputs a recalibration actually needs from such a folder are:

| Path | Purpose |
|---|---|
| `HSI/*.npy` | raw XIMEA mosaics of the board |
| `RGB/*.npy` | raw RGB frames of the board |
| `frame_log.csv` | robot end-effector poses + exposures (per frame) |

With `hand_eye=True` the per-camera frame subsets `RGB/used_frame_log.csv` and
`HSI/used_frame_log.csv` are derived from `frame_log.csv` during the run (by
`save_frame_log`), so you do not need to provide them yourself.

> The released `HE_calib_*` folders also ship intermediate artifacts left over
> from the original processing: PNG dumps (`RGB/images/`, `RGB/undistorted/`,
> `HSI/labels/`, `HSI/ground_truth_features/`), a top-level `used_frame_log.csv`,
> and (in one folder) an `old_hand_eye/` subfolder with per-band preview videos
> and a `calibration_data_546nm.xml`. They can be ignored: `import_imgs` loads
> only the `*.npy` frames, so recalibration uses just the inputs above.

---

## 2. Intrinsics + hand-eye + stereo

`CameraCalibration.calibrate_dataset()` detects the board in both cameras, runs
`cv2.calibrateCamera` per camera, optionally solves the eye-in-hand extrinsics,
and runs stereo calibration. It writes three files **into the capture folder**:
`RGB/calibration_data.xml`, `HSI/calibration_data.xml`, and
`stereo_calibration.xml`.

```python
import cv2
from data_processing.utils.calibrate_utils import CameraCalibration

# Board geometry must match the capture (see the folder name).
charuco_params = {
    "rows": 37, "cols": 37,
    "square_length": 0.005,   # metres (5 mm)
    "marker_length": 0.003,   # metres (3 mm)
    "dictionary": cv2.aruco.DICT_4X4_1000,
}

calib = CameraCalibration(charuco_params=charuco_params)
calib.calibrate_dataset(
    "data/spectral/HE_calib_37_37_5_3_4X4_acquisition_2024-08-28_19-18-59",
    hand_eye=True,   # also solve eye-in-hand extrinsics (needs used_frame_log.csv)
    video=False,
)
```

Each `calibration_data.xml` stores the camera matrix, distortion coefficients,
reprojection error, and (with `hand_eye=True`) the hand-eye rotation/translation,
exactly the fields `SpectralDataset` reads.

---

## 3. RGB↔HSI registration homographies

The registration homographies are derived from the two camera intrinsics (the
relative aspect-ratio scaling), so once the `calibration_data.xml` files exist
you can regenerate them with `register_intrinsic`:

```python
import numpy as np
from data_processing.utils.calibrate_utils import load_calibration_data
from data_processing.utils.registration_utils import register_intrinsic

rgb_calib = load_calibration_data("RGB/calibration_data.xml")
hsi_calib = load_calibration_data("HSI/calibration_data.xml")

# Any representative undistortable RGB/HSI frame pair; H depends on the
# intrinsics, the frames are only used for the optional warp preview.
_, _, H_hsi2rgb, H_rgb2hsi = register_intrinsic(
    rgb_img, hsi_img, rgb_calib, hsi_calib, target_space="hsi"
)
np.save("H_hsi2rgb.npy", H_hsi2rgb)
np.save("H_rgb2hsi.npy", H_rgb2hsi)
```

---

## 4. Assemble the `calibration/` folder

Place the outputs so they match the layout the loader expects under
`<spectral_data_root>/calibration/`:

```
calibration/
├── RGB/calibration_data.xml
├── HSI/calibration_data.xml
├── stereo_calibration.xml
└── homographies/
    ├── H_hsi2rgb.npy
    └── H_rgb2hsi.npy
```

That's it. `SpectralDataset` will auto-discover this folder (or point at it
explicitly with the `calibration_dir` config key).
