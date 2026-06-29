import cv2
import numpy as np
import os
import pandas as pd


def import_imgs(folder, normalise=False):
    """Import images from a folder of .npy files."""
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        if not filename.endswith(".npy"):
            continue
        img = np.load(os.path.join(folder, filename))
        if img is not None:
            if normalise:
                if img.dtype == np.uint16:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            images.append(img)
    if images == []:
        print("No images found in the specified folder.")
    return images


def load_robot_kinematic_data(file_path):
    """Load robot kinematic data from a CSV file."""
    df = pd.read_csv(file_path)
    translations = []
    rotations = []
    for idx, row in enumerate(df.iterrows()):
        translation = np.array([row[1]['link_position_x'], row[1]['link_position_y'], row[1]['link_position_z']])
        translations.append(translation)
        quaternion = np.array([row[1]['link_orientation_w'], row[1]['link_orientation_x'], row[1]['link_orientation_y'], row[1]['link_orientation_z']])
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        rotations.append(rotation_matrix)
    return {"translations": translations, "rotations": rotations}


def quaternion_to_rotation_matrix(quaternion):
    """Convert a quaternion in [w, x, y, z] order to a rotation matrix."""
    w, x, y, z = quaternion
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,       2*x*z + 2*y*w      ],
        [2*x*y + 2*z*w,        1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w      ],
        [2*x*z - 2*y*w,        2*y*z + 2*x*w,         1 - 2*x**2 - 2*y**2],
    ])


def rotation_matrix_to_quaternion(rotation_matrix):
    """Convert rotation matrix to quaternion [qw, qx, qy, qz]."""
    qw = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2.0
    qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
    qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
    qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)
    return np.array([qw, qx, qy, qz])


def undistort_image_with_K(img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, alpha: float = 0.0):
    """Undistort an image and return (undistorted_image, new_camera_matrix)."""
    dist_coeffs = np.asarray(dist_coeffs).flatten()
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha)
    if img.ndim == 3 and img.shape[2] > 1:
        undist = np.zeros_like(img)
        for i in range(img.shape[2]):
            undist[:, :, i] = cv2.undistort(img[:, :, i], camera_matrix, dist_coeffs, None, newK)
    else:
        undist = cv2.undistort(img, camera_matrix, dist_coeffs, None, newK)
    return undist, newK
