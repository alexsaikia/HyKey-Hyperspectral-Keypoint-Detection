import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from data_processing.utils.registration_utils import *
from data_processing.utils.hsi_utils import *
from data_processing.utils.utils import load_robot_kinematic_data, quaternion_to_rotation_matrix, import_imgs, rotation_matrix_to_quaternion

# Constants
DETECTION_THRESHOLD = 4  # Minimum number of corners to detect in an image
SKIP = 150 # Skip every x-1 images
VIDEO_FPS = 15.0  # Frame rate for output videos
VIDEO_CODEC = 'XVID'  # Video codec for output files
SUBPIXEL_WINDOW_SIZE_RGB = (11, 11)  # Window size for RGB subpixel refinement
SUBPIXEL_WINDOW_SIZE_HSI = (5, 5)   # Window size for HSI subpixel refinement
SUBPIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
RESIZE_FACTOR = 0.6  # Factor for resizing display images
HSI_CALIBRATION_WAVELENGTH = 546  # Wavelength (nm) used for HSI calibration
# Band index in the wavelength-sorted demosaic output closest to 546 nm (~545.1 nm, index 9).
HSI_CALIBRATION_BAND_IDX = 9
HANDEYE_CALIBRATION_METHOD = cv2.CALIB_HAND_EYE_DANIILIDIS

# Default Charuco board parameters
DEFAULT_CHARUCO_PARAMS = {
    "rows": 37,
    "cols": 37,
    "square_length": 0.005,
    "marker_length": 0.003,
    "dictionary": cv2.aruco.DICT_4X4_1000
}

class CameraCalibration:
    def __init__(self, charuco_params=None, detector_params=None):
        self.charuco_params = charuco_params or DEFAULT_CHARUCO_PARAMS
        self.charuco_board, self.charuco_detector = self.create_charuco(**self.charuco_params)
        self.detector_params = detector_params if detector_params else cv2.aruco.DetectorParameters()
        
    def apply_hand_eye_transform(self, frame_log_path, hand_eye_rotation, hand_eye_translation, output_csv_path):
        T_hand_to_rgb = np.eye(4)
        T_hand_to_rgb[:3, :3] = hand_eye_rotation
        T_hand_to_rgb[:3, 3] = hand_eye_translation.flatten()
        T_hand_to_rgb = np.linalg.inv(T_hand_to_rgb)

        df = pd.read_csv(frame_log_path)

        transformed_positions = []
        transformed_orientations = []
        transformation_matrices = []

        for _, row in df.iterrows():
            position = np.array([row['link_position_x'], row['link_position_y'], row['link_position_z']])
            quaternion = np.array([row['link_orientation_w'], row['link_orientation_x'], row['link_orientation_y'], row['link_orientation_z']])

            rotation_matrix = quaternion_to_rotation_matrix(quaternion)
            robot_pose_matrix = np.eye(4)
            robot_pose_matrix[:3, :3] = rotation_matrix
            robot_pose_matrix[:3, 3] = position
            robot_pose_matrix_inv = np.linalg.inv(robot_pose_matrix)
            transformed_pose_matrix_local = T_hand_to_rgb @ robot_pose_matrix_inv
            transformed_pose_matrix_world = np.linalg.inv(transformed_pose_matrix_local)
            transformed_position = transformed_pose_matrix_world[:3, 3]
            transformed_rotation = transformed_pose_matrix_world[:3, :3]
            transformed_quaternion = rotation_matrix_to_quaternion(transformed_rotation)

            transformation_matrix_flat = transformed_pose_matrix_world.flatten()

            transformed_positions.append(transformed_position)
            transformed_orientations.append(transformed_quaternion)
            transformation_matrices.append(transformation_matrix_flat)

        transformed_df = pd.DataFrame({
            'frame_id': [i for i in range(len(transformed_positions))],
            'link_position_x': [pos[0] for pos in transformed_positions],
            'link_position_y': [pos[1] for pos in transformed_positions],
            'link_position_z': [pos[2] for pos in transformed_positions],
            'link_orientation_w': [quat[0] for quat in transformed_orientations],
            'link_orientation_x': [quat[1] for quat in transformed_orientations],
            'link_orientation_y': [quat[2] for quat in transformed_orientations],
            'link_orientation_z': [quat[3] for quat in transformed_orientations],
            'transformation_matrix': transformation_matrices
        })

        transformed_df.to_csv(output_csv_path, index=False)
        print(f"Transformed poses saved to {output_csv_path}")
    
    @staticmethod
    def create_charuco(rows, cols, square_length, marker_length, dictionary):
        board = cv2.aruco.CharucoBoard(
            (cols, rows),
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=cv2.aruco.getPredefinedDictionary(dictionary)
        )
        detector = cv2.aruco.CharucoDetector(board)
        return board, detector

    def calibrate_camera(self, all_obj_points, all_img_points, out_path, img_size):
        if len(all_img_points) == 0:
            raise ValueError('No corners found in any image. Calibration failed.')
            
        flags = None
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            all_obj_points, 
            all_img_points, 
            img_size,
            None, 
            None,
            flags=flags
        )
        print('Calibration complete.')
        print(f'Reprojection error: {ret}')
        self.save_camera_params(ret, mtx, dist, rvecs, tvecs, out_path)
        
        return {
            "camera_matrix": mtx, 
            "dist_coeffs": dist, 
            "rvecs": rvecs, 
            "tvecs": tvecs,
            "reprojection_error": ret
        }

    def calibrate_stereo(self, rgb_data, hsi_data, common_object_points):
        common_image_points_rgb = rgb_data["common_image_points_rgb"]
        common_image_points_hsi = hsi_data["common_image_points_hsi"]

        n_points = len(common_object_points)
        print(f"Found {n_points} common point sets for stereo calibration")

        if n_points < 4:
            raise ValueError("Not enough common points found for stereo calibration")

        obj_points = [np.array(pts, dtype=np.float32).reshape(-1, 3) for pts in common_object_points]
        img_points_rgb = [np.array(pts, dtype=np.float32).reshape(-1, 2) for pts in common_image_points_rgb]
        img_points_hsi = [np.array(pts, dtype=np.float32).reshape(-1, 2) for pts in common_image_points_hsi]

        valid_indices = [
            i for i in range(len(obj_points))
            if len(obj_points[i]) >= 4 and len(img_points_rgb[i]) >= 4 and len(img_points_hsi[i]) >= 4
        ]

        if not valid_indices:
            raise ValueError("No valid point sets found with at least 4 points each")

        obj_points = [obj_points[i] for i in valid_indices]
        img_points_rgb = [img_points_rgb[i] for i in valid_indices]
        img_points_hsi = [img_points_hsi[i] for i in valid_indices]

        print(f"Using {len(valid_indices)} valid point sets for stereo calibration")

        flags = (cv2.CALIB_FIX_INTRINSIC +
                cv2.CALIB_USE_INTRINSIC_GUESS +
                cv2.CALIB_FIX_PRINCIPAL_POINT +
                cv2.CALIB_FIX_FOCAL_LENGTH)

        ret, rgb_mtx, rgb_dist, hsi_mtx, hsi_dist, R, T, E, F = cv2.stereoCalibrate(
            objectPoints=obj_points,
            imagePoints1=img_points_rgb,
            imagePoints2=img_points_hsi,
            cameraMatrix1=rgb_data["camera_matrix"],
            distCoeffs1=rgb_data["dist_coeffs"],
            cameraMatrix2=hsi_data["camera_matrix"],
            distCoeffs2=hsi_data["dist_coeffs"],
            imageSize=rgb_data["image_size"],
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        )

        print(f'Stereo calibration complete. RMS error: {ret}')

        return {
            "rotation": R,
            "translation": T,
            "essential": E,
            "fundamental": F,
            "error": ret
        }

    def calibrate_dataset(self, path, hand_eye=False, video=True):
        path = Path(path)
        hsi_path = path / 'HSI'
        rgb_path = path / 'RGB'
        frame_log_path = path / 'frame_log.csv'

        print("\nLoading RGB images...")
        rgb_images = import_imgs(str(rgb_path), True)

        print("\nLoading and de-mosaicing HSI images...")
        hsi_raw_images = import_imgs(str(hsi_path), False)
        demosaic_images = np.zeros((len(hsi_raw_images), 272, 512, 16), dtype=np.float32)
        for i, img in enumerate(tqdm(hsi_raw_images, desc='De-mosaicing images')):
            demosaic_images[i], _ = demosaic_aligned_ximea(img)
        lo = demosaic_images.min(axis=(0, 1, 2), keepdims=True)
        hi = demosaic_images.max(axis=(0, 1, 2), keepdims=True)
        demosaic_images = ((demosaic_images - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
        hsi_images = demosaic_images[:, :, :, HSI_CALIBRATION_BAND_IDX]
        
        print("\nCalibrating RGB camera...")
        rgb_calib_path = rgb_path / 'calibration_data.xml'
        rgb_video_path = rgb_path / 'calibration_video.avi' if video else None

        print("\nCalibrating HSI camera...")
        hsi_calib_path = hsi_path / 'calibration_data.xml'
        hsi_video_path = hsi_path / 'calibration_video.avi' if video else None
        
        rgb_data, hsi_data, common_object_points = self.calibrate_charuco(
            rgb_images=rgb_images,
            hsi_images=hsi_images,
            out_path_rgb=str(rgb_calib_path),
            out_path_hsi=str(hsi_calib_path),
            video_path_rgb=str(rgb_video_path) if video else None,
            video_path_hsi=str(hsi_video_path) if video else None,
            hand_eye=hand_eye,
            show_image=False
        )
        
        if hand_eye:
            rgb_robot_data = load_robot_kinematic_data(str(rgb_path / 'used_frame_log.csv'))

            print("\nPerforming RGB hand-eye calibration...")
            R_rgb_to_hand, t_rgb_to_hand = self.perform_eye_in_hand_calibration(rgb_robot_data, rgb_data)
            self.save_camera_params(
                rgb_data["reprojection_error"],
                rgb_data["camera_matrix"],
                rgb_data["dist_coeffs"],
                rgb_data["rvecs"],
                rgb_data["tvecs"],
                str(rgb_calib_path),
                R_rgb_to_hand,
                t_rgb_to_hand
            )

            hsi_robot_data = load_robot_kinematic_data(str(hsi_path / 'used_frame_log.csv'))

            print("\nPerforming HSI hand-eye calibration...")
            R_hsi_to_hand, t_hsi_to_hand = self.perform_eye_in_hand_calibration(hsi_robot_data, hsi_data)
            self.save_camera_params(
                hsi_data["reprojection_error"],
                hsi_data["camera_matrix"],
                hsi_data["dist_coeffs"],
                hsi_data["rvecs"],
                hsi_data["tvecs"],
                str(hsi_calib_path),
                R_hsi_to_hand,
                t_hsi_to_hand
            )

            self.apply_hand_eye_transform(
                str(frame_log_path), R_rgb_to_hand, t_rgb_to_hand, str(path / 'rgb_poses.csv')
            )
            self.apply_hand_eye_transform(
                str(frame_log_path), R_hsi_to_hand, t_hsi_to_hand, str(path / 'hsi_poses.csv')
            )

        print("\nPerforming stereo calibration...")
        stereo_results = self.calibrate_stereo(rgb_data, hsi_data, common_object_points)

        stereo_calib_path = path / 'stereo_calibration.xml'
        cv_file = cv2.FileStorage(str(stereo_calib_path), cv2.FILE_STORAGE_WRITE)
        cv_file.write("stereo_rotation", stereo_results["rotation"])
        cv_file.write("stereo_translation", stereo_results["translation"])
        cv_file.write("essential_matrix", stereo_results["essential"])
        cv_file.write("fundamental_matrix", stereo_results["fundamental"])
        cv_file.write("rms_error", stereo_results["error"])
        cv_file.release()
        
        print("\nCalibration complete!")
        print(f"RGB Camera RMS error: {rgb_data['reprojection_error']:.6f}")
        print(f"HSI Camera RMS error: {hsi_data['reprojection_error']:.6f}")
        print(f"Stereo Calibration RMS error: {stereo_results['error']:.6f}")
        
        return rgb_data, hsi_data, stereo_results

    def calibrate_charuco(self, rgb_images, hsi_images, out_path_rgb, out_path_hsi, video_path_rgb=None, video_path_hsi=None, hand_eye=False, show_image=False):
        count_rgb = 0
        count_hsi = 0
        all_obj_points_rgb = []
        all_img_points_rgb = []
        all_obj_points_hsi = []
        all_img_points_hsi = []
        used_frame_indices_rgb = []
        used_frame_indices_hsi = []
        common_object_points = []
        common_image_points_rgb = []
        common_image_points_hsi = []
        
        video_writer_rgb = None
        video_writer_hsi = None
        if video_path_rgb:
            height_rgb, width_rgb = rgb_images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            is_color = not video_path_rgb.endswith('nm.avi')
            video_writer_rgb = cv2.VideoWriter(video_path_rgb, fourcc, VIDEO_FPS, (width_rgb, height_rgb), isColor=is_color)
        if video_path_hsi:
            height_hsi, width_hsi = hsi_images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            # HSI calibration frames are single-channel, so the writer is always greyscale.
            video_writer_hsi = cv2.VideoWriter(video_path_hsi, fourcc, VIDEO_FPS, (width_hsi, height_hsi), isColor=False)

        try:
            for idx in range(min(len(rgb_images), len(hsi_images))):
                print(f"Processing frame {idx}")
                if idx % SKIP != 0:
                    continue
                rgb_img = rgb_images[idx]
                hsi_img = hsi_images[idx]
                

                rgb_processed_img, rgb_charuco_corners, rgb_charuco_ids, rgb_marker_corners, rgb_marker_ids = self.process_rgb_frame(rgb_img)
                hsi_processed_img, hsi_charuco_corners, hsi_charuco_ids, hsi_marker_corners, hsi_marker_ids = self.process_hsi_frame(hsi_img)

                if rgb_charuco_corners is not None and rgb_marker_corners is not None:
                    rgb_processed_img = self.draw_detections(rgb_processed_img, rgb_charuco_corners, rgb_charuco_ids, rgb_marker_corners, rgb_marker_ids)

                if hsi_charuco_corners is not None and hsi_marker_corners is not None:
                    hsi_processed_img = self.draw_detections(hsi_processed_img, hsi_charuco_corners, hsi_charuco_ids, hsi_marker_corners, hsi_marker_ids)

                self.display_image(rgb_processed_img, video_path_rgb, video_writer_rgb, show_image)
                self.display_image(hsi_processed_img, video_path_hsi, video_writer_hsi, show_image)

                if rgb_charuco_corners is not None and len(rgb_charuco_corners) > 8:
                    obj_point_rgb, img_point_rgb = self.charuco_board.matchImagePoints(rgb_charuco_corners, rgb_charuco_ids)
                    if len(obj_point_rgb) > DETECTION_THRESHOLD:
                        used_frame_indices_rgb.append(idx)
                        all_obj_points_rgb.append(obj_point_rgb)
                        all_img_points_rgb.append(img_point_rgb)
                        count_rgb += 1
                else:
                    obj_point_rgb = []
                    img_point_rgb = []

                if hsi_charuco_corners is not None and len(hsi_charuco_corners) > 8:
                    obj_point_hsi, img_point_hsi = self.charuco_board.matchImagePoints(hsi_charuco_corners, hsi_charuco_ids)
                    if len(obj_point_hsi) > DETECTION_THRESHOLD:
                        used_frame_indices_hsi.append(idx)
                        all_obj_points_hsi.append(obj_point_hsi)
                        all_img_points_hsi.append(img_point_hsi)
                        count_hsi += 1
                else:
                    obj_point_hsi = []
                    img_point_hsi = []
                        
                if len(obj_point_rgb) > DETECTION_THRESHOLD and len(obj_point_hsi) > DETECTION_THRESHOLD and rgb_charuco_corners is not None and hsi_charuco_corners is not None and len(rgb_charuco_corners) > 8 and len(hsi_charuco_corners) > 8:
                    obj_points_rgb_arr = np.array(obj_point_rgb)
                    obj_points_hsi_arr = np.array(obj_point_hsi)
                    img_points_rgb_arr = np.array(img_point_rgb)
                    img_points_hsi_arr = np.array(img_point_hsi)

                    frame_obj_points = []
                    frame_img_points_rgb = []
                    frame_img_points_hsi = []

                    for i, rgb_obj in enumerate(obj_points_rgb_arr):
                        matches = np.all(np.abs(obj_points_hsi_arr - rgb_obj) < 1e-6, axis=2)
                        if np.any(matches):
                            match_idx = np.where(matches)[0][0]
                            frame_obj_points.append(rgb_obj)
                            frame_img_points_rgb.append(img_points_rgb_arr[i])
                            frame_img_points_hsi.append(img_points_hsi_arr[match_idx])

                    common_object_points.append(frame_obj_points)
                    common_image_points_rgb.append(frame_img_points_rgb)
                    common_image_points_hsi.append(frame_img_points_hsi)
                    
                    print(f"Frame {idx}: Found {len(frame_obj_points)} common points")
                                
        finally:
            if video_writer_rgb:
                video_writer_rgb.release()
            if video_writer_hsi:
                video_writer_hsi.release()
            if show_image:
                cv2.destroyAllWindows()

        if hand_eye:
            self.save_frame_log(used_frame_indices_rgb, out_path_rgb)
            self.save_frame_log(used_frame_indices_hsi, out_path_hsi)

        height_rgb, width_rgb = rgb_images[0].shape
        calib_results_rgb = self.calibrate_camera(all_obj_points_rgb, all_img_points_rgb, out_path_rgb, (width_rgb, height_rgb))
        height_hsi, width_hsi = hsi_images[0].shape
        calib_results_hsi = self.calibrate_camera(all_obj_points_hsi, all_img_points_hsi, out_path_hsi, (width_hsi, height_hsi))

        calib_results_rgb.update({
            "object_points": all_obj_points_rgb,
            "image_points": all_img_points_rgb,
            "image_size": (width_rgb, height_rgb),
            "reprojection_error": calib_results_rgb["reprojection_error"],
            "common_image_points_rgb": common_image_points_rgb
        })
        
        calib_results_hsi.update({
            "object_points": all_obj_points_hsi,
            "image_points": all_img_points_hsi,
            "image_size": (width_hsi, height_hsi),
            "reprojection_error": calib_results_hsi["reprojection_error"],
            "common_image_points_hsi": common_image_points_hsi
        })

        return calib_results_rgb, calib_results_hsi, common_object_points


    def process_rgb_frame(self, img):
        """Process RGB camera frame for calibration."""
        img = cv2.flip(cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_BayerBG2BGR), 0)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(grey_img)
        
        if charuco_corners is not None and len(charuco_corners) > 0:
            charuco_corners = cv2.cornerSubPix(grey_img, charuco_corners, 
                                             SUBPIXEL_WINDOW_SIZE_RGB, (-1, -1), 
                                             SUBPIXEL_CRITERIA)
        
        return img, charuco_corners, charuco_ids, marker_corners, marker_ids

    def process_hsi_frame(self, img):
        """Process HSI camera frame for calibration."""
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(img)
        
        if charuco_corners is not None and len(charuco_corners) > 0:
            charuco_corners = cv2.cornerSubPix(img, charuco_corners, 
                                             SUBPIXEL_WINDOW_SIZE_HSI, (-1, -1), 
                                             SUBPIXEL_CRITERIA)
        
        return img, charuco_corners, charuco_ids, marker_corners, marker_ids

    def draw_detections(self, img, charuco_corners, charuco_ids, marker_corners, marker_ids):
        """Draw detected Charuco corners and markers on the image."""
        img = cv2.aruco.drawDetectedCornersCharuco(
            image=cv2.UMat(img), charucoCorners=charuco_corners, charucoIds=charuco_ids)
        img = cv2.aruco.drawDetectedMarkers(image=img, corners=marker_corners, ids=marker_ids)
        return img

    def save_frame_log(self, used_frame_indices, out_path):
        calib_dir = Path(out_path).parent
        new_frame_log_path = calib_dir / 'used_frame_log.csv'
        original_frame_log = calib_dir.parent / 'frame_log.csv'
        
        self.save_new_frame_log(original_frame_log, new_frame_log_path, used_frame_indices)

    def display_image(self, img, video_path, video_writer, show_image=False):
        if show_image:
            rs_img = cv2.resize(img, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            cv2.imshow('Charuco board', rs_img)
            cv2.waitKey(10)
        if video_writer:
            video_writer.write(img)

    def save_new_frame_log(self, original_frame_log, new_frame_log_path, used_frame_indices):
        try:
            df = pd.read_csv(original_frame_log)
            used_frames_df = df.iloc[used_frame_indices]
            used_frames_df.to_csv(new_frame_log_path, index=False)
            print(f"Saved new frame log with {len(used_frame_indices)} frames to {new_frame_log_path}")
        except Exception as e:
            print(f"Error saving new frame log: {e}")

    @staticmethod
    def save_camera_params(ret, mtx, dist, rvecs, tvecs, out_path, hand_eye_rotation=None, hand_eye_translation=None):
        try:
            cv_file = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
            cv_file.write("reprojection_error", ret)
            cv_file.write("camera_matrix", mtx)
            cv_file.write("dist_coeff", dist)
            cv_file.write("rvecs", np.array(rvecs))
            cv_file.write("tvecs", np.array(tvecs))
            
            if hand_eye_rotation is not None and hand_eye_translation is not None:
                cv_file.write("hand_eye_rotation", hand_eye_rotation)
                cv_file.write("hand_eye_translation", hand_eye_translation)
                
            cv_file.release()
            print(f"Saved camera parameters to {out_path}")
            
        except Exception as e:
            print(f"Error saving camera parameters: {e}")
            raise

    def perform_eye_in_hand_calibration(self, robot_data, camera_data, method=cv2.CALIB_HAND_EYE_DANIILIDIS):
        robot_translations = robot_data["translations"]
        robot_rotations = robot_data["rotations"]
        camera_rvecs = camera_data["rvecs"]
        camera_tvecs = camera_data["tvecs"]

        camera_rot_matrices = [cv2.Rodrigues(rot)[0] for rot in camera_rvecs]

        robot_transformations = []
        camera_transformations = []

        for i in range(len(robot_rotations)):
            robot_matrix = np.eye(4)
            robot_matrix[:3, :3] = robot_rotations[i]
            robot_matrix[:3, 3] = robot_translations[i]
            robot_transformations.append(robot_matrix)

            camera_matrix = np.eye(4)
            camera_matrix[:3, :3] = camera_rot_matrices[i]
            camera_matrix[:3, 3] = camera_tvecs[i].flatten()
            camera_transformations.append(camera_matrix)

        robot_rotations = [trans[:3, :3] for trans in robot_transformations]
        robot_translations = [trans[:3, 3] for trans in robot_transformations]
        camera_rot_matrices = [trans[:3, :3] for trans in camera_transformations]
        camera_tvecs = [trans[:3, 3] for trans in camera_transformations]

        R_eye_to_hand, t_eye_to_hand = cv2.calibrateHandEye(
            R_gripper2base=robot_rotations,
            t_gripper2base=robot_translations,
            R_target2cam=camera_rot_matrices,
            t_target2cam=camera_tvecs,
            method=method
        )

        return R_eye_to_hand, t_eye_to_hand

    def register_image_pair(self, rgb_img, hsi_img, rgb_calib, hsi_calib,
                            target_space='rgb', save_dir=None):
        """Compute and optionally save the RGB<->HSI registration homography."""
        if save_dir is not None:
            homography_dir = Path(save_dir) / 'homographies'
            homography_dir.mkdir(exist_ok=True, parents=True)

        reg_hsi, reg_rgb, H_hsi2rgb, H_rgb2hsi = register_intrinsic(
            rgb_img, hsi_img, rgb_calib, hsi_calib, target_space)

        if save_dir is not None:
            np.save(homography_dir / 'H_hsi2rgb.npy', H_hsi2rgb)
            np.save(homography_dir / 'H_rgb2hsi.npy', H_rgb2hsi)

        return reg_hsi, reg_rgb


    def process_calibration(self, calib_dir):
        rgb_calib_file = calib_dir / 'RGB' / 'calibration_data.xml'
        hsi_calib_file = calib_dir / 'HSI' / 'calibration_data.xml'

        if not all(f.exists() for f in [rgb_calib_file, hsi_calib_file]):
            print("\nStarting calibration process...")
            self.calibrate_dataset(path=calib_dir, hand_eye=True, video=True)

        print("\nLoading calibration data...")
        rgb_calib = load_calibration_data(rgb_calib_file)
        hsi_calib = load_calibration_data(hsi_calib_file)

        return rgb_calib, hsi_calib

def load_calibration_data(file_path):
        """Load camera calibration data from an OpenCV XML file."""
        cv_file = cv2.FileStorage(str(file_path), cv2.FILE_STORAGE_READ)

        if not cv_file.isOpened():
            raise FileNotFoundError(f"File {file_path} could not be opened.")

        try:
            reprojection_error = cv_file.getNode("reprojection_error").real()
            camera_matrix = cv_file.getNode("camera_matrix").mat()
            dist_coeffs = cv_file.getNode("dist_coeff").mat()
            rvecs = cv_file.getNode("rvecs").mat()
            tvecs = cv_file.getNode("tvecs").mat()

            if any(x is None for x in [camera_matrix, dist_coeffs, rvecs, tvecs]):
                raise ValueError(f"Required calibration data missing from {file_path}")

            hand_eye_rotation = cv_file.getNode("hand_eye_rotation").mat()
            hand_eye_translation = cv_file.getNode("hand_eye_translation").mat()

            rvecs = [rvecs[i] for i in range(rvecs.shape[0])]
            tvecs = [tvecs[i] for i in range(tvecs.shape[0])]

            return {
                "reprojection_error": reprojection_error,
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "rvecs": rvecs,
                "tvecs": tvecs,
                "hand_eye_rotation": hand_eye_rotation,
                "hand_eye_translation": hand_eye_translation
            }

        finally:
            cv_file.release()

def main():
    try:
        calibration = CameraCalibration()
        calib_dir = Path(__file__).resolve().parents[2] / "calibration"
        rgb_calib, hsi_calib = calibration.process_calibration(calib_dir)

        rgb_files = sorted(list((calib_dir / 'RGB').glob('*.npy')))
        hsi_files = sorted(list((calib_dir / 'HSI').glob('*.npy')))
        if not rgb_files or not hsi_files:
            print("No calibration images found.")
            return 1

        frame_num = min(1500, len(rgb_files) - 1)
        rgb_img = np.load(str(rgb_files[frame_num])).astype(np.float32) / 255.0
        hsi_raw, _ = demosaic_aligned_ximea(np.load(str(hsi_files[frame_num])))
        hsi_img = hsi_raw[:, :, HSI_CALIBRATION_BAND_IDX].astype(np.float32)

        print("\nPerforming registration...")
        calibration.register_image_pair(
            rgb_img, hsi_img, rgb_calib, hsi_calib,
            target_space='rgb', save_dir=str(calib_dir))
        print(f"Homographies saved to {calib_dir}/homographies/")

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())

