import torch
import numpy as np
import cv2
from copy import deepcopy


class EmptyTensorError(Exception):
    pass

def mutual_argmax(value, mask=None, as_tuple=True):
    """
    Args:
        value: MxN
        mask:  MxN

    Returns:

    """
    value = value - value.min()  # convert to non-negative tensor
    if mask is not None:
        value = value * mask

    max0 = value.max(dim=1, keepdim=True)  # the col index the max value in each row
    max1 = value.max(dim=0, keepdim=True)

    valid_max0 = value == max0[0]
    valid_max1 = value == max1[0]

    mutual = valid_max0 * valid_max1
    if mask is not None:
        mutual = mutual * mask

    return mutual.nonzero(as_tuple=as_tuple)


def mutual_argmin(value, mask=None):
    return mutual_argmax(-value, mask)


def compute_keypoints_distance(kpts0, kpts1, p=2):
    """
    Args:
        kpts0: torch.tensor [M,2]
        kpts1: torch.tensor [N,2]
        p: (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm

    Returns:
        dist, torch.tensor [N,M]
    """
    dist = kpts0[:, None, :] - kpts1[None, :, :]  # [M,N,2]
    dist = torch.norm(dist, p=p, dim=2)  # [M,N]
    return dist


def plot_keypoints(image, kpts, radius=2, color=(255, 0, 0)):
    image = image.cpu().detach().numpy() if isinstance(image, torch.Tensor) else image
    kpts = kpts.cpu().detach().numpy() if isinstance(kpts, torch.Tensor) else kpts

    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 16:
        image = image[:,:,[-1,8,0]]

    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    for kp in kpts:
        x0, y0 = int(kp[1]), int(kp[0])
        cv2.drawMarker(out, (x0, y0), color, cv2.MARKER_CROSS, radius)

        # cv2.circle(out, (x0, y0), radius, color, -1, lineType=cv2.LINE_4)
    return out


def to_homogeneous(kpts):
    '''
    :param kpts: Nx2
    :return: Nx3
    '''
    ones = kpts.new_ones([kpts.shape[0], 1])
    return torch.cat((kpts, ones), dim=1)


def warp_homography(kpts0, params, border = None):
    '''
    :param kpts: Nx2
    :param homography_matrix: 3x3
    :return:
    '''
    homography_matrix = params['homography_matrix']
    w, h = params['width'], params['height']
    kpts0_homogeneous = to_homogeneous(kpts0)
    kpts01_homogeneous = torch.einsum('ij,kj->ki', homography_matrix, kpts0_homogeneous)
    kpts01 = kpts01_homogeneous[:, :2] / kpts01_homogeneous[:, 2:]

    # kpts01_ = kpts01.detach()
    # due to float coordinates, the upper boundary should be (w-1) and (h-1).
    # For example, if the image size is 480, then the coordinates should in [0~470].
    # 470.5 is not acceptable.
    if border is not None:
        valid01 = (kpts01[:, 0] >= border) * (kpts01[:, 0] <= w - 1 - border) * (kpts01[:, 1] >= border) * (kpts01[:, 1] <= h - 1 - border)
    else:
        valid01 = (kpts01[:, 0] >= 0) * (kpts01[:, 0] <= w - 1) * (kpts01[:, 1] >= 0) * (kpts01[:, 1] <= h - 1)
    kpts0_valid = kpts0[valid01]
    kpts01_valid = kpts01[valid01]
    ids = torch.nonzero(valid01, as_tuple=False)[:, 0]
    ids_out = torch.nonzero(~valid01, as_tuple=False)[:, 0]

    # kpts0_valid: valid keypoints0, the invalid and inconsistance keypoints are removed
    # kpts01_valid: the warped valid keypoints0
    # ids: the valid indices
    return kpts0_valid, kpts01_valid, ids, ids_out


def warp(kpts0, params: dict):
    mode = params['mode']
    if mode == 'homo':
        return warp_homography(kpts0, params)
    else:
        raise ValueError('unknown mode!')


def sample_homography(
        height,
        width,
        shift=0.0,
        perspective=True,
        scaling=True,
        rotation=True,
        translation=True,
        n_scales=5,
        n_angles=25,
        scaling_amplitude=0.2,
        perspective_amplitude_x=0.2,
        perspective_amplitude_y=0.2,
        patch_ratio=0.85,
        max_angle=1.57,
        allow_artifacts=True,
        translation_overflow=0.0,
        random_state=None):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        height, width: Dimensions of the image.
        shift: Shift parameter for homography computation.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.
        random_state: RNG seed or instance for reproducibility.

    Returns:
        A numpy array of shape (3,3) corresponding to the homography transform.
    """
    # RNG setup
    if random_state is None or isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state

    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # patch_ratio = rng.normal(patch_ratio, 0.05)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]])

    from scipy.stats import truncnorm

    # Random perspective and affine perturbations
    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
        h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[rng.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([rng.uniform(-t_min[0], t_max[0],1), rng.uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all angles are valid except angle=0
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[rng.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]

    # Rescale to actual size
    shape = np.array([height, width])[::-1]  # different convention [y, x]
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    # Use OpenCV's getPerspectiveTransform like in the original
    homography = cv2.getPerspectiveTransform(np.float32(pts1+shift), np.float32(pts2+shift))
    
    # Test the homography with sample points
    test_points = np.array([[width/2, height/2], [0, 0], [width, height], [0, height], [width, 0]])
    homogeneous_pts = np.hstack([test_points, np.ones((len(test_points), 1))])
    warped_pts = homogeneous_pts @ homography.T
    denominators = warped_pts[:, 2]

    # Ensure denominators aren't close to zero
    if np.any(np.abs(denominators) < 1e-6):
        # Try again with less extreme transformation
        print("Homography estimation failed, retrying...")
        return sample_homography(height, width,
                              perspective_amplitude_x=perspective_amplitude_x*0.8,
                              perspective_amplitude_y=perspective_amplitude_y*0.8,
                              random_state=random_state)
    return homography

def plot_matches_with_accuracy(image0, image1, kpts0, kpts1, mkpts0_warped, accuracy_mask, radius=2, color=(255, 0, 0), 
                              correct_color=(0, 255, 0), incorrect_color=(255, 0, 0), layout='lr'):
    """
    Plot matches with different colors based on accuracy.
    
    Args:
        image0, image1: Input images
        kpts0, kpts1: Keypoints for each image
        mkpts0_warped: Warped keypoints from image0 to image1 (optional)
        accuracy_mask: Boolean mask indicating which matches are correct (within threshold)
        radius: Radius for keypoint markers
        color: Color for keypoint markers
        correct_color: Color for correct matches (BGR format)
        incorrect_color: Color for incorrect matches (BGR format)
        layout: 'lr' for left-right, 'ud' for up-down layout
    """
    image0 = image0.cpu().detach().numpy() if isinstance(image0, torch.Tensor) else image0
    image1 = image1.cpu().detach().numpy() if isinstance(image1, torch.Tensor) else image1
    kpts0 = kpts0.cpu().detach().numpy() if isinstance(kpts0, torch.Tensor) else kpts0
    kpts1 = kpts1.cpu().detach().numpy() if isinstance(kpts1, torch.Tensor) else kpts1
    accuracy_mask = accuracy_mask.cpu().detach().numpy() if isinstance(accuracy_mask, torch.Tensor) else accuracy_mask

    out0 = plot_keypoints(image0, kpts0, radius, color)
    out1 = plot_keypoints(image1, kpts1, radius, color)
    
    # Add warped keypoints to image1 if available
    if mkpts0_warped is not None:
        mkpts0_warped = mkpts0_warped.cpu().detach().numpy() if isinstance(mkpts0_warped, torch.Tensor) else mkpts0_warped
        # Filter out NaN values
        valid_warped = ~np.isnan(mkpts0_warped).any(axis=1)
        if valid_warped.any():
            valid_warped_pts = mkpts0_warped[valid_warped]
            out1 = plot_keypoints(out1, valid_warped_pts, radius, (0, 0, 255))  # Blue for warped points

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = out0
        out[:H1, W0:, :] = out1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = out0
        out[H0:, :W1, :] = out1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0 = np.round(kpts0).astype(int)
    kpts1 = np.round(kpts1).astype(int)

    for i, (kpt0, kpt1) in enumerate(zip(kpts0, kpts1)):
        (y0, x0), (y1, x1) = kpt0, kpt1
        
        # Choose color based on accuracy
        line_color = correct_color if accuracy_mask[i] else incorrect_color

        if layout == "lr":
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=line_color, thickness=1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=line_color, thickness=1, lineType=cv2.LINE_AA)
    
    # Draw lines from kpt1 to warped points on the right side if available
    if mkpts0_warped is not None:
        # Convert to numpy and handle NaN values properly
        mkpts0_warped = mkpts0_warped.cpu().detach().numpy() if isinstance(mkpts0_warped, torch.Tensor) else mkpts0_warped
        
        for i, (kpt1, warped_pt) in enumerate(zip(kpts1, mkpts0_warped)):
            (y1, x1) = kpt1
            (yw, xw) = warped_pt
            
            # Skip if warped point is NaN (couldn't be warped)
            if np.isnan(yw) or np.isnan(xw):
                continue
            
            # Convert to integers for OpenCV
            x1, y1 = int(x1), int(y1)
            xw, yw = int(xw), int(yw)
            
            if layout == "lr":
                # Draw line from matched keypoint to warped keypoint on right side
                cv2.line(out, (x1 + W0, y1), (xw + W0, yw), color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            elif layout == "ud":
                # Draw line from matched keypoint to warped keypoint on bottom side
                cv2.line(out, (x1, y1 + H0), (xw, yw + H0), color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    return out