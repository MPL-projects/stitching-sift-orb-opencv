import cv2
import matplotlib.pyplot as plt
import numpy as np


from utile import *

MODE = ORB #1

# ==============================================
# Détection et correspondance des points clés avec ORB
# ==============================================

def detect_orb(img):
    """
    Détecte les points clés et descripteurs avec ORB.

    :param img: Image en niveaux de gris.
    :return: Liste des points clés et descripteurs.
    """
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


# ==============================================
# Calcul de la matrice d'homographie avec RANSAC
# ==============================================

def compute_homography_ransac(matches_array, threshold=5.0, max_iters=2000):
    """
    Computes homography matrix using RANSAC for robust feature matching.

    :param matches_array: Array of matched keypoint coordinates.
    :param threshold: RANSAC inlier threshold.
    :param max_iters: Maximum number of iterations for RANSAC.
    :return: Homography matrix and inliers.
    """
    src_pts = matches_array[:, :2]
    dst_pts = matches_array[:, 2:]

    # Compute homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold, maxIters=max_iters)
    inliers = matches_array[mask.ravel() == 1]
    print(f"Inliers: {len(inliers)}/{len(matches_array)}")
    return H, inliers


# ==============================================
# Main ORB Stitching 
# ==============================================

def orb_stitching(path1, path2, plot = False):
    """
    Extends the ORB-based stitching workflow to include RANSAC, homography, and image stitching.

    :param path1: Path to the first image.
    :param path2: Path to the second image.
    """
    # Load images
    left_gray, _, left_rgb = read_image(path1)
    right_gray, _, right_rgb = read_image(path2)

    # Detect ORB features
    kp_left, des_left = detect_orb(left_gray)
    kp_right, des_right = detect_orb(right_gray)

    # Match features
    matches_array = match_features(kp_left, des_left, kp_right, des_right, mode = MODE)
    # debug_matches(left_rgb, kp_left, right_rgb, kp_right, matches_raw, matches_array)

    # Compute homography with RANSAC
    H, inliers = compute_homography_ransac(matches_array)

    # Warp and stitch the images
    stitched_img = warp_and_stitch(left_rgb, right_rgb, H)

    if plot :
        plot_all(left_rgb, right_rgb, inliers, stitched_img)

    return stitched_img


# Run the extended stitching workflow
if __name__ == "__main__":
    orb_stitching("../images/bastille/Capture d'écran 2025-02-07 211913.png","../images/bastille/Capture d'écran 2025-02-07 211920.png", plot = True)

