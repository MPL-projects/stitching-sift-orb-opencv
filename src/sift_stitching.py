import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from utile import *

MODE = SIFT #0

# ==============================================
# Détection et correspondance des points clés avec SIFT
# ==============================================

def detect_sift(img):
    """
    Détecte les points clés et descripteurs avec SIFT.

    :param img: Image en niveaux de gris.
    :return: Liste des points clés et des descripteurs.
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


# ==============================================
# Calcul de la matrice d'homographie avec RANSAC
# ==============================================

def compute_homography(pairs):
    """
    Calcule la matrice d'homographie à partir des paires de points.

    :param pairs: Liste des points correspondants.
    :return: Matrice d'homographie normalisée.
    """
    A = []
    for x1, y1, x2, y2 in pairs:
        A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]

def random_point(matches, k=4):
    """
    Sélectionne aléatoirement k points parmi les correspondances.

    :param matches: Liste des correspondances.
    :param k: Nombre de points à sélectionner.
    :return: Tableau des points sélectionnés.
    """
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)


def get_error(points, H):
    """
    Calcule l'erreur de transformation pour chaque point.

    :param points: Correspondances de points.
    :param H: Matrice d'homographie.
    :return: Erreur pour chaque point.
    """

    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # Normalisation
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2 # Calcul de l'erreur

    return errors

def ransac(matches, threshold, iters):
    """
    Applique l'algorithme RANSAC pour estimer l'homographie.

    :param matches: Correspondances de points.
    :param threshold: Seuil d'erreur pour définir un inlier.
    :param iters: Nombre d'itérations.
    :return: Liste des inliers et matrice d'homographie optimale.
    """

    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = compute_homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H


# ==============================================
# Main SIFT Stitching 
# ==============================================

def sift_stitching(path1, path2, plot = False):
    left_gray, left_origin, left_rgb = read_image(path1)
    right_gray, right_origin, right_rgb = read_image(path2)

    kp_left, des_left = detect_sift(left_gray)
    kp_right, des_right = detect_sift(right_gray)

    matches = match_features(kp_left, des_left, kp_right, des_right, mode = MODE)

    H = compute_homography(matches)
    inliers, H = ransac(matches, 0.5, 2000)

    print("Left Image Shape:", left_rgb.shape)
    print("Right Image Shape:", right_rgb.shape)

    stitched_img = warp_and_stitch(left_rgb, right_rgb, H)

    if plot :
        plot_all(left_rgb, right_rgb, inliers, stitched_img)



    print("stitched_img Image Shape:", stitched_img.shape)


    return stitched_img



if __name__ == "__main__":
    # sift_stitching('../images/1.jpg','../images/2.jpg', plot = True)
    sift_stitching("stitched_temp_1.jpg","../images/salon_tele/3.jpg", plot = True)
    # sift_stitching("../images/pant2/a1.jpg","../images/pant2/a2.jpg", plot = True)
    # stitching('../images/1.jpg','../images/2.jpg', type = MODE)
