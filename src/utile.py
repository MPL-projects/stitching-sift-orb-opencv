import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==============================================
# MODE
# ==============================================

SIFT = 0
ORB = 1

# ==============================================
# Chargement et traitement des images
# ==============================================

def read_image(path):
    """
    Charge une image en couleur et en niveaux de gris.
    
    :param path: Chemin de l'image.
    :return: Tuple (image en gris, image originale, image en RGB).
    """
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

# ==============================================
# Correspondance des points clés
# ==============================================

def match_features(kp1, des1, kp2, des2, mode = 0,  threshold=0.5,):
    """
    Trouve les correspondances entre deux ensembles de points clés avec ORB.

    :param kp1: Points clés de l'image 1.
    :param des1: Descripteurs de l'image 1.
    :param kp2: Points clés de l'image 2.
    :param des2: Descripteurs de l'image 2.
    :return: Liste des points correspondants.
    """
    matches_array = np.array([])

    match mode:
        case 0:
            
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < threshold * n.distance]
            matches_array = np.array([list(kp1[m.queryIdx].pt) + list(kp2[m.trainIdx].pt) for m in good])
            
        case 1:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Utilisation de NORM_HAMMING pour ORB
            matches = bf.match(des1, des2)  # Trouver les correspondances
            matches = sorted(matches, key=lambda x: x.distance)  # Trier par distance
            matches_array = np.array([list(kp1[m.queryIdx].pt) + list(kp2[m.trainIdx].pt) for m in matches])

        case _:
            print("erreur")

    return matches_array

# ==============================================
# Affichage des correspondances
# ==============================================

def plot_matches(matches, total_img, ax):
    """
    Affiche les correspondances entre deux images dans un subplot.

    :param matches: Liste des points correspondants.
    :param total_img: Image combinée des deux images à comparer.
    :param ax: Subplot dans lequel afficher les correspondances.
    """
    offset = total_img.shape[1] // 2

    ax.set_aspect('equal')
    ax.imshow(np.array(total_img).astype('uint8'))
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr', label="Points de gauche")
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr', label="Points de droite")
    ax.plot([matches[:, 0], matches[:, 2] + offset], 
            [matches[:, 1], matches[:, 3]], 'r', linewidth=0.5)

    ax.set_title("Match Features")


# ==============================================
# Image Warping and Stitching
# ==============================================

def warp_and_stitch(left, right, H):
    """
    Applique l'homographie et fusionne les images en maximisant la surface stitchée.

    :param left: Image de gauche.
    :param right: Image de droite.
    :param H: Matrice d'homographie.
    :return: Image stitchée finale.
    """
    print("Stitching images ...")

    # Dimensions des images
    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]

    # Définition des coins des images
    corners_left = np.array([[0, 0], [0, h_left], [w_left, h_left], [w_left, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners_right = np.array([[0, 0], [0, h_right], [w_right, h_right], [w_right, 0]], dtype=np.float32).reshape(-1, 1, 2)

    # Appliquer l'homographie aux coins de l'image de droite
    warped_corners_right = cv2.perspectiveTransform(corners_right, H)

    # Fusionner les coins pour trouver la bounding box
    all_corners = np.vstack((corners_left, warped_corners_right))

    # Déterminer les min et max de x et y
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Ajustement pour se rapprocher de la taille maximale voulue
    expected_xmax = w_left + w_right
    expected_ymax = h_left + h_right

    xmax = max(xmax, expected_xmax)
    ymax = max(ymax, expected_ymax)

    # Matrice de translation pour éviter les valeurs négatives
    translation_matrix = np.array([[1, 0, -xmin],
                                   [0, 1, -ymin],
                                   [0, 0, 1]], dtype=np.float32)

    # Appliquer les transformations
    output_size = (xmax - xmin, ymax - ymin)
    left_warped = cv2.warpPerspective(left, translation_matrix @ H, output_size)
    right_warped = cv2.warpPerspective(right, translation_matrix, output_size)

    # Fusionner les images
    mask_left = (left_warped > 0).astype(np.uint8)
    mask_right = (right_warped > 0).astype(np.uint8)

    # Fusion intelligente des images pour éviter les trous noirs
    stitched_img = np.where(mask_left, left_warped, right_warped)


    # Recadrage de l'image finale pour supprimer les zones noires 
    gray_stitched = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
    non_black_pixels = np.where(gray_stitched > 0)  # Coordonnées des pixels non noirs

    # Calculer les nouvelles limites
    if non_black_pixels[0].size > 0 and non_black_pixels[1].size > 0:
        ymin_crop = np.min(non_black_pixels[0])  # Premier y non noir
        ymax_crop = np.max(non_black_pixels[0])  # Dernier y non noir
        xmin_crop = np.min(non_black_pixels[1])  # Premier x non noir
        xmax_crop = np.max(non_black_pixels[1])  # Dernier x non noir

        # Recadrer l'image
        stitched_img = stitched_img[ymin_crop:ymax_crop+1, xmin_crop:xmax_crop+1]

    return stitched_img

# ==============================================
# Affiche le subplot
# ==============================================

def resize_to_match_height(img, target_height):
    """
    Redimensionne une image pour qu'elle ait la même hauteur qu'une autre.
    """
    h, w, _ = img.shape
    scale = target_height / h
    new_w = int(w * scale)
    resized_img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    return resized_img

def plot_all(left_rgb, right_rgb, inliers, stitched_img):
    # Ajuste la hauteur des images
    target_height = min(left_rgb.shape[0], right_rgb.shape[0])  # Prend la plus petite hauteur
    left_rgb = resize_to_match_height(left_rgb, target_height)
    right_rgb = resize_to_match_height(right_rgb, target_height)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].imshow(left_rgb)
    axes[0, 0].set_title("Image de gauche")
    
    axes[0, 1].imshow(right_rgb)
    axes[0, 1].set_title("Image de droite")
    
    total_img = np.concatenate((left_rgb, right_rgb), axis=1)  # Maintenant, elles ont la même hauteur
    plot_matches(inliers, total_img, axes[1, 0])

    axes[1, 1].imshow(stitched_img)
    axes[1, 1].set_title("Image assemblée")

    plt.tight_layout()
    plt.show()

