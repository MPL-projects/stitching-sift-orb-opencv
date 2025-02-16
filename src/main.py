import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importer les fonctions de stitching
from sift_stitching import sift_stitching
from orb_stitching import orb_stitching

# Définition des modes
SIFT = 0
ORB = 1
OPENCV = 2

def stitch_opencv(image1, image2):
    """
    Utilise OpenCV Stitcher pour fusionner deux images automatiquement.

    :param image1: Chemin de la première image.
    :param image2: Chemin de la deuxième image.
    :return: Image stitchée ou None si l'assemblage échoue.
    """
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    images = [img1, img2]
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, stitched_img = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        print("Erreur : Échec du stitching avec OpenCV Stitcher.")
        return None

    return stitched_img

def stitch_images(image1, image2, mode=SIFT):
    """
    Assemble deux images avec la méthode spécifiée.

    :param image1: Chemin de la première image.
    :param image2: Chemin de la deuxième image.
    :param mode: Mode de stitching (SIFT=0, ORB=1, OPENCV=2).
    :return: Image stitchée.
    """
    if mode == SIFT:
        return sift_stitching(image1, image2, plot=False)
    elif mode == ORB:
        return orb_stitching(image1, image2)
    elif mode == OPENCV:
        return stitch_opencv(image1, image2)
    else:
        print("Erreur: Mode invalide. Utilisez SIFT=0, ORB=1 ou OPENCV=2.")
        return None

# --- Affichage des résultats avec les 3 méthodes ---
if __name__ == "__main__":
    image1 = "../images/salon_tele/1.jpg"  # Remplace par ton image
    image2 = "../images/salon_tele/2.jpg"  # Remplace par ton image

    # Exécuter le stitching pour SIFT, ORB et OpenCV Stitcher
    stitched_sift = stitch_images(image1, image2, mode=SIFT)
    stitched_orb = stitch_images(image1, image2, mode=ORB)
    stitched_opencv = stitch_images(image1, image2, mode=OPENCV)

    # Création du subplot
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Affichage SIFT
    if stitched_sift is not None:
        axes[0].imshow(stitched_sift)
        axes[0].set_title("Stitching avec SIFT")
        axes[0].axis("off")

    # Affichage ORB
    if stitched_orb is not None:
        axes[1].imshow(stitched_orb)
        axes[1].set_title("Stitching avec ORB")
        axes[1].axis("off")

    # Affichage OpenCV Stitcher
    if stitched_opencv is not None:
        axes[2].imshow(cv2.cvtColor(stitched_opencv, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Stitching avec OpenCV Stitcher")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()
