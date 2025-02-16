import cv2
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt

# Import feature matching and stitching functions
from sift_stitching import sift_stitching
from orb_stitching import orb_stitching

# Define constants
SIFT = 0
ORB = 1
OPENCV = 2 


def extract_number(filename):
    """ Extrait le premier nombre trouvé dans un nom de fichier. """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def get_image_files(folder, extensions=[".jpg", ".png", ".jpeg"]):
    """ Récupère une liste de fichiers image triés numériquement depuis un dossier. """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))

    files.sort(key=lambda x: extract_number(os.path.basename(x)))

    return files


def stitch_images_from_folder(folder, mode=SIFT):
    """
    Stitch ensemble toutes les images d'un dossier en suivant l'ordre numérique.
    Retourne l'image stitchée finale.

    :param folder: Chemin du dossier contenant les images.
    :param mode: Mode de stitching (SIFT=0, ORB=1, OPENCV=2).
    :return: Image stitchée finale ou None en cas d'échec.
    """
    image_files = get_image_files(folder)
    if len(image_files) < 2:
        print("Erreur: Il faut au moins deux images pour le stitching.")
        return None

    # Stitching avec OpenCV
    if mode == OPENCV:
        return stitch_opencv(image_files)

    # Charger la première image comme point de départ
    base_image = image_files[0]

    for i in range(1, len(image_files)):
        next_image = image_files[i]
        print(f"Stitching: {base_image} + {next_image}")

        if mode == SIFT:
            stitched_img = sift_stitching(base_image, next_image, plot=False)
        elif mode == ORB:
            stitched_img = orb_stitching(base_image, next_image)
        else:
            print("Erreur: Mode invalide. Utilisez SIFT=0, ORB=1 ou OPENCV=2.")
            return None

        # Sauvegarde temporaire et mise à jour
        temp_output = f"stitched_temp_{mode}_{i}.jpg"
        stitched_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(temp_output, stitched_img)
        base_image = temp_output

    return stitched_img


def stitch_opencv(image_files):
    """
    Utilise OpenCV Stitcher pour fusionner toutes les images automatiquement.

    :param image_files: Liste des chemins des images à assembler.
    :return: Image stitchée ou None si l'assemblage échoue.
    """
    images = [cv2.imread(img) for img in image_files]

    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, stitched_img = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        print("Erreur : Échec du stitching avec OpenCV Stitcher.")
        return None

    return stitched_img


# --- Affichage des résultats avec les 3 méthodes ---
if __name__ == "__main__":
    folder_path = "../images/salon_tele"

    # Exécuter le stitching pour SIFT, ORB et OpenCV Stitcher
    stitched_sift = stitch_images_from_folder(folder_path, mode=SIFT)
    stitched_orb = stitch_images_from_folder(folder_path, mode=ORB)
    stitched_opencv = stitch_images_from_folder(folder_path, mode=OPENCV)

    # Création du subplot
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Affichage SIFT
    if stitched_sift is not None:
        axes[0].imshow(cv2.cvtColor(stitched_sift, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Stitching avec SIFT")
        axes[0].axis("off")

    # Affichage ORB
    if stitched_orb is not None:
        axes[1].imshow(cv2.cvtColor(stitched_orb, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Stitching avec ORB")
        axes[1].axis("off")

    # Affichage OpenCV Stitcher
    if stitched_opencv is not None:
        axes[2].imshow(cv2.cvtColor(stitched_opencv, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Stitching avec OpenCV Stitcher")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()
