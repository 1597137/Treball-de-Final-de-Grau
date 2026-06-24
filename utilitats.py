
"""
Utilitats per a la gestió de les imatges i els resultats dels experiments.

Aquest fitxer conté:
    - Funcions per afegir soroll gaussià a un sinograma.
    - Funcions per calcular mètriques de qualitat entre imatges.
    - Funcions per visualitzar les imatges i els resultats.
    - Funcions per reescalar les imatges per a la comparació de mètriques.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim


# .......
# SOROLL
# .......

def add_gaussian_noise(sino, sigma_rel, rng=None):
    """
    Afegeix soroll gaussià a un sinograma.
    La desviació típica del soroll es calcula com sigma_rel * max(abs(sino)).

    Parameters
    ----------
    sino : np.ndarray
        Sinograma al qual s'afegirà el soroll.
    
    sigma_rel : float
        Factor relatiu per calcular la desviació típica del soroll.
    
    rng : np.random.Generator, optional
        Generador aleatori per a la reproducció dels resultats. Si no es proporciona,
        es crea un nou generador aleatori.

    Returns
    -------
    sino_noisy : np.ndarray
        Sinograma amb soroll afegit.
    """

    # Si no passem un generador aleatori se'n crea un
    rng = np.random.default_rng() if rng is None else rng   

    # Intensitat absoluta del soroll (sigma) basada en la intensitat màxima del sinograma i el paràmetre sigma_rel
    # Afegim un petit valor (1e-12) per evitar problemes de divisió per zero en cas que el sinograma sigui tot zeros.
    sigma = sigma_rel * (np.max(np.abs(sino)) + 1e-12) 
    
    # Generem soroll gaussià amb mitjana 0 i desviació típica sigma
    noise = rng.normal(loc=0.0, scale=sigma, size=sino.shape)  
    
    sino_noisy = sino + noise

    return sino_noisy


# ..........
# MÈTRIQUES 
# ..........

def metrics(rec, gt):
    """
    Calcula les mètriques de qualitat entre la imatge reconstruïda i la ground truth.
    
    Parameters
    ---------- 
    rec : np.ndarray
        Imatge reconstruïda.
    
    gt : np.ndarray
        Imatge original o ground truth.
    
    Returns
    -------
    dict
        Diccionari amb les mètriques calculades:
        - "RelL2": Error relatiu en norma L^2.
        - "RelL2_percent": Error relatiu en norma L^2 expressat en percentatge.
        - "SSIM": Índex de semblança estructural.
    """
    
    # Convertim les imatges a tipus float64 per assegurar la precisió en els càlculs
    rec = np.asarray(rec, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    # Error relatiu en norma L^2
    rel_l2 = np.linalg.norm(rec - gt)/(np.linalg.norm(gt) + 1e-12)

    # Rang de valors de la ground truth per al càlcul de SSIM
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        data_range = 1.0

    ssim_value = ssim(gt, rec, data_range=data_range)

    return {
        "RelL2": rel_l2,
        "RelL2_percent": 100 * rel_l2,
        "SSIM": ssim_value
    }


# ..............
# VISUALITZACIÓ
# ..............

def show_triplet(title, gt, rec, sino=None):
    """
    Mostra la ground truth, la reconstrucció i opcionalment el sinograma en una figura.

    Parameters
    ----------
    title : str
        Títol per a la imatge reconstruïda.
    
    gt : np.ndarray
        Imatge original o ground truth.

    rec : np.ndarray
        Imatge reconstruïda.

    sino : np.ndarray, optional
        Sinograma corresponent a la reconstrucció. Si no es proporciona, només es mostraran
        la ground truth i la reconstrucció.
    """

    def normalize_for_display(img):
        """
        Normalitza una imatge per a la visualització, escalant els valors al rang [0, 1].
        """
        img = np.asarray(img,dtype=np.float64)

        img_min = img.min()
        img_max = img.max()

        if img_max - img_min < 1e-12:
            return np.zeros_like(img)
        
        return (img - img_min) / (img_max - img_min)

    gt_disp = normalize_for_display(gt)
    rec_disp = normalize_for_display(rec)

    if sino is None:
        fig, axes = plt.subplots(1,2,figsize=(9,4))

        axes[0].imshow(gt_disp, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Ground truth")
        axes[0].axis("off")

        axes[1].imshow(rec_disp, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(title)
        axes[1].axis("off")

    else:
        fig, axes = plt.subplots(1,3,figsize=(13,4))

        axes[0].imshow(gt_disp, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Ground truth")
        axes[0].axis("off")

        axes[1].imshow(rec_disp, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(title)
        axes[1].axis("off")

        axes[2].imshow(sino, cmap="gray", aspect="auto")
        axes[2].set_title("Sinograma")
        axes[2].set_xlabel("Angle")
        axes[2].set_ylabel("Detector")

    plt.tight_layout()
    plt.show()
