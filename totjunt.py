
"""
Executa experiments de reconstrucció tomogràfica amb diferents phantoms i casos experimentals.
Permet estudiar l'efecte del soroll, la quantitat d'angles i l'interval angular en les reconstruccions amb FBP, Tikhonov i TV.
També permet estudiar l'efecte del paràmetre de regularització en Tikhonov i TV.

Permet executar un experiment individual, tots els experiments o un estudi de lambda per a un phantom i cas concret.

Els resultats es guarden en un CSV i les figures de reconstrucció es guarden com a figures PNG.
"""

import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt

from phantoms import make_all_phantoms
from experiments import EXPERIMENT_CASES
import reconstruccio as rec
import utilitats as ut


# .....................
# EXPERIMENT PRINCIPAL
# .....................

def run_one_case(gt, n=256, n_angles=180, theta_min=0, theta_max=180, sigma_rel=0.0, rng_seed=0, 
                 lam_tikhonov=1e-2, lam_tv=0.02, tau_tv=0.01, n_iter_tv=50):
    """
    Executa un experiment de reconstrucció per a un phantom i un cas experimental concret.
    Retorna la imatge original, el sinograma, els angles i un diccionari amb els resultats de cada mètode de reconstrucció (FBP, Tikhonov, TV).
    Cada entrada del diccionari de resultats conté:
    - "image": la imatge reconstruïda
    - "metrics": un diccionari amb les mètriques de qualitat (RelL2, RelL2_percent, SSIM)
    - "time": el temps d'execució de la reconstrucció en segons

    Parametres
    ----------
    gt : np.ndarray
        Imatge original (ground truth) del phantom.
    n : int
        Mida de la imatge de reconstrucció (n x n).
    n_angles : int
        Nombre d'angles de projecció.
    theta_min : float
        Angle mínim de projecció (en graus).
    theta_max : float
        Angle màxim de projecció (en graus).
    sigma_rel : float
        Desviació estàndard relativa del soroll gaussià a afegir al sinograma. Si és 0, no s'afegeix soroll.
    rng_seed : int
        Semilla per al generador de nombres aleatoris per a la reproducció dels resultats.
    lam_tikhonov : float
        Paràmetre de regularització per a la reconstrucció Tikhonov.
    lam_tv : float
        Paràmetre de regularització per a la reconstrucció TV.
    tau_tv : float
        Paràmetre de pas per a la reconstrucció TV.
    n_iter_tv : int
        Nombre d'iteracions per a la reconstrucció TV.

    Returns
    -------
    gt : np.ndarray
        Imatge original (ground truth) del phantom.
    sino : np.ndarray
        Sinograma amb soroll (si s'ha afegit).
    theta : np.ndarray
        Angles de projecció.
    results : dict
        Diccionari amb els resultats de cada mètode de reconstrucció.
    """

    rng = np.random.default_rng(rng_seed)

    # 1. Copiem la imatge original per evitar modificar-la
    gt = gt.copy()

    # 2. Generem els angles de projecció
    theta = np.linspace(theta_min, theta_max, n_angles, endpoint=False)

    # 3. Generem el sinograma sense soroll
    sino_clean = rec.forward_project_manual(
        gt,
        theta,
        n_detectors=n
    )

    # 4. Afegim soroll gaussià al sinograma si sigma_rel > 0
    if sigma_rel > 0:
        sino = ut.add_gaussian_noise(
            sino_clean,
            sigma_rel=sigma_rel,
            rng=rng
        )
    else:
        sino = sino_clean

    results = {}

    # .....
    # FBP
    # .....

    t0 = time.perf_counter()

    rec_fbp = rec.recon_fbp_manual(
        sino,
        theta,
        out_size=n,
        filter_name="ramp"
    )

    time_fbp = time.perf_counter() - t0

    results["FBP"] = {
        "image": rec_fbp,
        "metrics": ut.metrics(rec_fbp, gt),
        "time": time_fbp
    }

    # ................
    # TIKHONOV AMB CG
    # ................

    t0 = time.perf_counter()

    rec_tikh = rec.recon_tikhonov_cg_manual(
        sino,
        theta,
        out_size=n,
        lam=lam_tikhonov,
        cg_tol=1e-5,
        cg_maxiter=100
    )

    time_tikh = time.perf_counter() - t0

    results["Tikhonov"] = {
        "image": rec_tikh,
        "metrics": ut.metrics(rec_tikh, gt),
        "time": time_tikh
    }

    # ..................
    # REGULARITZACIÓ TV
    # ..................

    t0 = time.perf_counter()

    rec_tv = rec.recon_tv_manual(
        sino,
        theta,
        out_size=n,
        lam=lam_tv,
        tau=tau_tv,
        n_iter=n_iter_tv
    )

    time_tv = time.perf_counter() - t0

    results["TV"] = {
        "image": rec_tv,
        "metrics": ut.metrics(rec_tv, gt),
        "time": time_tv
    }

    return gt, sino, theta, results


# ................
# GUARDAR FIGURES
# ................

def save_reconstruction_figure(gt, rec_img, sino, method_name, phantom_name, case_name, output_dir):
    """
    Guarda una figura amb ground truth, reconstrucció i sinograma.
    La normalització que es fa només és per visualitzar les imatges.
    """

    def normalize_for_display(img):
        """
        Normalitza una imatge a l'interval [0, 1] per a visualització.
        """
        img = np.asarray(img, dtype=np.float64)
        img_min = img.min()
        img_max = img.max()

        if img_max - img_min < 1e-12:
            return np.zeros_like(img)
        
        return (img - img_min) / (img_max - img_min)
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    gt_disp = normalize_for_display(gt)
    rec_disp = normalize_for_display(rec_img)

    axes[0].imshow(gt_disp, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground truth")
    axes[0].axis("off")

    axes[1].imshow(rec_disp, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(method_name)
    axes[1].axis("off")

    axes[2].imshow(sino, cmap="gray", aspect="auto")
    axes[2].set_title("Sinograma")
    axes[2].set_xlabel("Angle")
    axes[2].set_ylabel("Detector")

    plt.tight_layout()

    filename = f"{phantom_name}_{case_name}_{method_name}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# .................................
# EXECUCIÓ DE TOTS ELS EXPERIMENTS
# .................................

def run_all_experiments(
    phantoms_to_run,
    cases_to_run,
    n=256,
    rng_seed=0,
    lam_tikhonov=1e-3,
    lam_tv=0.001,
    tau_tv=0.01,
    n_iter_tv=50,
    output_root="resultats"
):
    """
    Executa tots els experiments per a una llista de phantoms i casos experimentals.
    
    Parameters
    ----------
    phantoms_to_run : list of str
        Llista de noms de phantoms a executar.
    cases_to_run : list of str
        Llista de noms de casos experimentals a executar.
    n : int
        Mida de la imatge de reconstrucció (n x n).
    rng_seed : int
        Semilla per al generador de nombres aleatoris per a la reproducció dels resultats.
    lam_tikhonov : float
        Paràmetre de regularització per a la reconstrucció Tikhonov.
    lam_tv : float
        Paràmetre de regularització per a la reconstrucció TV.
    tau_tv : float
        Paràmetre de pas de temps per a la reconstrucció TV.
    n_iter_tv : int
        Nombre d'iteracions per a la reconstrucció TV.
    output_root : str
        Ruta arrel per a la sortida dels resultats.
    """

    os.makedirs(output_root, exist_ok=True)

    csv_path = os.path.join(output_root, "metrics_results.csv")

    rows = []

    phantoms = make_all_phantoms(n)

    # Mostrem informació dels phantoms
    # És útil per verificar les intensitats i la norma L2 abans de començar els experiments
    for name, img in phantoms.items():
        print(name)
        print("  min =", img.min())
        print("  max =", img.max())
        print("  norma L2 =", np.linalg.norm(img))
        print()

    for phantom_name in phantoms_to_run:

        gt_phantom = phantoms[phantom_name]

        phantom_dir = os.path.join(output_root, phantom_name)
        os.makedirs(phantom_dir, exist_ok=True)

        for case_name in cases_to_run:

            case = EXPERIMENT_CASES[case_name]

            print(f"\nExecutant: phantom={phantom_name}, cas={case_name}")

            gt, sino, theta, results = run_one_case(
                gt=gt_phantom,
                n=n,
                n_angles=case["n_angles"],
                theta_min=case["theta_min"],
                theta_max=case["theta_max"],
                sigma_rel=case["sigma_rel"],
                rng_seed=rng_seed,
                lam_tikhonov=lam_tikhonov,
                lam_tv=lam_tv,
                tau_tv=tau_tv,
                n_iter_tv=n_iter_tv
            )

            case_dir = os.path.join(phantom_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)

            for method_name, result in results.items():

                rec_img = result["image"]
                met = result["metrics"]
                elapsed = result["time"]

                # Guardem les mètriques en una fila del CSV
                rows.append({
                    "phantom": phantom_name,
                    "case": case_name,
                    "case_description": case["description"],
                    "method": method_name,
                    "n": n,
                    "n_angles": case["n_angles"],
                    "theta_min": case["theta_min"],
                    "theta_max": case["theta_max"],
                    "sigma_rel": case["sigma_rel"],
                    "RelL2": met["RelL2"],
                    "RelL2_percent": met["RelL2_percent"],
                    "SSIM": met["SSIM"],
                    "time_seconds": elapsed
                })

                # Guardem la figura amb ground truth, reconstrucció i sinograma
                save_reconstruction_figure(
                    gt=gt,
                    rec_img=rec_img,
                    sino=sino,
                    method_name=method_name,
                    phantom_name=phantom_name,
                    case_name=case_name,
                    output_dir=case_dir
                )

    # Guardem totes les mètriques en un CSV
    fieldnames = [
        "phantom",
        "case",
        "case_description",
        "method",
        "n",
        "n_angles",
        "theta_min",
        "theta_max",
        "sigma_rel",
        "RelL2",
        "RelL2_percent",
        "SSIM",
        "time_seconds"
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMètriques guardades a: {csv_path}")
    print(f"Figures guardades a la carpeta: {output_root}")


# .......................................
# ESTUDI DEL PARÀMETRE DE REGULARITZACIÓ
# .......................................

def run_lambda_study(
    phantom_name,
    case_name,
    n=512,
    rng_seed=0,
    lam_tikhonov_values=None,
    lam_tv_values=None,
    tau_tv=0.01,
    n_iter_tv=50,
    output_root="resultats_lambda"
):
    """
    Estudia l'efecte del paràmetre de regularització en Tikhonov i TV
    per a un phantom i un cas experimental concrets.

    Guarda les mètriques en un CSV i les figures en carpetes.
    
    Parameters
    ----------
    phantom_name : str
        Nom del phantom a utilitzar.
    case_name : str
        Nom del cas experimental a utilitzar.
    n : int
        Mida de la imatge de reconstrucció (n x n).
    rng_seed : int
        Semilla per al generador de nombres aleatoris per a la reproducció dels resultats.
    lam_tikhonov_values : list of float, optional
        Llista de valors de lambda per a Tikhonov a estudiar. Si és None, s'utilitzen els valors per defecte [1e-4, 1e-3, 1e-2, 1e-1].
    lam_tv_values : list of float, optional
        Llista de valors de lambda per a TV a estudiar. Si és None, s'utilitzen els valors per defecte [1e-4, 5e-4, 1e-3, 5e-3, 1e-2].
    tau_tv : float
        Paràmetre de pas de temps per a la reconstrucció TV.
    n_iter_tv : int
        Nombre d'iteracions per a la reconstrucció TV.
    output_root : str
        Ruta arrel per a la sortida dels resultats.    
    """

    if lam_tikhonov_values is None:
        lam_tikhonov_values = [1e-4, 1e-3, 1e-2, 1e-1]

    if lam_tv_values is None:
        lam_tv_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    os.makedirs(output_root, exist_ok=True)

    phantoms = make_all_phantoms(n)
    gt = phantoms[phantom_name]
    case = EXPERIMENT_CASES[case_name]

    rng = np.random.default_rng(rng_seed)

    # Angles de projecció
    theta = np.linspace(
        case["theta_min"],
        case["theta_max"],
        case["n_angles"],
        endpoint=False
    )

    # Sinograma sense soroll
    sino_clean = rec.forward_project_manual(
        gt,
        theta,
        n_detectors=n
    )

    # Afegim soroll gaussià al sinograma si sigma_rel > 0
    if case["sigma_rel"] > 0:
        sino = ut.add_gaussian_noise(
            sino_clean,
            sigma_rel=case["sigma_rel"],
            rng=rng
        )
    else:
        sino = sino_clean

    rows = []

    # Carpeta especifica per al phantom i cas
    case_dir = os.path.join(output_root, phantom_name, case_name)
    os.makedirs(case_dir, exist_ok=True)


    # ESTUDI DE LAMBDA: TIKHONOV
 
    for lam in lam_tikhonov_values:

        print(f"\nTikhonov: lambda = {lam}")

        t0 = time.perf_counter()

        rec_img = rec.recon_tikhonov_cg_manual(
            sino,
            theta,
            out_size=n,
            lam=lam,
            cg_tol=1e-5,
            cg_maxiter=100
        )

        elapsed = time.perf_counter() - t0
        met = ut.metrics(rec_img, gt)

        rows.append({
            "phantom": phantom_name,
            "case": case_name,
            "method": "Tikhonov",
            "lambda": lam,
            "tau": "",
            "n_iter": "",
            "n": n,
            "n_angles": case["n_angles"],
            "theta_min": case["theta_min"],
            "theta_max": case["theta_max"],
            "sigma_rel": case["sigma_rel"],
            "RelL2": met["RelL2"],
            "RelL2_percent": met["RelL2_percent"],
            "SSIM": met["SSIM"],
            "time_seconds": elapsed
        })

        save_reconstruction_figure(
            gt=gt,
            rec_img=rec_img,
            sino=sino,
            method_name=f"Tikhonov_lambda_{lam}",
            phantom_name=phantom_name,
            case_name=case_name,
            output_dir=case_dir
        )


    # ESTUDI DE LAMBDA: TV
 
    for lam in lam_tv_values:

        print(f"\nTV: lambda = {lam}")

        t0 = time.perf_counter()

        rec_img = rec.recon_tv_manual(
            sino,
            theta,
            out_size=n,
            lam=lam,
            tau=tau_tv,
            n_iter=n_iter_tv
        )

        elapsed = time.perf_counter() - t0
        met = ut.metrics(rec_img, gt)

        rows.append({
            "phantom": phantom_name,
            "case": case_name,
            "method": "TV",
            "lambda": lam,
            "tau": tau_tv,
            "n_iter": n_iter_tv,
            "n": n,
            "n_angles": case["n_angles"],
            "theta_min": case["theta_min"],
            "theta_max": case["theta_max"],
            "sigma_rel": case["sigma_rel"],
            "RelL2": met["RelL2"],
            "RelL2_percent": met["RelL2_percent"],
            "SSIM": met["SSIM"],
            "time_seconds": elapsed
        })

        save_reconstruction_figure(
            gt=gt,
            rec_img=rec_img,
            sino=sino,
            method_name=f"TV_lambda_{lam}",
            phantom_name=phantom_name,
            case_name=case_name,
            output_dir=case_dir
        )

    # Guardem el resultat de l'estudi de lambda en un CSV
    csv_path = os.path.join(output_root, "lambda_study_results.csv")

    fieldnames = [
        "phantom",
        "case",
        "method",
        "lambda",
        "tau",
        "n_iter",
        "n",
        "n_angles",
        "theta_min",
        "theta_max",
        "sigma_rel",
        "RelL2",
        "RelL2_percent",
        "SSIM",
        "time_seconds"
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultats de l'estudi de lambda guardats a: {csv_path}")
    print(f"Figures guardades a: {case_dir}")


# ..........
# EXECUCIÓ
# ..........

if __name__ == "__main__":

    mode = "single"   # Opcions: "single", "all" o "lambda_study"

    if mode == "single":

        # Triem el phantom i el cas experimental
        phantom_name = "regions_suaus"
        phantoms = make_all_phantoms(512)
        gt_phantom = phantoms[phantom_name]

        case_name = "limited_angle"
        case = EXPERIMENT_CASES[case_name]

        # Executem l'experiment
        gt, sino, theta, results = run_one_case(
            gt=gt_phantom,
            n=512,
            n_angles=case["n_angles"],
            theta_min=case["theta_min"],
            theta_max=case["theta_max"],
            sigma_rel=case["sigma_rel"],
            rng_seed=0,
            lam_tikhonov=1e-3,
            lam_tv=0.001,
            tau_tv=0.01,
            n_iter_tv=50
        )

        # Mostrem els resultats per consola i visualitzem les figures
        print(f"\nPHANTOM: {phantom_name}")
        print(f"CAS: {case_name}")
        print(f"DESCRIPCIÓ: {case['description']}")
        print("=" * 50)

        for method_name, result in results.items():

            rec_img = result["image"]
            met = result["metrics"]
            elapsed = result["time"]

            print("\n" + method_name)
            print("-" * len(method_name))
            print(f"Error relatiu L2: {met['RelL2_percent']:.2f}%")
            print(f"SSIM: {met['SSIM']:.4f}")
            print(f"Temps: {elapsed:.3f} s")

            ut.show_triplet(
                title=f"{method_name} - {case_name}",
                gt=gt,
                rec=rec_img,
                sino=sino
            )

    elif mode == "all":

        phantoms_to_run = [
            "shepp_logan",
            "regions_suaus",
            "frontera_koch",
            "triangular_iterat"
        ]

        cases_to_run = [
            "full_data",
            "full_data_noise",
            "sparse_angle",
            "sparse_angle_noise",
            "limited_angle",
            "limited_angle_noise"
        ]

        run_all_experiments(
            phantoms_to_run=phantoms_to_run,
            cases_to_run=cases_to_run,
            n=512,
            rng_seed=0,
            lam_tikhonov=1e-3,
            lam_tv=0.001,
            tau_tv=0.01,
            n_iter_tv=50,
            output_root="resultats"
        )

    elif mode == "lambda_study":

        run_lambda_study(
            phantom_name="trianglar_iterat",
            case_name="sparse_angle_noise",
            n=512,
            rng_seed=0,
            lam_tikhonov_values=[1e-4, 1e-3, 1e-2, 1e-1],
            lam_tv_values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            tau_tv=0.01,
            n_iter_tv=50,
            output_root="resultats_lambda"
        )
        

    else:
        raise ValueError("mode ha de ser 'single', 'all' o 'lambda_study'.")