
"""
Funcions per a la reconstrucció de imatges a partir de dades de projecció.

Aquest mòdul implementa diverses tècniques de reconstrucció, incloent:
- Projector directe i retroprojector manual (discretització de la transformada de Radon).
- Filtrat de sinogrames amb filtres com ramp i Shepp-Logan.
- Reconstrucció per retroprojecció filtrada (FBP).
- Resolució de problemes de Tikhonov amb gradient conjugat.
- Reconstrucció amb regularització de total variation (TV) mitjançant un esquema iteratiu.

L'objectiu és proporcionar una implementació manual i didàctica de les tècniques de reconstrucció, sense dependre de funcions predefinides, tot i que es poden utilitzar paquets com `scipy` i `skimage` per a algunes operacions específiques com el denoising TV.
"""

import numpy as np

from scipy.sparse.linalg import cg, LinearOperator
from skimage.restoration import denoise_tv_chambolle


# .....................
# GRAELLA DE LA IMATGE
# .....................

def image_grid(n):
    """
    Genera una graella de coordenades (X, Y) per a una imatge quadrada de mida n x n.
    Les coordenades es normalitzen a l'interval [-1, 1].
    """
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)

    return X, Y


# .........................
# PROJECTOR DIRECTE MANUAL
# .........................

def forward_project_manual(img, theta, n_detectors=None):
    """
    Calcula el sinograma d'una imatge mitjançant un projector directe manual.
    Per cada angle, es calcula la projecció de la imatge sobre l'eix del detector, 
    utilitzant interpolació lineal per a les coordenades contínues del detector.

    Parameters
    ----------  
    img : np.ndarray
        Imatge d'entrada de mida n x n.
    
    theta : np.ndarray
        Angles de projecció, en graus.

    n_detectors : int, optional
        Nombre de posicions del detector. Si no s'indica, es pren n.

    Returns
    -------
    sino : np.ndarray
        Sinograma de mida n_detectors x len(theta).
    """

    n = img.shape[0]
    assert img.shape[0] == img.shape[1], "La imatge ha de ser quadrada."

    if n_detectors is None:
        n_detectors = n

    X, Y = image_grid(n)

    # Definim l'interval de coordenades del detector. Per a una imatge normalitzada a [-1, 1], 
    # les coordenades del detector també es poden normalitzar a [-sqrt(2), sqrt(2)] per cobrir tota la diagonal de la imatge.
    s_min = -np.sqrt(2)
    s_max = np.sqrt(2)

    sino = np.zeros((n_detectors, len(theta)), dtype=np.float64)

    x_flat = X.ravel()
    y_flat = Y.ravel()
    img_flat = img.ravel()

    for k, angle_deg in enumerate(theta):
        angle = np.deg2rad(angle_deg)

        c = np.cos(angle)
        s = np.sin(angle)

        # Coordenada contínua del detector per a cada píxel de la imatge
        detector_coord = x_flat * c + y_flat * s

        # Mapeig de la coordenada contínua del detector a l'índex discret del sinograma 
        detector_pos = (detector_coord - s_min) / (s_max - s_min) * (n_detectors - 1)

        # Identifiquem els dos detectors més propers per a cada píxel
        i0 = np.floor(detector_pos).astype(int)
        i1 = i0 + 1

        # Pesos per a la interpolació lineal
        w1 = detector_pos - i0
        w0 = 1.0 - w1

        valid0 = (i0 >= 0) & (i0 < n_detectors)
        valid1 = (i1 >= 0) & (i1 < n_detectors)

        # Afegim els valors de la imatge als detectors corresponents amb els pesos d'interpolació
        np.add.at(
            sino[:, k],
            i0[valid0],
            w0[valid0] * img_flat[valid0]
        )

        np.add.at(
            sino[:, k],
            i1[valid1],
            w1[valid1] * img_flat[valid1]
        )

    return sino


# ......................
# RETROPROJECCIÓ MANUAL
# ......................

def backproject_manual(sino, theta, out_size):
    """
    Retroprojecció d'un sinograma per reconstruir una imatge.
    Aquesta funció implementa la retroprojecció manual, on cada projecció del sinograma 
    es distribueix sobre la imatge reconstruïda segons l'angle corresponent.

    Parameters
    ----------
    sino : np.ndarray
        Sinograma de mida n_detectors x n_angles.

    theta : np.ndarray
        Angles de projecció, en graus.
    
    out_size : int
        Mida de la imatge reconstruïda.

    Returns
    -------
    img : np.ndarray
        Imatge reconstruïda de mida out_size x out_size.
    """

    n_detectors = sino.shape[0]

    X, Y = image_grid(out_size)

    s_min = -np.sqrt(2)
    s_max = np.sqrt(2)

    x_flat = X.ravel()
    y_flat = Y.ravel()

    img_flat = np.zeros(out_size*out_size, dtype=np.float64)

    detector_grid = np.arange(n_detectors)

    for k, angle_deg in enumerate(theta):
        angle = np.deg2rad(angle_deg)

        c = np.cos(angle)
        s = np.sin(angle)

        detector_coord = x_flat * c + y_flat * s

        detector_pos = (detector_coord - s_min) / (s_max - s_min) * (n_detectors - 1)

        # Interpolació lineal del sinograma per obtenir els valors corresponents a cada píxel de la imatge
        values = np.interp(
            detector_pos,
            detector_grid,
            sino[:, k],
            left=0.0,
            right=0.0
        )

        img_flat += values

    img = img_flat.reshape((out_size, out_size))

    # Normalització pel nombre d'angles
    # S'utilitza per visualització i per FBP.
    img = img / len(theta)

    return img


def backproject_adjoint_manual(sino, theta, out_size):
    """
    Retroprojecció d'un sinograma per reconstruir una imatge, sense normalització pel nombre d'angles.
    Aquesta funció implementa la retroprojecció manual, on cada projecció del sinograma es distribueix sobre la imatge reconstruïda segons l'angle corresponent.

    Parameters
    ----------
    sino : np.ndarray
        Sinograma de mida n_detectors x n_angles.

    theta : np.ndarray
        Angles de projecció, en graus.
    
    out_size : int
        Mida de la imatge reconstruïda.

    Returns
    -------
    img : np.ndarray
        Imatge reconstruïda de mida out_size x out_size.
    """

    n_detectors = sino.shape[0]

    X, Y = image_grid(out_size)

    s_min = -np.sqrt(2)
    s_max = np.sqrt(2)

    x_flat = X.ravel()
    y_flat = Y.ravel()

    img_flat = np.zeros(out_size * out_size, dtype=np.float64)

    detector_grid = np.arange(n_detectors)

    for k, angle_deg in enumerate(theta):
        angle = np.deg2rad(angle_deg)

        c = np.cos(angle)
        s = np.sin(angle)

        detector_coord = x_flat * c + y_flat*s
        detector_pos = (detector_coord - s_min) / (s_max - s_min) * (n_detectors - 1)

        values = np.interp(
            detector_pos,
            detector_grid,
            sino[:, k],
            left=0.0,
            right=0.0
        )

        img_flat += values

    img = img_flat.reshape((out_size, out_size))

    return img


# ......................
# FILTRAT DEL SINOGRAMA
# ......................

def filter_sinogram_manual(sino, filter_name="ramp"):
    """
    Filtra  cada projecció del sinograma.
    En la reconstrucció FBP, és habitual filtrar el sinograma abans de la retroprojecció per millorar la qualitat de la imatge reconstruïda.

    Parameters
    ----------
    sino : np.ndarray
        Sinograma de mida n_detectors x n_angles.

    filter_name : str or None
        Nom del filtre a aplicar. Opcions: "ramp", "shepp-logan" o None.
    
    Returns
    -------
    sino_filt : np.ndarray
        Sinograma filtrat de la mateixa mida que l'entrada. 
    """

    if filter_name is None:
        return sino.copy()

    n_detectors, _ = sino.shape

    # Freqüències associades a l'eix del detector
    # La funció np.fft.fftfreq retorna les freqüències corresponents a la FFT d'una senyal de longitud n_detectors.
    freqs = np.fft.fftfreq(n_detectors)

    # Definim el filtre en el domini de la freqüència. Inicialment, comencem amb un filtre ramp (|f|).
    filt = np.abs(freqs)

    if filter_name == "ramp":
        pass

    elif filter_name == "shepp-logan":
        # El filtre de Shepp-Logan és un filtre ramp modificat amb una funció sinc per reduir l'efecte de les altes freqüències.
        filt = filt * np.sinc(freqs / 2)

    else:
        raise ValueError("filter_name ha de ser 'ramp', 'shepp-logan' o None.")

    # Transformada de Fourier del sinograma al llarg de l'eix del detector.
    sino_fft = np.fft.fft(sino, axis=0)

    # Multipliquem cada projecció del sinograma per l'espectre del filtre corresponent.
    sino_fft_filt = sino_fft * filt[:, np.newaxis]

    # Transformada inversa de Fourier per obtenir el sinograma filtrat.
    sino_filt = np.real(np.fft.ifft(sino_fft_filt, axis=0))

    return sino_filt

# ...........
# FBP MANUAL
# ...........

def recon_fbp_manual(sino, theta, out_size, filter_name="ramp"):
    """
    Reconstrueix una imatge a partir d'un sinograma utilitzant la retroprojecció filtrada (FBP).
    Aquesta funció aplica un filtre al sinograma i després realitza la retroprojecció per obtenir la imatge reconstruïda.

    Parameters
    ----------
    sino : np.ndarray
        Sinograma de mida n_detectors x n_angles.

    theta : np.ndarray
        Angles de projecció, en graus.
    
    out_size : int
        Mida de la imatge reconstruïda.

    filter_name : str or None
        Nom del filtre a aplicar al sinograma abans de la retroprojecció. Opcions: "ramp", "shepp-logan" o None.

    Returns
    -------
    rec : np.ndarray
        Imatge reconstruïda de mida out_size x out_size.
    """

    sino_filt = filter_sinogram_manual(sino, filter_name=filter_name)

    rec = backproject_manual(
        sino_filt,
        theta,
        out_size=out_size
    )

    return rec


# ..........................
# OPERADORS MANUALS A I A^T
# ..........................

def make_operators_manual(theta, img_shape, n_detectors=None):
    """
    Crea els operadors A i A^T per a una imatge de mida img_shape i angles theta.
    A és l'operador directe que transforma una imatge en un sinograma.
    A^T és l'operador de retroprojecció que transforma un sinograma en una imatge.

    Parameters
    ----------
    theta : np.ndarray
        Angles de projecció, en graus.

    img_shape : tuple
        Mida de la imatge (n, n).
    
    n_detectors : int, optional
        Nombre de posicions del detector. Si no s'indica, es pren n.

    Returns
    -------
    A : function
        Funció que aplica l'operador directe A a una imatge vectoritzada.
    
    AT : function
        Funció que aplica l'operador adjunt A^T a un sinograma vector
   
    m : int
        Dimensió del sinograma (n_detectors * len(theta)).
    
    N : int
        Dimensió de la imatge vectoritzada (n * n).
    """

    n = img_shape[0]
    assert img_shape[0] == img_shape[1], "La imatge ha de ser quadrada."

    if n_detectors is None:
        n_detectors = n

    N = n * n
    m = n_detectors * len(theta)

    def A(x_flat):
        """
        Aplica l'operador directe A a una imatge vectoritzada.
        """
        x = x_flat.reshape((n, n))

        sino = forward_project_manual(
            x,
            theta,
            n_detectors=n_detectors
        )

        return sino.ravel()

    def AT(y_flat):
        """
        Aplica l'operador adjunt A^T a un sinograma vectoritzat.
        """
        sino = y_flat.reshape((n_detectors, len(theta)))

        x = backproject_adjoint_manual(
            sino,
            theta,
            out_size=n
        )

        return x.ravel()

    return A, AT, m, N


# ......................................
# TIKHONOV MANUAL AMB GRADIENT CONJUGAT
# ......................................

def recon_tikhonov_cg_manual(sino, theta, out_size, lam=1e-2, cg_tol=1e-5, cg_maxiter=200):
    """
    Resolució del problema de Tikhonov mitjançant el mètode de gradient conjugat.
    El problema de Tikhonov és:

        min_f ||A f - m||_2^2 + lambda ||f||_2^2

    Aquesta funció utilitza els operadors A i A^T definits manualment per resoldre el sistema lineal associat al problema de Tikhonov.

    Parameters
    ----------
    sino : np.ndarray
        Sinograma observat.
    
    theta : np.ndarray
        Angles de projecció, en graus.

    out_size : int
        Mida de la imatge reconstruïda.

    lam : float
        Pes de la regularització Tikhonov.

    cg_tol : float
        Tolerància per al mètode de gradient conjugat.

    cg_maxiter : int
        Nombre màxim d'iteracions per al mètode de gradient conjugat.

    Returns
    -------
    rec : np.ndarray
        Imatge reconstruïda.
    """

    n_detectors = sino.shape[0]

    A, AT, _, N = make_operators_manual(
        theta,
        img_shape=(out_size, out_size),
        n_detectors=n_detectors
    )

    # Vector del costat dret del sistema lineal: A^T m
    b = AT(sino.ravel())

    # Operador del sistema: v -> (A^T A + lambda I)v
    def Mv(v):
        return AT(A(v)) + lam * v

    linop = LinearOperator(
        shape=(N, N),
        matvec=Mv,
        dtype=np.float64
    )

    x0 = np.zeros(N, dtype=np.float64)

    x, info = cg(
        linop,
        b,
        x0=x0,
        rtol=cg_tol,
        atol=0.0,
        maxiter=cg_maxiter
    )

    if info != 0:
        print(f"[WARN] CG no ha convergit completament (info={info}).")

    rec = x.reshape((out_size, out_size))

    return rec


# ...................................................................
# RECONSTRUCCIÓ AMB REGULARITZACIÓ TV MITJANÇANT UN ESQUEMA ITERATIU
# ...................................................................

def recon_tv_manual(sino, theta, out_size, lam=0.001, tau=0.0001, n_iter=50):
    """
    Reconstrucció d'imatges amb regularització de total variation (TV) mitjançant un esquema iteratiu.
    Aquesta funció implementa un mètode de descens de gradient amb regularització TV, on cada iteració consisteix en un pas de descens de gradient seguit d'un pas de denoising TV.
    
    El problema resolt és:
        min_f ||A f - m||_2^2 + lambda TV(f)
    
    Parameters
    ----------
    sino : np.ndarray
        Sinograma observat.
    
    theta : np.ndarray
        Angles de projecció, en graus.

    out_size : int
        Mida de la imatge reconstruïda.

    lam : float
        Pes del pas de regularització TV.

    tau : float
        Pas del gradient.

    n_iter : int
        Nombre d'iteracions.

    Returns
    -------
    rec : np.ndarray
        Imatge reconstruïda.
    """

    n_detectors = sino.shape[0]

    A, AT, _, N = make_operators_manual(
        theta,
        img_shape=(out_size, out_size),
        n_detectors=n_detectors
    )

    m = sino.ravel()

    # Inicialització amb la reconstrucció FBP
    f = recon_fbp_manual(sino, theta, out_size=out_size, filter_name="ramp").ravel()

    for _ in range(n_iter):
        # Calcul del residu: A f - m
        residual = A(f) - m

        # Calcul del gradient: A^T (A f - m)
        grad = AT(residual)

        # Normalització del gradient per evitar passos massa grans
        grad = grad / (np.linalg.norm(grad)+ 1e-12)

        # Actualització de la imatge amb un pas de gradient
        f = f - tau * grad

        # Reshape de f a la forma de la imatge per aplicar el denoising TV
        img = f.reshape((out_size, out_size))

        # Imposem valors físicament raonables per als phantoms normalitzats
        img = np.clip(img, 0, None)

        # Denoising TV amb el pes lam
        img = denoise_tv_chambolle(
            img,
            weight=lam
        )

        # Reshape de nou a vector per a la següent iteració
        img = np.clip(img, 0, None)

        f = img.ravel()

    rec = f.reshape((out_size, out_size))

    return rec