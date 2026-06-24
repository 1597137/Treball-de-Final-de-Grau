"""
Aquest mòdul conté funcions per generar diferents phantoms de prova
per a experiments de reconstrucció d'imatges mèdiques.

Conté quatre tipus de phantoms:
1. Shepp-Logan: un phantom clàssic de referència en tomografia.
2. Model de regions suaus: un phantom amb contorn exterior irregular però suau,
   amb una closca i una estructura interior.    
3. Model amb frontera de Koch: un phantom amb contorn fractal tipus floc de Koch.
4. Model triangular iterat: un phantom amb una figura central i 6 còpies petites al voltant,
   amb un contorn complex i simètric.
"""

import numpy as np
from matplotlib.path import Path

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# ....................
# FUNCIONS AUXILIARS
# ....................

def polygon_mask(X, Y, points):
    """
    Retorna una màscara booleana que indica quins punts de la graella
    (X,Y) es troben dins del polígon definit pels vèrtexs "points".

    Parameters
    ----------
    X : np.ndarray
        Matriu 2D amb les coordenades X de la graella.
    Y : np.ndarray
        Matriu 2D amb les coordenades Y de la graella. 
    points : np.ndarray
        Array de forma (N, 2) amb els vèrtexs del polígon   
    
    Returns
    -------
    mask : np.ndarray
        Matriu booleana de la mateixa forma que X i Y, amb True per als punts dins del polígon. 
    """
    coords = np.vstack((X.ravel(), Y.ravel())).T
    path = Path(points)
    mask = path.contains_points(coords)

    return mask.reshape(X.shape)


def koch_segment(p1, p2):
    """
    Retorna els punts que defineixen un segment de Koch entre els punts p1 i p2.
    El segment de Koch es construeix dividint el segment en tres parts iguals,
    i substituint la part central per un triangle equilàter.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    v = p2 - p1

    a = p1
    b = p1 + v / 3
    d = p1 + 2 * v / 3

    # Rotació de 60 graus per obtenir el punt superior del triangle equilàter
    angle = -np.pi / 3
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    c = b + rot @ (v / 3)

    return [a, b, c, d]

def koch_snowflake_points(iterations=3, scale=0.85):
    """
    Genera els punts del contorn d'un floc de Koch a partir d'un triangle inicial.
    
    Parameters
    ----------
    iterations : int   
        Nombre d'iteracions per generar el fractal.
    scale : float
        Factor d'escala per ajustar la mida del floc dins del domini [-1, 1] x [-1, 1].
    
    Returns
    -------
    points : np.ndarray
        Array de forma (N, 2) amb els punts del contorn del floc de Koch.
    """
    # Triangle inicial
    angles = np.array([
        np.pi/2,
        np.pi/2 + 2*np.pi/3,
        np.pi/2 + 4*np.pi/3
    ])

    points = [scale * np.array([np.cos(a), np.sin(a)]) for a in angles]
    points.append(points[0])

    # Iteracions per generar el fractal de Koch
    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            new_points.extend(koch_segment(points[i], points[i+1]))
        new_points.append(new_points[0])
        points = new_points

    return np.array(points)


def draw_triangle(X, Y, img, p1, p2, p3, value=0.0):
    """
    Dibuixa un triangle definit pels punts p1, p2 i p3 a la imatge img,
    assignant el valor especificat als píxels dins del triangle.   
    """
    points = np.array([p1, p2, p3])
    mask = polygon_mask(X, Y, points)
    img[mask] = value

def draw_basic_star(X, Y, img, cx, cy, scale=1.0, angle0=np.pi/2, value=0.0):
    """
    Dibuixa una estrella de 6 puntes centrada a (cx, cy) amb una mida determinada per "scale"
    i una orientació inicial "angle0". Assigna el valor especificat als píxels dins de l'estrella.
    """
    R = 0.22 * scale

    angles = np.array([
        angle0,
        angle0 + np.pi / 3,
        angle0 + 2*np.pi / 3,
        angle0 + 3*np.pi / 3,
        angle0 + 4*np.pi / 3,
        angle0 + 5*np.pi / 3
    ])

    outer = np.column_stack((
        cx + R * np.cos(angles),
        cy + R * np.sin(angles)
    ))

    center = np.array([cx, cy])

    # Dibuixem els 6 triangles exteriors
    for i in range(6):
        p1 = outer[i]
        p2 = outer[(i + 1) % 6]
        midpoint = (p1 + p2) / 2
        p3 = center + 2.0 * (midpoint - center)

        draw_triangle(X, Y, img, p1, p2, p3, value=value)

    # Dibuixem el triangle central
    draw_triangle(X, Y, img, outer[2], outer[4], outer[0], value=value)


def draw_level1(X, Y, img, cx, cy, scale=1.0, phi=0.0, value=0.0):
    """
    Dibuixa la figura de nivell 1: una estrella gran al centre i 6 estrelles petites al voltant, 
    amb una orientació determinada per "phi".
    """
    # Figura central: estrella gran
    draw_basic_star(
        X, Y, img,
        cx=cx,
        cy=cy,
        scale=1.0 * scale,
        angle0=3*np.pi/2 + phi,   
        value=value
    )

    # Paràmetres per les 6 figures petites
    child_scale = 0.575 * scale
    child_dist = 0.44 * scale

    # Orientació de les estrelles petites respecte a la figura central
    angles_small = [
        2*np.pi/3,   # dalt
        np.pi,       # dalt-esquerra
        2*np.pi/3,   # baix-esquerra
        np.pi,       # baix
        2*np.pi/3,   # baix-dreta
        np.pi        # dalt-dreta
    ]

    # Dibuixem les 6 figures petites al voltant de la figura central
    for k in range(6):
        a = phi + np.pi / 2 + k * np.pi / 3

        child_cx = cx + child_dist * np.cos(a)
        child_cy = cy + child_dist * np.sin(a)

        draw_basic_star(
            X, Y, img,
            cx=child_cx,
            cy=child_cy,
            scale=child_scale,
            angle0=angles_small[k] + phi,
            value=value
        )


# .......................
# PHANTOM 1: SHEPP-LOGAN
# .......................

def make_phantom_shepp_logan(n = 256):
    """
    Genera el phantom de Shepp-Logan redimensionat a mida n x n.
    Aquest phantom és un estàndard de referència en tomografia computada.
    
    Parameters
    ----------
    n : int
        Mida de la imatge final (n x n píxels).
    
    Returns
    -------
    np.ndarray
        Matriu que representa la imatge del phantom de Shepp-Logan, amb valors normalitzats entre 0 i 1.
    """
    
    ph = shepp_logan_phantom().astype(np.float64)
    # Redimensionem la imatge a mida n x n amb anti-aliasing per evitar distorsions.
    ph = resize(ph, (n, n), anti_aliasing=True) 

    # Normalitzem els valors de la imatge a l'interval [0, 1] per assegurar una escala d'intensitat consistent.
    # Afegim un petit valor (1e-12) al denominador per evitar divisions per zero en casos on la imatge sigui constant.
    ph = (ph - ph.min()) / (ph.max() - ph.min() + 1e-12) 
  
    return ph 


# ........................
# PHANTOM 2: REGIONS SUAU
# ........................

def make_phantom_suau(n=256):
    """
    Genera un phantom amb contorn exterior irregular però suau, amb una closca i una estructura interior.
    Aquesta figura és útil per estudiar com els algoritmes de reconstrucció tracten amb transicions suaus i geometries més orgàniques.
    """
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)

    R = np.sqrt(X**2 + Y**2)
    T = np.arctan2(Y, X)

    img = np.zeros((n, n))

    # Contorn exterior de la closca amb una forma irregular però suau
    r_outer = (
        0.90
        + 0.035*np.sin(3*T + 0.4)
        + 0.025*np.sin(7*T - 0.8)
        + 0.015*np.cos(11*T)
    )

    # Contorn interior de la closca
    r_inner = r_outer - 0.10

    # Intensitat de la closca exterior
    inside = R <= r_outer
    img[inside] = 0.20

    # Intensitat de la closca interior
    shell = (R <= r_outer) & (R >= r_inner)
    img[shell] = 0.75

    # Estructura interior: una estrella amb 8 puntes dins de la closca
    interior = R < r_inner
    r_star = 0.48 + 0.18 * np.sin(8 * T)
    star = (R < r_star) & interior
    img[star] = 0.55

    return img


# ..............................
# PHANTOM 3: CONTORN TIPUS KOCH
# ..............................

def make_phantom_koch(n=256):
    """
    Genera un phantom amb un contorn fractal tipus floc de Koch.
    Aquesta figura és útil per estudiar com els algoritmes de reconstrucció tracten amb geometries més intricades i amb detalls finament definits.
    """
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)

    img = np.zeros((n, n))

    # Contorn exterior del floc de Koch
    points = koch_snowflake_points(iterations=4, scale=0.88)

    # Creem una màscara booleana que indica quins punts de la graella (X, Y) es troben dins del polígon definit pels punts del floc de Koch.
    mask = polygon_mask(X, Y, points)

    # Inicialitzem la imatge amb fons negre (intensitat 0.0)
    img[:, :] = 0.0

    # Interior del floc de Koch = 0.75
    img[mask] = 0.75

    # Afegim una estructura interior: un floc de Koch més petit dins del floc principal
    inner_points = koch_snowflake_points(iterations=4, scale=0.45)
    inner_mask = polygon_mask(X, Y, inner_points)

    # Assignem una intensitat més baixa a l'interior del floc de Koch petit per crear un efecte de profunditat i complexitat visual.
    img[inner_mask] = 0.35

    return img


#.....................................
# PHANTOM 4: MODEL TRIANGULAR ITERAT
#.....................................

def make_phantom_triangular_iterat(n=256):
    """
    Genera un phantom amb una figura central i 6 còpies petites al voltant,
    amb un contorn complex i simètric. Aquesta figura és útil per estudiar com els algoritmes de reconstrucció tracten amb patrons repetitius i simètrics, 
    així com amb transicions suaus entre les diferents regions de la imatge.
    """

    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)

    img = np.ones((n, n), dtype=np.float64)

    # Factor per reduir la figura dins del domni [-1, 1] x [-1, 1].
    global_scale = 0.85

    # Figura principal de nivell 1.
    draw_level1(X, Y, img,
                cx=0.0, cy=0.0,
                scale=1.0 * global_scale,
                phi=np.pi/6,
                value=0.0)

    # Paràmetres de les còpies petites al voltant de la figura central.
    motif_scale = 0.57 * global_scale
    motif_dist = 0.75 * global_scale

    # Sis còpies petites al voltant de la figura central.
    for k in range(6):
        a = np.pi / 2 + k * np.pi / 3

        child_cx = motif_dist * np.cos(a)
        child_cy = motif_dist * np.sin(a)

        draw_level1(X, Y, img,
                    cx=child_cx,
                    cy=child_cy,
                    scale=motif_scale,
                    phi=0.0,
                    value=0.0)

    return img



# ..........................................
# NORMALITZACIÓ DE LES NORMES DELS PHANTOMS
# ..........................................

def normalize_to_l2(img, target_norm):
    """
    Reescala una imatge perquè tingui una norma L2 determinada.

    Aquesta normalització no canvia la geometria del phantom,
    només l'escala dels valors d'intensitat.
    """
    norm = np.linalg.norm(img)

    if norm < 1e-12:
        return img

    return img * (target_norm / norm)


def make_all_phantoms(n=256):
    """
    Genera tots els phantoms definits en aquest mòdul i els normalitza perquè tinguin la mateixa norma L2.
    Retorna un diccionari amb els phantoms.
    """

    shepp_logan = make_phantom_shepp_logan(n)
    suau = make_phantom_suau(n)
    koch = make_phantom_koch(n)
    triangular_iterat = make_phantom_triangular_iterat(n)

    # Invertim els valors del phantom triangular iterat per assegurar que el fons sigui fosc i les figures clares, 
    # mantenint la coherència amb els altres phantoms.
    triangular_iterat = 1.0 - triangular_iterat

    # Norma de referència: Shepp-Logan
    target_norm = np.linalg.norm(shepp_logan)

    suau = normalize_to_l2(suau, target_norm)
    koch = normalize_to_l2(koch, target_norm)
    triangular_iterat = normalize_to_l2(triangular_iterat, target_norm)

    return {
        "shepp_logan": shepp_logan,
        "suau": suau,
        "koch": koch,
        "triangular_iterat": triangular_iterat,
    }