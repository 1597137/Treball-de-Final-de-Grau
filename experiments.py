
"""
Definició dels casos experimentals per a la reconstrucció tomogràfica.
Cada cas experimental es defineix amb els següents paràmetres:
    - description: Descripció del cas experimental.
    - n_angles: Nombre d'angles de projecció.
    - theta_min: Angle mínim de projecció (en graus).
    - theta_max: Angle màxim de projecció (en graus).
    - sigma_rel: Soroll relatiu a afegir a les dades de projecció (0.0 significa sense soroll).
"""

# ....................
# CASOS EXPERIMENTALS
# ....................

EXPERIMENT_CASES = {

    # ....................................
    # CAS 1: DADES COMPLETES SENSE SOROLL
    # ....................................
    # Dades completes sense soroll, amb 180 angles equidistats entre 0 i 180 graus.
    "full_data": {
        "description": "Dades completes sense soroll",
        "n_angles": 180,
        "theta_min": 0,
        "theta_max": 180,
        "sigma_rel": 0.0,
    },

    # ..................................
    # CAS 2: DADES COMPLETES AMB SOROLL
    # ..................................
    # Dades completes amb soroll gaussià, amb 180 angles equidistats entre 0 i 180 graus.
    "full_data_noise": {
        "description": "Dades completes amb soroll gaussià",
        "n_angles": 180,
        "theta_min": 0,
        "theta_max": 180,
        "sigma_rel": 0.03,
    },

    # ....................
    # CAS 3: SPARSE-ANGLE
    # ....................
    # Només es disposa d'un nombre reduït d'angles de projecció, amb 30 angles equidistats entre 0 i 180 graus.
    "sparse_angle": {
        "description": "Sparse-angle sense soroll",
        "n_angles": 30,
        "theta_min": 0,
        "theta_max": 180,
        "sigma_rel": 0.0,
    },

    # ...............................
    # CAS 4: SPARSE-ANGLE AMB SOROLL
    # ...............................
    # Només es disposa d'un nombre reduït d'angles de projecció, amb 30 angles equidistats entre 0 i 180 graus, amb soroll gaussià.
    "sparse_angle_noise": {
        "description": "Sparse-angle amb soroll gaussià",
        "n_angles": 30,
        "theta_min": 0,
        "theta_max": 180,
        "sigma_rel": 0.03,
    },

    # .....................
    # CAS 5: LIMITED-ANGLE
    # .....................
    # Només es disposa d'un nombre reduït d'angles de projecció, amb 60 angles equidistats entre 0 i 70 graus.
    "limited_angle": {
        "description": "Limited-angle sense soroll",
        "n_angles": 60,
        "theta_min": 0,
        "theta_max": 70,
        "sigma_rel": 0.0,
    },

    # ................................
    # CAS 6: LIMITED-ANGLE AMB SOROLL
    # ................................
    # Només es disposa d'un nombre reduït d'angles de projecció, amb 60 angles equidistats entre 0 i 70 graus, amb soroll gaussià.
    "limited_angle_noise": {
        "description": "Limited-angle amb soroll gaussià",
        "n_angles": 60,
        "theta_min": 0,
        "theta_max": 70,
        "sigma_rel": 0.03,
    },
}