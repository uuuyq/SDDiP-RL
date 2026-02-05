import numpy as np

def bundle_features(bundle, pi_t, w_prev, eta_prev, t):
    """
    bundle: list of (pi_j, g_j, phi_j)
    """

    pi, g, phi = bundle[-1]

    g_norms = [np.linalg.norm(gj) for _, gj, _ in bundle]
    pi_norms = [np.linalg.norm(pj) for pj, _, _ in bundle]

    features = [
        eta_prev,
        np.linalg.norm(w_prev)**2,
        eta_prev * np.linalg.norm(w_prev)**2,
        t,

        phi,
        np.linalg.norm(pi)**2,
        np.linalg.norm(g)**2,

        np.mean(g),
        np.var(g),
        np.min(g),
        np.max(g),

        np.mean(pi),
        np.var(pi),
        np.min(pi),
        np.max(pi),

        min(g_norms),
        max(g_norms),
        min(pi_norms),
        max(pi_norms)
    ]

    return np.array(features, dtype=np.float32)
