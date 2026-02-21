import numpy as np

def greedy_community_detection(W, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    labels = np.arange(N)
    for _ in range(n_iter):
        order = rng.permutation(N)
        changed = False
        for i in order:
            nbr = {}
            for j in range(N):
                if i != j and W[i,j] > 0:
                    c = labels[j]
                    nbr[c] = nbr.get(c, 0.0) + W[i,j]
            if nbr:
                best = max(nbr, key=nbr.get)
                if best != labels[i]:
                    labels[i] = best
                    changed = True
        if not changed:
            break
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[l] for l in labels])

def hhi_separability(W):
    N = W.shape[0]
    B = W.sum(axis=1)
    hhi = []
    for i in range(N):
        if B[i] < 1e-15:
            continue
        p = W[i] / B[i]
        hhi.append(np.sum(p**2))
    return (N-1) * np.mean(hhi) if hhi else 1.0

def weighted_modularity(W, labels=None, seed=0):
    if labels is None:
        labels = greedy_community_detection(W, seed=seed)
    m2 = W.sum()
    if m2 < 1e-15:
        return 0.0
    k = W.sum(axis=1)
    same = (labels[:,None] == labels[None,:]).astype(float)
    Q = np.sum((W - np.outer(k,k)/m2) * same) / m2
    return float(Q)

def fiedler_value(W):
    B = W.sum(axis=1)
    L = np.diag(B) - W
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(B > 1e-15, B**-0.5, 0.0)
    L_sym = np.diag(d) @ L @ np.diag(d)
    eigvals = np.sort(np.linalg.eigvalsh(L_sym))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0

def compute_all(W, seed=0):
    labels = greedy_community_detection(W, seed=seed)
    return {
        "chi_hhi":  hhi_separability(W),
        "Q_star":   weighted_modularity(W, labels=labels),
        "lambda2":  fiedler_value(W),
        "labels":   labels,
    }
