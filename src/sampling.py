import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def sample_srs(X, y, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[idx], y.iloc[idx]

def sample_stratified(X, y, n, seed=42):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    idx0 = rng.choice(np.where(y.values == 0)[0], size=n0, replace=False)
    idx1 = rng.choice(np.where(y.values == 1)[0], size=n1, replace=False)
    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return X.iloc[idx], y.iloc[idx]

def sample_systematic(X, y, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    k = max(1, len(X) // n)
    start = int(rng.integers(0, k))
    sys_idx = idx[start::k][:n]
    return X.iloc[sys_idx], y.iloc[sys_idx]

def sample_cluster(X, y, n, seed=42, k_clusters=10):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = min(k_clusters, max(2, len(X) // 5))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(Xs)

    rng = np.random.default_rng(seed)
    clusters = np.unique(labels)
    rng.shuffle(clusters)

    chosen = []
    for c in clusters:
        chosen.extend(np.where(labels == c)[0].tolist())
        if len(chosen) >= n:
            break

    chosen = np.array(chosen)
    if len(chosen) > n:
        chosen = rng.choice(chosen, size=n, replace=False)
    return X.iloc[chosen], y.iloc[chosen]

def sample_bootstrap(X, y, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=True)
    return X.iloc[idx], y.iloc[idx]
