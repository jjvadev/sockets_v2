"""
stratified_split.py
────────────────────────────────────────────────────────────
Utilidad compartida por las 3 versiones de la red neuronal.

Provee dos funciones:
  · stratified_sample(X, Y, n_samples, seed)
        Extrae n_samples del dataset manteniendo la proporción
        original de cada clase (≈ misma cantidad de 0s, 1s, …, 9s).

  · stratified_split_workers(X, Y, n_workers, seed)
        Divide (X, Y) en n_workers partes de igual tamaño,
        garantizando que cada worker reciba ≈ la misma cantidad
        de ejemplos por clase.

Algoritmo (interleaved stratified split):
  1. Para cada clase c (0-9):
       · Obtener los índices donde Y == c  →  idx_c
       · Mezclarlos aleatoriamente
       · Repartirlos en round-robin entre los workers
  2. Mezclar los índices de cada worker (para no agrupar por clase)
  3. Devolver la lista de (X_w, Y_w) por worker

Ejemplo de distribución con 50 000 muestras y 4 workers:
  Clase 0: ~5 000 muestras  → cada worker recibe ~1 250
  Clase 1: ~5 500 muestras  → cada worker recibe ~1 375
  …
  Total por worker: ~12 500  (50 000 / 4)
"""

import numpy as np


def stratified_sample(X: np.ndarray, Y: np.ndarray,
                      n_samples: int, seed: int = 0):
    """
    Extrae exactamente n_samples del dataset (X, Y) respetando
    la proporción de cada clase.

    X : (784, N)  float
    Y : (N,)      int  con clases 0-9
    Retorna X_s (784, n_samples), Y_s (n_samples,)
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(Y)
    N       = Y.shape[0]

    selected = []
    for c in classes:
        idx_c  = np.where(Y == c)[0]
        n_c    = round(len(idx_c) / N * n_samples)
        n_c    = min(n_c, len(idx_c))
        chosen = rng.choice(idx_c, size=n_c, replace=False)
        selected.append(chosen)

    idx_all = np.concatenate(selected)

    if len(idx_all) < n_samples:
        remaining = np.setdiff1d(np.arange(N), idx_all)
        extra     = rng.choice(remaining, size=n_samples - len(idx_all), replace=False)
        idx_all   = np.concatenate([idx_all, extra])
    elif len(idx_all) > n_samples:
        idx_all = rng.choice(idx_all, size=n_samples, replace=False)

    rng.shuffle(idx_all)
    return X[:, idx_all], Y[idx_all]


def stratified_split_workers(X: np.ndarray, Y: np.ndarray,
                              n_workers: int, seed: int = 0):
    """
    Divide (X, Y) en n_workers particiones de igual tamaño,
    con distribución balanceada de clases en cada una.

    X : (784, N)
    Y : (N,)
    Retorna lista de n_workers tuplas (X_w, Y_w).
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(Y)

    worker_indices = [[] for _ in range(n_workers)]

    for c in classes:
        idx_c = np.where(Y == c)[0]
        rng.shuffle(idx_c)

        for i, idx in enumerate(idx_c):
            worker_indices[i % n_workers].append(idx)

    batches = []
    for w in range(n_workers):
        w_idx = np.array(worker_indices[w])
        rng.shuffle(w_idx)
        batches.append((X[:, w_idx], Y[w_idx]))

    return batches


def print_distribution(batches, label="Workers"):
    """
    Imprime la distribución de clases de cada batch/worker.
    """
    classes = list(range(10))
    print(f"\n  Distribución de clases por worker [{label}]")
    print(f"  {'Worker':<8}", end="")
    for c in classes:
        print(f"  [{c}]", end="")
    print(f"  {'Total':>7}")
    print(f"  {'-'*75}")

    for w, (_, Y_w) in enumerate(batches):
        print(f"  W{w:<7}", end="")
        for c in classes:
            print(f"  {np.sum(Y_w == c):>4}", end="")
        print(f"  {len(Y_w):>7}")
    print()