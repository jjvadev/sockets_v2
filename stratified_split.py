import numpy as np


def stratified_sample(X, Y, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(Y)
    N = Y.shape[0]

    selected = []
    for c in classes:
        idx_c = np.where(Y == c)[0]
        n_c = round(len(idx_c) / N * n_samples)
        n_c = min(n_c, len(idx_c))
        chosen = rng.choice(idx_c, size=n_c, replace=False)
        selected.append(chosen)

    idx_all = np.concatenate(selected)

    if len(idx_all) < n_samples:
        remaining = np.setdiff1d(np.arange(N), idx_all)
        extra = rng.choice(remaining, size=n_samples - len(idx_all), replace=False)
        idx_all = np.concatenate([idx_all, extra])
    elif len(idx_all) > n_samples:
        idx_all = rng.choice(idx_all, size=n_samples, replace=False)

    rng.shuffle(idx_all)
    return X[idx_all], Y[idx_all]


def stratified_split_workers(X, Y, n_workers, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(Y)

    worker_indices = [[] for _ in range(n_workers)]

    for c in classes:
        idx_c = np.where(Y == c)[0]
        rng.shuffle(idx_c)
        for i, idx in enumerate(idx_c):
            worker_indices[i % n_workers].append(idx)

    batches = []
    for w in range(n_workers):
        w_idx = np.array(worker_indices[w], dtype=np.int64)
        rng.shuffle(w_idx)
        batches.append((X[w_idx], Y[w_idx]))

    return batches


def print_distribution(batches, label="Workers", max_classes_to_show=20):
    all_y = np.concatenate([Y_w for _, Y_w in batches])
    classes = np.unique(all_y)

    print(f"\n  Distribución de clases por worker [{label}]")
    print(f"  Clases totales detectadas: {len(classes)}")

    for w, (_, Y_w) in enumerate(batches):
        unique, counts = np.unique(Y_w, return_counts=True)
        stats = dict(zip(unique.tolist(), counts.tolist()))

        preview = []
        for c in classes[:max_classes_to_show]:
            preview.append(f"{c}:{stats.get(c, 0)}")

        print(
            f"  W{w:<3} total={len(Y_w):>6} | "
            f"{'  '.join(preview)}"
            + ("  ..." if len(classes) > max_classes_to_show else "")
        )