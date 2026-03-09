import argparse
from typing import Dict, List

import numpy as np
from torchvision import datasets

from connection import HOST, PORT, create_server_socket, recv_msg, send_msg
from stratified_split import print_distribution, stratified_split_workers


# ─────────────────────────────────────────────────────────────
# Modelo simple: softmax regression
# X: (784, N)
# W: (10, 784)
# b: (10, 1)
# ─────────────────────────────────────────────────────────────
def init_model(seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(10, 784)).astype(np.float32)
    b = np.zeros((10, 1), dtype=np.float32)
    return {"W": W, "b": b}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=0, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=0, keepdims=True)


def forward(params: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return params["W"] @ X + params["b"]


def predict(params: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    probs = softmax(forward(params, X))
    return np.argmax(probs, axis=0)


def accuracy(params: Dict[str, np.ndarray], X: np.ndarray, Y: np.ndarray) -> float:
    pred = predict(params, X)
    return float(np.mean(pred == Y))


def average_models(models: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    avg = {}
    for key in models[0].keys():
        avg[key] = np.mean([m[key] for m in models], axis=0).astype(np.float32)
    return avg


# ─────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────
def load_mnist(data_dir: str = "./data"):
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True)

    X_train = train_ds.data.numpy().reshape(-1, 28 * 28).T.astype(np.float32) / 255.0
    Y_train = train_ds.targets.numpy().astype(np.int64)

    X_test = test_ds.data.numpy().reshape(-1, 28 * 28).T.astype(np.float32) / 255.0
    Y_test = test_ds.targets.numpy().astype(np.int64)

    return X_train, Y_train, X_test, Y_test


# ─────────────────────────────────────────────────────────────
# Workers
# ─────────────────────────────────────────────────────────────
def accept_workers(server_sock, n_workers: int):
    workers = []
    print(f"\n  Esperando {n_workers} worker(s)...")

    while len(workers) < n_workers:
        conn, addr = server_sock.accept()
        print(f"  Worker conectado desde {addr[0]}:{addr[1]}")

        try:
            hello = recv_msg(conn)
            if hello.get("type") != "hello":
                print("  Mensaje inicial inválido. Cerrando conexión.")
                conn.close()
                continue

            worker_slot = len(workers)
            send_msg(conn, {
                "type": "hello_ack",
                "worker_slot": worker_slot,
            })
            workers.append((conn, addr, hello))

        except Exception as e:
            print(f"  Error durante handshake con {addr}: {e}")
            try:
                conn.close()
            except Exception:
                pass

    return workers


def close_workers(workers):
    for conn, _, _ in workers:
        try:
            send_msg(conn, {"type": "shutdown"})
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Entrenamiento distribuido
# ─────────────────────────────────────────────────────────────
def run(n_workers: int, rounds: int, local_epochs: int, lr: float, seed: int):
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Server — Entrenamiento Distribuido vía Sockets         ║")
    print(f"║  Escuchando en {HOST}:{PORT:<41}║")
    print("╚══════════════════════════════════════════════════════════╝")

    print("\n  Cargando MNIST...")
    X_train, Y_train, X_test, Y_test = load_mnist()

    print("  Generando shards estratificados...")
    batches = stratified_split_workers(X_train, Y_train, n_workers=n_workers, seed=seed)
    print_distribution(batches, label="Train shards")

    params = init_model(seed=seed)

    server_sock = create_server_socket(HOST, PORT)

    workers = []
    try:
        workers = accept_workers(server_sock, n_workers)

        # Enviar shard inicial a cada worker
        for i, (conn, _, _) in enumerate(workers):
            X_w, Y_w = batches[i]
            payload = {
                "type": "init",
                "worker_id": i,
                "X": X_w,
                "Y": Y_w,
                "params": params,
                "lr": lr,
                "local_epochs": local_epochs,
            }
            print(f"  Enviando shard al worker {i} | muestras={Y_w.shape[0]}")
            send_msg(conn, payload)

        # Rondas globales
        for rnd in range(1, rounds + 1):
            print(f"\n  ── Ronda global {rnd}/{rounds} ──")

            for i, (conn, _, _) in enumerate(workers):
                send_msg(conn, {
                    "type": "train_round",
                    "round": rnd,
                    "params": params,
                })

            local_models = []
            for i, (conn, _, _) in enumerate(workers):
                msg = recv_msg(conn)
                if msg.get("type") != "round_result":
                    raise RuntimeError(f"Respuesta inesperada del worker {i}: {msg.get('type')}")

                local_models.append(msg["params"])
                print(
                    f"  Recibido worker {i} | "
                    f"loss={msg['loss']:.4f} | acc={msg['acc']:.4f}"
                )

            params = average_models(local_models)

            train_acc_small = accuracy(params, X_train[:, :10000], Y_train[:10000])
            test_acc = accuracy(params, X_test, Y_test)

            print(
                f"  Modelo agregado | "
                f"train_acc(10k)={train_acc_small:.4f} | "
                f"test_acc={test_acc:.4f}"
            )

        final_test_acc = accuracy(params, X_test, Y_test)
        print("\n  Entrenamiento finalizado.")
        print(f"  Accuracy final en test: {final_test_acc:.4f}")

    finally:
        close_workers(workers)
        server_sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2, help="Número de workers")
    parser.add_argument("--rounds", type=int, default=5, help="Rondas globales")
    parser.add_argument("--local-epochs", type=int, default=1, help="Épocas locales por worker")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Semilla")
    args = parser.parse_args()

    run(
        n_workers=args.workers,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        seed=args.seed,
    )