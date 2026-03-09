import argparse
import socket
from typing import Dict, List, Tuple

import numpy as np

from connection import PORT, connect_with_retry, recv_msg, send_msg


# ═══════════════════════════════════════════════════════════
# ACTIVACIONES
# ═══════════════════════════════════════════════════════════
def leaky_relu(Z: np.ndarray, a: float = 0.01) -> np.ndarray:
    return np.where(Z > 0, Z, a * Z)


def d_leaky_relu(Z: np.ndarray, a: float = 0.01) -> np.ndarray:
    return np.where(Z > 0, 1.0, a).astype(np.float32)


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - np.max(Z, axis=0, keepdims=True)
    e = np.exp(Z)
    return e / np.sum(e, axis=0, keepdims=True)


def one_hot(Y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    out = np.zeros((n_classes, Y.shape[0]), dtype=np.float32)
    out[Y, np.arange(Y.shape[0])] = 1.0
    return out


# ═══════════════════════════════════════════════════════════
# FORWARD / BACKWARD
# ═══════════════════════════════════════════════════════════
def forward_pass(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
    layer_dims: List[int],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Retorna:
      A_last: salida softmax
      cache: valores intermedios para backward
    """
    cache = {}
    A = X
    L = len(layer_dims) - 1

    cache["A0"] = X

    for l in range(1, L + 1):
        W = params[f"W{l}"]
        b = params[f"b{l}"]

        Z = W @ A + b
        cache[f"Z{l}"] = Z

        if l == L:
            A = softmax(Z)
        else:
            A = leaky_relu(Z)

        cache[f"A{l}"] = A

    return A, cache


def compute_cost(A_last: np.ndarray, Y: np.ndarray) -> float:
    Yh = one_hot(Y, n_classes=A_last.shape[0])
    eps = 1e-12
    cost = -np.mean(np.sum(Yh * np.log(A_last + eps), axis=0))
    return float(cost)


def backward_pass(
    Y: np.ndarray,
    params: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    layer_dims: List[int],
) -> Dict[str, np.ndarray]:
    grads = {}
    L = len(layer_dims) - 1
    m = Y.shape[0]

    Yh = one_hot(Y, n_classes=layer_dims[-1])

    # Capa de salida: softmax + cross entropy
    A_last = cache[f"A{L}"]
    dZ = (A_last - Yh) / m

    for l in range(L, 0, -1):
        A_prev = cache[f"A{l-1}"]
        W = params[f"W{l}"]

        dW = dZ @ A_prev.T
        db = np.sum(dZ, axis=1, keepdims=True)

        grads[f"dW{l}"] = dW.astype(np.float32)
        grads[f"db{l}"] = db.astype(np.float32)

        if l > 1:
            Z_prev = cache[f"Z{l-1}"]
            dA_prev = W.T @ dZ
            dZ = dA_prev * d_leaky_relu(Z_prev)

    return grads


def compute_grads_and_cost(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    layer_dims: List[int],
) -> Tuple[Dict[str, np.ndarray], float]:
    A_last, cache = forward_pass(X, params, layer_dims)
    cost = compute_cost(A_last, Y)
    grads = backward_pass(Y, params, cache, layer_dims)
    return grads, cost


# ═══════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════
def predict(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    layer_dims: List[int],
) -> np.ndarray:
    probs, _ = forward_pass(X, params, layer_dims)
    return np.argmax(probs, axis=0)


def accuracy(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    layer_dims: List[int],
) -> float:
    pred = predict(params, X, layer_dims)
    return float(np.mean(pred == Y))


# ═══════════════════════════════════════════════════════════
# WORKER
# ═══════════════════════════════════════════════════════════
def run(host: str, port: int):
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Worker — Entrenamiento Distribuido vía Sockets         ║")
    print(f"║  Servidor: {host}:{port:<41}║")
    print("╚══════════════════════════════════════════════════════════╝")

    print(f"\n  Conectando a {host}:{port}...")
    sock = connect_with_retry(host, port)

    try:
        # 1. Avisar que está listo
        send_msg(sock, {
            "type": "ready",
            "name": socket.gethostname(),
        })

        # 2. Recibir shard de datos
        init_msg = recv_msg(sock)
        if init_msg.get("type") != "data":
            raise RuntimeError(f"Esperaba 'data', recibí '{init_msg.get('type')}'")

        worker_id = init_msg["worker_id"]
        X = init_msg["X"].astype(np.float32)
        Y = init_msg["Y"].astype(np.int64)
        layer_dims = init_msg["layer_dims"]
        alpha = init_msg.get("alpha", 0.1)

        print(f"  Worker ID: {worker_id}")
        print(f"  Muestras recibidas: {Y.shape[0]}")
        print(f"  Arquitectura: {layer_dims}")
        print(f"  LR del servidor: {alpha}")

        while True:
            msg = recv_msg(sock)
            msg_type = msg.get("type")

            if msg_type == "params":
                epoch = msg["epoch"]
                params = msg["params"]

                print(f"\n  Calculando gradientes de época {epoch}...")

                grads, cost = compute_grads_and_cost(
                    params=params,
                    X=X,
                    Y=Y,
                    layer_dims=layer_dims,
                )

                acc = accuracy(
                    params=params,
                    X=X,
                    Y=Y,
                    layer_dims=layer_dims,
                )

                send_msg(sock, {
                    "type": "grads",
                    "worker_id": worker_id,
                    "epoch": epoch,
                    "grads": grads,
                    "cost": float(cost),
                    "acc": float(acc),
                    "n_samples": int(Y.shape[0]),
                })

                print(
                    f"  Enviado epoch {epoch} | "
                    f"cost={cost:.4f} | acc_local={acc:.4f}"
                )

            elif msg_type == "stop":
                print("\n  El servidor indicó cierre.")
                break

            else:
                raise RuntimeError(f"Mensaje desconocido: {msg_type}")

    finally:
        try:
            sock.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True, help="IP del servidor")
    parser.add_argument("--port", type=int, default=PORT, help="Puerto del servidor")
    args = parser.parse_args()

    run(host=args.host, port=args.port)