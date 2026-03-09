import argparse
import socket
from typing import Dict

import numpy as np

from connection import PORT, connect_with_retry, recv_msg, send_msg


# ─────────────────────────────────────────────────────────────
# Modelo simple
# ─────────────────────────────────────────────────────────────
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=0, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=0, keepdims=True)


def one_hot(Y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    out = np.zeros((n_classes, Y.shape[0]), dtype=np.float32)
    out[Y, np.arange(Y.shape[0])] = 1.0
    return out


def forward(params: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return params["W"] @ X + params["b"]


def predict(params: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return np.argmax(softmax(forward(params, X)), axis=0)


def compute_loss_and_acc(params: Dict[str, np.ndarray], X: np.ndarray, Y: np.ndarray):
    logits = forward(params, X)
    probs = softmax(logits)
    Yh = one_hot(Y)

    eps = 1e-12
    loss = -np.mean(np.sum(Yh * np.log(probs + eps), axis=0))
    acc = np.mean(np.argmax(probs, axis=0) == Y)

    return float(loss), float(acc)


def local_train(
    params: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    lr: float,
    local_epochs: int,
    batch_size: int = 128,
):
    W = params["W"].copy()
    b = params["b"].copy()

    n = Y.shape[0]

    for _ in range(local_epochs):
        idx = np.random.permutation(n)
        Xs = X[:, idx]
        Ys = Y[idx]
        Yh = one_hot(Ys)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            Xb = Xs[:, start:end]
            Yb = Yh[:, start:end]

            logits = W @ Xb + b
            probs = softmax(logits)

            m = Xb.shape[1]
            dZ = (probs - Yb) / m
            dW = dZ @ Xb.T
            db = np.sum(dZ, axis=1, keepdims=True)

            W -= lr * dW.astype(np.float32)
            b -= lr * db.astype(np.float32)

    new_params = {
        "W": W.astype(np.float32),
        "b": b.astype(np.float32),
    }

    loss, acc = compute_loss_and_acc(new_params, X, Y)
    return new_params, loss, acc


# ─────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────
def run(host: str, port: int):
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Worker — Entrenamiento Distribuido vía Sockets         ║")
    print(f"║  Servidor: {host}:{port:<41}║")
    print("╚══════════════════════════════════════════════════════════╝")

    print(f"\n  Conectando a {host}:{port}...")
    sock = connect_with_retry(host, port)

    try:
        send_msg(sock, {
            "type": "hello",
            "name": socket.gethostname(),
        })

        ack = recv_msg(sock)
        if ack.get("type") != "hello_ack":
            raise RuntimeError("Handshake inválido con el servidor")

        print(f"  Conectado. Slot asignado: {ack['worker_slot']}")

        init_msg = recv_msg(sock)
        if init_msg.get("type") != "init":
            raise RuntimeError("No llegó mensaje init")

        worker_id = init_msg["worker_id"]
        X = init_msg["X"]
        Y = init_msg["Y"]
        lr = init_msg["lr"]
        local_epochs = init_msg["local_epochs"]
        current_params = init_msg["params"]

        print(f"  Worker ID: {worker_id}")
        print(f"  Muestras recibidas: {Y.shape[0]}")
        print(f"  lr={lr} | local_epochs={local_epochs}")

        while True:
            msg = recv_msg(sock)
            msg_type = msg.get("type")

            if msg_type == "train_round":
                rnd = msg["round"]
                current_params = msg["params"]

                print(f"\n  Entrenando ronda {rnd}...")
                new_params, loss, acc = local_train(
                    current_params,
                    X,
                    Y,
                    lr=lr,
                    local_epochs=local_epochs,
                )

                send_msg(sock, {
                    "type": "round_result",
                    "worker_id": worker_id,
                    "round": rnd,
                    "params": new_params,
                    "loss": loss,
                    "acc": acc,
                })

                print(f"  Enviado resultado ronda {rnd} | loss={loss:.4f} | acc={acc:.4f}")

            elif msg_type == "shutdown":
                print("\n  El servidor indicó cierre.")
                break

            else:
                raise RuntimeError(f"Mensaje desconocido: {msg_type}")

    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True, help="IP del servidor")
    parser.add_argument("--port", type=int, default=PORT, help="Puerto del servidor")
    args = parser.parse_args()

    run(host=args.host, port=args.port)