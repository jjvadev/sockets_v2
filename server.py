import argparse
import csv
import json
import os
import socket
import threading
import time
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from connection import HOST, PORT, BACKLOG, send_msg, recv_msg
from stratified_split import stratified_sample, stratified_split_workers, print_distribution


# ═══════════════════════════════════════════════════════════
# HIPERPARÁMETROS FIJOS
# ═══════════════════════════════════════════════════════════
ALPHA = 0.1
LAYER_DIMS = [784, 256, 128, 10]
SEED = 42
N_TRAIN = 50_000
PRINT_EVERY = 10


# ═══════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════
def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tr_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)

    Xa, Ya = next(iter(DataLoader(tr_ds, batch_size=60000, shuffle=False)))
    Xe, Ye = next(iter(DataLoader(te_ds, batch_size=10000, shuffle=False)))

    Xa = Xa.numpy().reshape(Xa.shape[0], -1).T.astype(np.float32)
    Ya = Ya.numpy().astype(np.int64)

    Xe = Xe.numpy().reshape(Xe.shape[0], -1).T.astype(np.float32)
    Ye = Ye.numpy().astype(np.int64)

    X_train, Y_train = stratified_sample(Xa, Ya, n_samples=N_TRAIN, seed=SEED)
    print(f"  [Datos] Train: {X_train.shape} | Test: {Xe.shape}")
    return X_train, Y_train, Xe, Ye


# ═══════════════════════════════════════════════════════════
# 2. INICIALIZACIÓN DE PARÁMETROS
# ═══════════════════════════════════════════════════════════
def init_params(layer_dims: List[int] = LAYER_DIMS, seed: int = SEED) -> Dict[str, np.ndarray]:
    np.random.seed(seed)
    params = {}
    L = len(layer_dims) - 1

    for l in range(1, L + 1):
        params[f"W{l}"] = (
            np.random.randn(layer_dims[l], layer_dims[l - 1]).astype(np.float32)
            * np.sqrt(2.0 / layer_dims[l - 1])
        ).astype(np.float32)
        params[f"b{l}"] = np.zeros((layer_dims[l], 1), dtype=np.float32)

    return params


# ═══════════════════════════════════════════════════════════
# 3. FORWARD DE EVALUACIÓN
# ═══════════════════════════════════════════════════════════
def leaky_relu(Z: np.ndarray, a: float = 0.01) -> np.ndarray:
    return np.where(Z > 0, Z, a * Z)


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - np.max(Z, axis=0, keepdims=True)
    e = np.exp(Z)
    return e / np.sum(e, axis=0, keepdims=True)


def forward_eval(X: np.ndarray, params: Dict[str, np.ndarray], L: int) -> np.ndarray:
    A = X
    for l in range(1, L + 1):
        Z = params[f"W{l}"] @ A + params[f"b{l}"]
        A = softmax(Z) if l == L else leaky_relu(Z)
    return A


def accuracy(X: np.ndarray, Y: np.ndarray, params: Dict[str, np.ndarray], L: int) -> float:
    A = forward_eval(X, params, L)
    pred = np.argmax(A, axis=0)
    return float(np.mean(pred == Y))


# ═══════════════════════════════════════════════════════════
# 4. ACTUALIZACIÓN DE PARÁMETROS
# ═══════════════════════════════════════════════════════════
def update_params(
    params: Dict[str, np.ndarray],
    avg_grads: Dict[str, np.ndarray],
    L: int,
    alpha: float = ALPHA,
) -> Dict[str, np.ndarray]:
    for l in range(1, L + 1):
        params[f"W{l}"] -= alpha * avg_grads[f"dW{l}"]
        params[f"b{l}"] -= alpha * avg_grads[f"db{l}"]
    return params


# ═══════════════════════════════════════════════════════════
# 5. COMUNICACIÓN CON WORKERS
# ═══════════════════════════════════════════════════════════
def accept_workers(n_workers: int, port: int = PORT):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(BACKLOG)

    print(f"\n  Servidor escuchando en {HOST}:{port}")
    print(f"  Esperando {n_workers} worker(s)...\n")

    worker_sockets = []
    worker_addrs = []

    while len(worker_sockets) < n_workers:
        conn, addr = srv.accept()
        wid = len(worker_sockets)
        print(f"  ✓ Worker {wid} conectado desde {addr[0]}:{addr[1]}")
        worker_sockets.append(conn)
        worker_addrs.append(addr)

    srv.close()
    print("\n  Todos los workers conectados. Iniciando entrenamiento.\n")
    return worker_sockets, worker_addrs


def send_to_all(sockets, msg):
    for sock in sockets:
        send_msg(sock, msg)


def recv_from_worker(sock, results, idx):
    results[idx] = recv_msg(sock)


def recv_from_all(sockets):
    results = [None] * len(sockets)
    threads = []

    for i, sock in enumerate(sockets):
        t = threading.Thread(target=recv_from_worker, args=(sock, results, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return results


# ═══════════════════════════════════════════════════════════
# 6. PROMEDIO DE GRADIENTES
# ═══════════════════════════════════════════════════════════
def average_gradients(grad_messages, L: int) -> Dict[str, np.ndarray]:
    all_grads = [m["grads"] for m in grad_messages]
    avg = {}

    for l in range(1, L + 1):
        avg[f"dW{l}"] = np.mean([g[f"dW{l}"] for g in all_grads], axis=0).astype(np.float32)
        avg[f"db{l}"] = np.mean([g[f"db{l}"] for g in all_grads], axis=0).astype(np.float32)

    return avg


# ═══════════════════════════════════════════════════════════
# 7. GENERACIÓN AUTOMÁTICA DEL NOTEBOOK
# ═══════════════════════════════════════════════════════════
def generate_notebook(n_workers: int, epochs: int):
    os.makedirs("results", exist_ok=True)

    csv_path = f"results/history_w{n_workers}_e{epochs}.csv"
    json_path = f"results/summary_w{n_workers}_e{epochs}.json"
    nb_path = f"results/analysis_w{n_workers}_e{epochs}.ipynb"

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Análisis de entrenamiento distribuido vía sockets\n",
                    "\n",
                    f"**Workers:** {n_workers}  \n",
                    f"**Épocas:** {epochs}  \n",
                    "\n",
                    "Este cuaderno fue generado automáticamente al finalizar el entrenamiento.\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Arquitectura del sistema\n",
                    "\n",
                    "- Servidor central que coordina el entrenamiento.\n",
                    "- Workers distribuidos conectados por sockets.\n",
                    "- El servidor envía parámetros globales.\n",
                    "- Cada worker calcula gradientes sobre su shard local.\n",
                    "- El servidor promedia gradientes y actualiza el modelo.\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Arquitectura del modelo\n",
                    "\n",
                    "Red neuronal multicapa:\n",
                    "\n",
                    "- Entrada: 784\n",
                    "- Capa oculta 1: 256 + LeakyReLU\n",
                    "- Capa oculta 2: 128 + LeakyReLU\n",
                    "- Salida: 10 + Softmax\n",
                    "\n",
                    f"Arquitectura usada: `{LAYER_DIMS}`\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Carga de resultados\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from pathlib import Path\n",
                    "\n",
                    f'csv_file = Path("{csv_path}")\n',
                    f'json_file = Path("{json_path}")\n',
                    "\n",
                    "df = pd.read_csv(csv_file)\n",
                    "with open(json_file, 'r', encoding='utf-8') as f:\n",
                    "    summary = json.load(f)\n",
                    "\n",
                    "df.head()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Resumen del experimento\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "summary\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "resumen_df = pd.DataFrame([{\n",
                    "    'Workers': summary['workers'],\n",
                    "    'Épocas': summary['epochs'],\n",
                    "    'Learning rate': summary['learning_rate'],\n",
                    "    'Arquitectura': str(summary['layer_dims']),\n",
                    "    'Train final': summary['final_train_acc'],\n",
                    "    'Test final': summary['final_test_acc'],\n",
                    "    'Tiempo total (s)': summary['total_time_sec'],\n",
                    "}])\n",
                    "resumen_df\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Historial completo\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Gráfica: costo por época\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(8,5))\n",
                    "plt.plot(df['epoch'], df['cost'], marker='o')\n",
                    "plt.title('Costo promedio por época')\n",
                    "plt.xlabel('Época')\n",
                    "plt.ylabel('Costo')\n",
                    "plt.grid(True)\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Gráfica: accuracy train vs test\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(8,5))\n",
                    "plt.plot(df['epoch'], df['train'], marker='o', label='Train Accuracy')\n",
                    "plt.plot(df['epoch'], df['test'], marker='s', label='Test Accuracy')\n",
                    "plt.title('Accuracy de entrenamiento y prueba')\n",
                    "plt.xlabel('Época')\n",
                    "plt.ylabel('Accuracy')\n",
                    "plt.legend()\n",
                    "plt.grid(True)\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Gráfica: tiempo por época\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(8,5))\n",
                    "plt.plot(df['epoch'], df['epoch_time'], marker='o')\n",
                    "plt.title('Tiempo por época')\n",
                    "plt.xlabel('Época')\n",
                    "plt.ylabel('Segundos')\n",
                    "plt.grid(True)\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Gráfica: tiempo acumulado\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(8,5))\n",
                    "plt.plot(df['epoch'], df['total_time'], marker='o')\n",
                    "plt.title('Tiempo acumulado')\n",
                    "plt.xlabel('Época')\n",
                    "plt.ylabel('Segundos')\n",
                    "plt.grid(True)\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Métricas derivadas\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "gap = summary['final_train_acc'] - summary['final_test_acc']\n",
                    "avg_epoch_time = df['epoch_time'].mean()\n",
                    "best_test = df['test'].max()\n",
                    "\n",
                    "metricas = pd.DataFrame([{\n",
                    "    'Gap train-test': gap,\n",
                    "    'Tiempo promedio por registro (s)': avg_epoch_time,\n",
                    "    'Mejor test accuracy registrado': best_test,\n",
                    "}])\n",
                    "metricas\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Conclusión automática\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Accuracy final en train: {summary['final_train_acc']:.4f}\")\n",
                    "print(f\"Accuracy final en test : {summary['final_test_acc']:.4f}\")\n",
                    "print(f\"Tiempo total           : {summary['total_time_sec']:.2f} s\")\n",
                    "print(f\"Gap train-test         : {summary['final_train_acc'] - summary['final_test_acc']:.4f}\")\n",
                    "\n",
                    "if (summary['final_train_acc'] - summary['final_test_acc']) < 0.02:\n",
                    "    print('Conclusión: el modelo generaliza bien y el entrenamiento fue estable.')\n",
                    "else:\n",
                    "    print('Conclusión: revisar posible sobreajuste.')\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.x"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"  📓 Notebook generado: {nb_path}")


# ═══════════════════════════════════════════════════════════
# 8. GUARDADO DE RESULTADOS
# ═══════════════════════════════════════════════════════════
def save_results(
    history: List[Dict],
    params: Dict[str, np.ndarray],
    n_workers: int,
    epochs: int,
    final_train_acc: float,
    final_test_acc: float,
    total_time: float,
):
    os.makedirs("results", exist_ok=True)

    csv_path = f"results/history_w{n_workers}_e{epochs}.csv"
    json_path = f"results/summary_w{n_workers}_e{epochs}.json"
    npz_path = f"results/params_socket_w{n_workers}_e{epochs}.npz"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "cost",
                "train",
                "test",
                "epoch_time",
                "total_time",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "workers": n_workers,
        "epochs": epochs,
        "learning_rate": ALPHA,
        "layer_dims": LAYER_DIMS,
        "seed": SEED,
        "n_train": N_TRAIN,
        "print_every": PRINT_EVERY,
        "final_train_acc": float(final_train_acc),
        "final_test_acc": float(final_test_acc),
        "total_time_sec": float(total_time),
        "history_points": len(history),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    np.savez(npz_path, **params)

    print(f"  💾 {csv_path}")
    print(f"  💾 {json_path}")
    print(f"  💾 {npz_path}")

    generate_notebook(n_workers, epochs)


# ═══════════════════════════════════════════════════════════
# 9. BUCLE DE ENTRENAMIENTO PRINCIPAL
# ═══════════════════════════════════════════════════════════
def train(n_workers: int, epochs: int, port: int = PORT):
    L = len(LAYER_DIMS) - 1

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Servidor — Entrenamiento Distribuido vía Sockets       ║")
    print(f"║  Workers: {n_workers:<4}  Épocas: {epochs:<5}  LR: {ALPHA:<8}      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    X_train, Y_train, X_test, Y_test = load_mnist()

    batches = stratified_split_workers(X_train, Y_train, n_workers=n_workers, seed=SEED)
    print_distribution(batches, label=f"n_workers={n_workers}")

    params = init_params()

    worker_sockets, _ = accept_workers(n_workers, port=port)

    try:
        print("  Enviando datos a workers...")
        for wid, (sock, (X_w, Y_w)) in enumerate(zip(worker_sockets, batches)):
            msg = recv_msg(sock)
            if msg.get("type") != "ready":
                raise RuntimeError(f"Esperaba 'ready' del worker {wid}, recibí {msg.get('type')}")

            send_msg(sock, {
                "type": "data",
                "worker_id": wid,
                "X": X_w,
                "Y": Y_w,
                "layer_dims": LAYER_DIMS,
                "alpha": ALPHA,
            })
            print(f"  → Worker {wid}: {X_w.shape[1]} ejemplos enviados")

        print(f"\n{'=' * 72}")
        print(f"  {'Época':>5}  {'Costo':>9}  {'Train':>8}  {'Test':>8}  {'t_ep':>8}  {'t_tot':>8}")
        print(f"  {'-' * 65}")

        t_start = time.time()
        history = []

        for epoch in range(epochs):
            t_ep = time.time()

            send_to_all(worker_sockets, {
                "type": "params",
                "params": params,
                "epoch": epoch,
            })

            grad_msgs = recv_from_all(worker_sockets)

            for i, msg in enumerate(grad_msgs):
                if msg.get("type") != "grads":
                    raise RuntimeError(f"Respuesta inesperada del worker {i}: {msg.get('type')}")

            avg_grads = average_gradients(grad_msgs, L)
            avg_cost = float(np.mean([m["cost"] for m in grad_msgs]))

            params = update_params(params, avg_grads, L)

            epoch_time = time.time() - t_ep
            total_time = time.time() - t_start

            if epoch % PRINT_EVERY == 0 or epoch == epochs - 1:
                tr = accuracy(X_train, Y_train, params, L)
                te = accuracy(X_test, Y_test, params, L)

                row = {
                    "epoch": int(epoch),
                    "cost": float(avg_cost),
                    "train": float(tr),
                    "test": float(te),
                    "epoch_time": float(epoch_time),
                    "total_time": float(total_time),
                }
                history.append(row)

                print(
                    f"  {epoch:5d}  {avg_cost:9.4f}  {tr:8.4f}  {te:8.4f}  "
                    f"{epoch_time:7.3f}s  {total_time:7.1f}s"
                )

        send_to_all(worker_sockets, {"type": "stop"})
        for sock in worker_sockets:
            sock.close()

        final_train = accuracy(X_train, Y_train, params, L)
        final_test = accuracy(X_test, Y_test, params, L)
        total = time.time() - t_start

        print(f"\n  ✓ Train final : {final_train * 100:.2f}%")
        print(f"  ✓ Test final  : {final_test * 100:.2f}%")
        print(f"  ⏱  Tiempo     : {total:.1f}s ({total / 60:.1f} min)")

        save_results(
            history=history,
            params=params,
            n_workers=n_workers,
            epochs=epochs,
            final_train_acc=final_train,
            final_test_acc=final_test,
            total_time=total,
        )

        return params, history

    finally:
        for sock in worker_sockets:
            try:
                sock.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Servidor de entrenamiento distribuido MNIST"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Número de workers a esperar (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Épocas de entrenamiento (default: 200)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"Puerto TCP (default: {PORT})",
    )
    args = parser.parse_args()

    train(
        n_workers=args.workers,
        epochs=args.epochs,
        port=args.port,
    )