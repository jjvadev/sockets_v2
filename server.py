import argparse
import csv
import json
import os
import random
import socket
import threading
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from connection import HOST, PORT, BACKLOG, send_msg, recv_msg
from stratified_split import stratified_sample, stratified_split_workers, print_distribution
from models import CIFAR100CNN
from analysis_notebook import generate_analysis_notebook



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_to_numpy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def numpy_to_torch_state_dict(state_dict: Dict[str, np.ndarray], device: torch.device):
    return {k: torch.tensor(v, device=device) for k, v in state_dict.items()}


def build_transforms(use_augmentation: bool):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    if use_augmentation:
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        tf_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return tf_train, tf_eval


def load_cifar100(
    data_dir: str,
    n_train: int,
    seed: int,
    use_augmentation: bool,
):
    tf_train, tf_eval = build_transforms(use_augmentation=use_augmentation)

    tr_ds = datasets.CIFAR100(data_dir, train=True, download=True, transform=tf_train)
    te_ds = datasets.CIFAR100(data_dir, train=False, download=True, transform=tf_eval)

    X_train_t, Y_train_t = next(iter(DataLoader(tr_ds, batch_size=50000, shuffle=False)))
    X_test_t, Y_test_t = next(iter(DataLoader(te_ds, batch_size=10000, shuffle=False)))

    X_train = X_train_t.numpy().astype(np.float32)   # (N,3,32,32)
    Y_train = Y_train_t.numpy().astype(np.int64)

    X_test = X_test_t.numpy().astype(np.float32)
    Y_test = Y_test_t.numpy().astype(np.int64)

    if n_train < len(Y_train):
        X_train, Y_train = stratified_sample(X_train, Y_train, n_samples=n_train, seed=seed)

    class_names = te_ds.classes

    print(f"  [Datos] Train: {X_train.shape} | Test: {X_test.shape}")
    print("  [Formato] RGB: (N, 3, 32, 32)")
    print("  [Clases] CIFAR-100 -> 100 clases")

    return X_train, Y_train, X_test, Y_test, class_names


def build_model(config: Dict) -> nn.Module:
    return CIFAR100CNN(
        num_classes=config["num_classes"],
        activation=config["activation"],
        negative_slope=config["negative_slope"],
        dropout=config["dropout"],
        channels=tuple(config["channels"]),
        fc_dim=config["fc_dim"],
    )


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    device: torch.device,
    batch_size: int,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

    return total_loss / total, total_correct / total


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


def average_state_dicts(worker_updates: List[Dict]) -> Dict[str, np.ndarray]:
    total_samples = sum(m["n_samples"] for m in worker_updates)
    keys = worker_updates[0]["state_dict"].keys()
    avg = {}

    for k in keys:
        weighted = sum(
            m["state_dict"][k] * (m["n_samples"] / total_samples)
            for m in worker_updates
        )
        avg[k] = weighted.astype(np.float32)

    return avg


def save_history_csv(history: List[Dict], csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "worker_loss_mean",
                "worker_acc_mean",
                "round_time",
                "total_time",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_summary_json(summary: Dict, json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_model_pt(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def save_plots(history: List[Dict], out_dir: str):
    rounds = [h["round"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [h["train_loss"] for h in history], marker="o", label="Train Loss")
    plt.plot(rounds, [h["test_loss"] for h in history], marker="s", label="Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss por round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [h["train_acc"] for h in history], marker="o", label="Train Acc")
    plt.plot(rounds, [h["test_acc"] for h in history], marker="s", label="Test Acc")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy por round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [h["round_time"] for h in history], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Segundos")
    plt.title("Tiempo por round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_per_round.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [h["total_time"] for h in history], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Segundos")
    plt.title("Tiempo acumulado")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_cumulative.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    gap = [h["train_acc"] - h["test_acc"] for h in history]
    plt.plot(rounds, gap, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Gap")
    plt.title("Gap Train-Test")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "generalization_gap.png"))
    plt.close()


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    run_name = args.run_name or (
        f"cifar100_w{args.workers}_r{args.rounds}_le{args.local_epochs}"
        f"_bs{args.batch_size}_lr{args.lr}_{args.activation}"
    )
    results_dir = os.path.join(args.results_dir, run_name)
    ensure_dir(results_dir)

    config = {
        "dataset": "CIFAR-100",
        "num_classes": 100,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "local_epochs": args.local_epochs,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "activation": args.activation,
        "negative_slope": args.negative_slope,
        "dropout": args.dropout,
        "channels": args.channels,
        "fc_dim": args.fc_dim,
    }

    print("╔═════════════════════════════════════════════════════════════════════╗")
    print("║  Servidor — Entrenamiento Distribuido CIFAR-100 vía Sockets       ║")
    print(f"║  Workers: {args.workers:<4}  Rounds: {args.rounds:<5}  LR: {args.lr:<10}║")
    print("╚═════════════════════════════════════════════════════════════════════╝")

    X_train, Y_train, X_test, Y_test, class_names = load_cifar100(
        data_dir=args.data_dir,
        n_train=args.n_train,
        seed=args.seed,
        use_augmentation=args.augmentation,
    )

    batches = stratified_split_workers(X_train, Y_train, n_workers=args.workers, seed=args.seed)
    print_distribution(batches, label=f"n_workers={args.workers}")

    model = build_model(config).to(device)
    best_test_acc = -1.0
    best_round = -1

    worker_sockets, _ = accept_workers(args.workers, port=args.port)

    history = []
    t_start = time.time()

    try:
        print("  Enviando shards y configuración a workers...")
        for wid, (sock, (X_w, Y_w)) in enumerate(zip(worker_sockets, batches)):
            msg = recv_msg(sock)
            if msg.get("type") != "ready":
                raise RuntimeError(f"Esperaba 'ready' del worker {wid}, recibí {msg.get('type')}")

            send_msg(sock, {
                "type": "data",
                "worker_id": wid,
                "X": X_w,
                "Y": Y_w,
                "config": config,
            })
            print(f"  → Worker {wid}: {len(Y_w)} ejemplos enviados")

        print(f"\n{'=' * 96}")
        print(
            f"  {'Round':>5}  {'TrLoss':>8}  {'TrAcc':>8}  {'TeLoss':>8}  {'TeAcc':>8}  "
            f"{'WLoss':>8}  {'WAcc':>8}  {'t_rnd':>8}  {'t_tot':>8}"
        )
        print(f"  {'-' * 92}")

        for round_idx in range(args.rounds):
            t_round = time.time()

            send_to_all(worker_sockets, {
                "type": "model",
                "round": round_idx,
                "state_dict": tensor_to_numpy_state_dict(model.state_dict()),
            })

            updates = recv_from_all(worker_sockets)

            for i, msg in enumerate(updates):
                if msg.get("type") != "model_update":
                    raise RuntimeError(f"Respuesta inesperada del worker {i}: {msg.get('type')}")

            avg_state_dict = average_state_dicts(updates)
            model.load_state_dict(numpy_to_torch_state_dict(avg_state_dict, device))

            train_loss, train_acc = evaluate_model(
                model=model,
                X=X_train,
                Y=Y_train,
                device=device,
                batch_size=args.eval_batch_size,
            )
            test_loss, test_acc = evaluate_model(
                model=model,
                X=X_test,
                Y=Y_test,
                device=device,
                batch_size=args.eval_batch_size,
            )

            worker_loss_mean = float(np.mean([m["loss"] for m in updates]))
            worker_acc_mean = float(np.mean([m["acc"] for m in updates]))

            round_time = time.time() - t_round
            total_time = time.time() - t_start

            row = {
                "round": int(round_idx),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "worker_loss_mean": worker_loss_mean,
                "worker_acc_mean": worker_acc_mean,
                "round_time": float(round_time),
                "total_time": float(total_time),
            }
            history.append(row)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_round = round_idx
                save_model_pt(model, os.path.join(results_dir, "best_model.pt"))

            if round_idx % args.print_every == 0 or round_idx == args.rounds - 1:
                print(
                    f"  {round_idx:5d}  {train_loss:8.4f}  {train_acc:8.4f}  "
                    f"{test_loss:8.4f}  {test_acc:8.4f}  {worker_loss_mean:8.4f}  "
                    f"{worker_acc_mean:8.4f}  {round_time:7.3f}s  {total_time:7.1f}s"
                )

        send_to_all(worker_sockets, {"type": "stop"})

        final_train_loss, final_train_acc = evaluate_model(
            model, X_train, Y_train, device, args.eval_batch_size
        )
        final_test_loss, final_test_acc = evaluate_model(
            model, X_test, Y_test, device, args.eval_batch_size
        )
        total_time = time.time() - t_start

        summary = {
            "run_name": run_name,
            "dataset": "CIFAR-100",
            "device": str(device),
            "workers": args.workers,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "activation": args.activation,
            "negative_slope": args.negative_slope,
            "dropout": args.dropout,
            "channels": args.channels,
            "fc_dim": args.fc_dim,
            "seed": args.seed,
            "n_train": args.n_train,
            "augmentation": bool(args.augmentation),
            "print_every": args.print_every,
            "final_train_loss": float(final_train_loss),
            "final_train_acc": float(final_train_acc),
            "final_test_loss": float(final_test_loss),
            "final_test_acc": float(final_test_acc),
            "best_test_acc": float(best_test_acc),
            "best_round": int(best_round),
            "total_time_sec": float(total_time),
            "history_points": len(history),
        }

        history_csv_path = os.path.join(results_dir, "history.csv")
        summary_json_path = os.path.join(results_dir, "summary.json")
        final_model_path = os.path.join(results_dir, "final_model.pt")

        save_history_csv(history, history_csv_path)
        save_summary_json(summary, summary_json_path)
        save_model_pt(model, final_model_path)
        save_plots(history, results_dir)

        generate_analysis_notebook(
            results_dir=results_dir,
            summary=summary,
            history_csv_path=history_csv_path,
            summary_json_path=summary_json_path,
        )

        print("\n  ✓ Entrenamiento finalizado")
        print(f"  ✓ Train final : {final_train_acc * 100:.2f}%")
        print(f"  ✓ Test final  : {final_test_acc * 100:.2f}%")
        print(f"  ✓ Best test   : {best_test_acc * 100:.2f}% (round {best_round})")
        print(f"  ⏱ Tiempo total: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  📁 Resultados : {results_dir}")

    finally:
        for sock in worker_sockets:
            try:
                sock.close()
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Servidor distribuido CIFAR-100 con CNN + FedAvg vía sockets"
    )

    parser.add_argument("--workers",         type=int,   default=2)
    parser.add_argument("--rounds",          type=int,   default=50)
    parser.add_argument("--local-epochs",    type=int,   default=1)        # 1 época local — comportamiento estándar FedAvg
    parser.add_argument("--batch-size",      type=int,   default=256)      # batch grande → rounds más rápidos
    parser.add_argument("--eval-batch-size", type=int,   default=512)
    parser.add_argument("--lr",              type=float, default=1e-2)     # SGD necesita LR más alto que Adam
    parser.add_argument("--weight-decay",    type=float, default=1e-4)
    parser.add_argument("--optimizer",       type=str,   default="sgd",        choices=["adam", "sgd"])
    parser.add_argument("--momentum",        type=float, default=0.9)
    parser.add_argument("--activation",      type=str,   default="leaky_relu", choices=["relu", "leaky_relu"])
    parser.add_argument("--negative-slope",  type=float, default=0.01)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--channels",        type=int,   nargs=3, default=[64, 128, 256])
    parser.add_argument("--fc-dim",          type=int,   default=256)
    parser.add_argument("--n-train",         type=int,   default=50000)   # 50k imágenes completas
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--print-every",     type=int,   default=1)
    parser.add_argument("--port",            type=int,   default=PORT)
    parser.add_argument("--data-dir",        type=str,   default="./data")
    parser.add_argument("--results-dir",     type=str,   default="./results")
    parser.add_argument("--run-name",        type=str,   default="")
    # augmentation desactivado por defecto — no pasar el flag
    parser.add_argument("--augmentation",   action="store_true")
    parser.add_argument("--cpu",            action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)