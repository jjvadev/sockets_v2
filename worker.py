import argparse
import socket
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from connection import PORT, connect_with_retry, recv_msg, send_msg
from models import CIFAR100CNN


def state_dict_to_numpy(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def numpy_to_state_dict(d: Dict[str, np.ndarray], device: torch.device):
    return {k: torch.tensor(v, device=device) for k, v in d.items()}


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

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


def train_local(model, loader, device, config):
    criterion = nn.CrossEntropyLoss()

    optimizer_name = config.get("optimizer", "adam").lower()
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Optimizer no soportado: {optimizer_name}")

    model.train()
    for _ in range(config["local_epochs"]):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return evaluate(model, loader, device)


def build_model(config):
    return CIFAR100CNN(
        num_classes=config["num_classes"],
        activation=config["activation"],
        negative_slope=config["negative_slope"],
        dropout=config["dropout"],
        channels=tuple(config["channels"]),
        fc_dim=config["fc_dim"],
    )


def run(host: str, port: int):
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Worker — CIFAR-100 distribuido vía sockets             ║")
    print(f"║  Servidor: {host}:{port:<41}║")
    print("╚══════════════════════════════════════════════════════════╝")

    print(f"\n  Conectando a {host}:{port}...")
    sock = connect_with_retry(host, port)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        send_msg(sock, {
            "type": "ready",
            "name": socket.gethostname(),
        })

        init_msg = recv_msg(sock)
        if init_msg.get("type") != "data":
            raise RuntimeError(f"Esperaba 'data', recibí '{init_msg.get('type')}'")

        worker_id = init_msg["worker_id"]
        X = torch.tensor(init_msg["X"], dtype=torch.float32)
        Y = torch.tensor(init_msg["Y"], dtype=torch.long)
        config = init_msg["config"]

        ds = TensorDataset(X, Y)
        loader = DataLoader(
            ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
        )

        model = build_model(config).to(device)

        print(f"  Worker ID: {worker_id}")
        print(f"  Samples: {len(ds)}")
        print(f"  Device: {device}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Local epochs: {config['local_epochs']}")
        print(f"  Optimizer: {config.get('optimizer', 'adam')}")

        while True:
            msg = recv_msg(sock)
            msg_type = msg.get("type")

            if msg_type == "model":
                round_idx = msg["round"]
                model.load_state_dict(numpy_to_state_dict(msg["state_dict"], device))

                print(f"\n  Entrenando round {round_idx}...")

                loss, acc = train_local(
                    model=model,
                    loader=loader,
                    device=device,
                    config=config,
                )

                send_msg(sock, {
                    "type": "model_update",
                    "worker_id": worker_id,
                    "round": round_idx,
                    "state_dict": state_dict_to_numpy(model.state_dict()),
                    "loss": float(loss),
                    "acc": float(acc),
                    "n_samples": len(ds),
                })

                print(f"  Enviado round {round_idx} | loss={loss:.4f} | acc_local={acc:.4f}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker distribuido CIFAR-100")
    parser.add_argument("--host", type=str, required=True, help="IP del servidor")
    parser.add_argument("--port", type=int, default=PORT, help="Puerto TCP")
    args = parser.parse_args()

    run(host=args.host, port=args.port)