"""
connection.py — Protocolo de comunicación compartido
════════════════════════════════════════════════════════════════════
Usado por server.py y worker.py.

Implementa mensajes con framing de longitud fija:
  ┌──────────────────┬──────────────────────────────┐
  │  8 bytes (uint64)│  N bytes (payload pickle)    │
  │  longitud N      │  objeto Python serializado   │
  └──────────────────┴──────────────────────────────┘
"""

import pickle
import socket
import struct
from typing import Any


# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN DE RED
# ═══════════════════════════════════════════════════════════

# Para que el server acepte conexiones desde otras PCs:
# usar 0.0.0.0 al hacer bind en server.py
HOST = "0.0.0.0"

# Puerto TCP compartido
PORT = 65432

# Conexiones pendientes máximas en listen()
BACKLOG = 32

# Timeouts recomendados
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 900
RETRY_DELAY = 3
MAX_RETRIES = 20

# Formato del header: uint64 big-endian (8 bytes)
HEADER_FMT = ">Q"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


# ═══════════════════════════════════════════════════════════
# HELPERS DE SOCKET
# ═══════════════════════════════════════════════════════════
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Recibe exactamente n bytes del socket.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Socket cerrado inesperadamente "
                f"(recibidos {len(buf)}/{n} bytes)"
            )
        buf.extend(chunk)
    return bytes(buf)


def create_server_socket(host: str = HOST, port: int = PORT,
                         backlog: int = BACKLOG) -> socket.socket:
    """
    Crea un socket TCP listo para escuchar.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    return sock


def connect_with_retry(host: str, port: int = PORT,
                       retries: int = MAX_RETRIES,
                       retry_delay: int = RETRY_DELAY,
                       connect_timeout: int = CONNECT_TIMEOUT,
                       read_timeout: int = READ_TIMEOUT) -> socket.socket:
    """
    Intenta conectarse varias veces al servidor antes de fallar.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(connect_timeout)
            sock.connect((host, port))
            sock.settimeout(read_timeout)
            return sock
        except Exception as e:
            last_error = e
            print(f"  Intento {attempt}/{retries} falló: {e}")
            try:
                sock.close()
            except Exception:
                pass

            if attempt < retries:
                import time
                time.sleep(retry_delay)

    raise RuntimeError(
        f"No se pudo conectar a {host}:{port}. "
        f"Último error: {last_error}"
    )


# ═══════════════════════════════════════════════════════════
# API PÚBLICA
# ═══════════════════════════════════════════════════════════
def send_msg(sock: socket.socket, obj: Any) -> int:
    """
    Serializa obj con pickle y lo envía con header de longitud fija.
    """
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack(HEADER_FMT, len(payload))
    sock.sendall(header + payload)
    return HEADER_SIZE + len(payload)


def recv_msg(sock: socket.socket) -> Any:
    """
    Recibe y deserializa el siguiente mensaje.
    """
    header = recv_exact(sock, HEADER_SIZE)
    n_bytes = struct.unpack(HEADER_FMT, header)[0]
    payload = recv_exact(sock, n_bytes)
    return pickle.loads(payload)