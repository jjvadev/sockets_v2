# CIFAR-100 — Entrenamiento distribuido v2

Entrenamiento distribuido tipo **FedAvg** sobre CIFAR-100 con CNN convolucional.

## Configuración de esta versión

| Parámetro        | Valor         | Motivo                                              |
|------------------|---------------|-----------------------------------------------------|
| Dataset          | CIFAR-100     | 50 000 train / 10 000 test — conjunto completo      |
| Rounds           | 50            | Suficiente para convergencia sin augmentation       |
| Local epochs     | 2             | Más progreso por round → menos rounds necesarios    |
| Batch size       | 256           | Batch grande → cada round más rápido                |
| Activation       | Leaky ReLU    | Más robusto que ReLU puro (sin neuronas muertas)    |
| Augmentation     | ❌ desactivado | Simplifica el pipeline, tiempos más predecibles     |
| Optimizer        | Adam lr=1e-3  | Convergencia estable sin tuning de LR               |
| Weight decay     | 1e-4          | Regularización leve, evita sobreajuste              |
| Dropout          | 0.3           | Regularización adicional                            |
| Arquitectura CNN | 64→128→256    | Suficiente capacidad para CIFAR-100 en CPU          |

---

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux/Mac
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

---

## Ejecución

### Servidor

```bash
# Con los defaults de esta versión (recomendado):
python3 server.py --workers 2

# Especificando todos los parámetros explícitamente:
python3 server.py \
    --workers 2 \
    --rounds 50 \
    --local-epochs 2 \
    --batch-size 256 \
    --activation leaky_relu \
    --port 65432

# Más workers para acelerar el entrenamiento:
python3 server.py --workers 4 --port 65432
python3 server.py --workers 7 --port 65432
```

### Workers (en cada máquina)

```bash
python3 worker.py --host <IP_SERVIDOR> --port 65432
```

---

## Tiempos estimados (CPU, sin GPU)

| Workers | Tiempo aprox. 50 rounds |
|---------|------------------------|
| 1       | 3 – 10 min             |
| 2       | 2 – 5 min              |
| 4       | 1 – 3 min              |

> Con GPU los tiempos se reducen drásticamente.

---

## Resultados

Los resultados se guardan en `results/<run_name>/`:

```
results/
└── cifar100_w2_r50_le2_bs256_0.001_leaky_relu/
    ├── history.csv        ← métricas por round
    ├── summary.json       ← resumen del experimento
    ├── best_model.pt      ← mejor modelo (mejor test acc)
    ├── final_model.pt     ← modelo al finalizar
    ├── analysis.ipynb     ← notebook generado automáticamente
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── time_per_round.png
    ├── time_cumulative.png
    └── generalization_gap.png
```
