import torch.nn as nn


def get_activation(name: str, negative_slope: float = 0.01):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    raise ValueError(f"Activación no soportada: {name}")


class CIFAR100CNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        activation: str = "relu",
        negative_slope: float = 0.01,
        dropout: float = 0.3,
        channels=(64, 128, 256),
        fc_dim: int = 256,
    ):
        super().__init__()

        c1, c2, c3 = channels
        act = lambda: get_activation(activation, negative_slope)

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            act(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            act(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            act(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            act(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            act(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            act(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, fc_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x