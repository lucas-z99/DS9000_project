"""CNN for grayscale characters; accepts the existing flattened shard batches."""
import torch.nn as nn


class CNN_Classifier(nn.Module):
    def __init__(self, n_class, image_size=48, dropout_p=0.2, leaky=False):
        super().__init__()
        if image_size < 16:
            raise ValueError("image_size must be at least 16")
        self.image_size = image_size
        self.model_type = 'cnn_v2' if leaky else 'cnn'
        activation = (lambda: nn.LeakyReLU(0.1)) if leaky else nn.ReLU
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), activation(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), activation(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), activation(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 3 * 3, 256), activation(),
            nn.Dropout(dropout_p), nn.Linear(256, n_class),
        )

    def forward(self, x):
        if x.ndim == 2 and x.shape[1] == self.image_size ** 2:
            x = x.reshape(-1, 1, self.image_size, self.image_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, self.image_size, self.image_size):
            raise ValueError("Expected flattened grayscale images or [N, 1, H, W]")
        return self.classifier(self.features(x))
