"""
Cancer Detection CNN Model
==========================
Convolutional Neural Network for binary classification:
    0 = Benign
    1 = Malignant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CancerCNN(nn.Module):
    """
    CNN Architecture for Cancer Detection from histopathology images.
    
    Input:  (batch, 3, 224, 224)  — RGB image
    Output: (batch, 2)            — logits [benign, malignant]
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super(CancerCNN, self).__init__()

        # ── Block 1 ──────────────────────────────────────────────
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)          # 112 × 112

        # ── Block 2 ──────────────────────────────────────────────
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)          # 56 × 56

        # ── Block 3 ──────────────────────────────────────────────
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)          # 28 × 28

        # ── Block 4 ──────────────────────────────────────────────
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7   = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8   = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)          # 14 × 14

        # ── Global Average Pooling ────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)        # 1 × 1

        # ── Classifier ───────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)

        # GAP → flatten
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    # ─────────────────────────────────────────────────────────────
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# ── Transfer-learning variant (ResNet-18 backbone) ────────────────
class CancerCNNTransfer(nn.Module):
    """
    ResNet-18 fine-tuned for cancer detection.
    Better accuracy when you have limited labelled data.
    """

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False):
        super(CancerCNNTransfer, self).__init__()
        import torchvision.models as models

        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            # Fallback: random init (weights will be fine-tuned during training)
            self.backbone = models.resnet18(weights=None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    model = CancerCNN()
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"CancerCNN output shape : {out.shape}")          # (4, 2)
    print(f"Probabilities          : {model.predict_proba(dummy)[0].detach()}")
    print("Model architecture:\n", model)
