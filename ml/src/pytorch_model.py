"""
PyTorch Multimodal / Tabular Deep Learning model architecture
for multi-class thermal classification.
"""

import torch
import torch.nn as nn


class MultimodalThermalNet(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        image_dim: int = 512,
        num_classes: int = 5
    ):
        super().__init__()

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, tabular_x, image_x):
        tabular_features = self.tabular_encoder(tabular_x)
        image_features = self.image_encoder(image_x)

        fused_features = torch.cat(
            [tabular_features, image_features],
            dim=1
        )

        logits = self.classifier(fused_features)

        return logits