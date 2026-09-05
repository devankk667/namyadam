"""
PyTorch Multimodal / Tabular Deep Learning model architecture for multi-class thermal classification.
"""

import torch
import torch.nn as nn

class MultimodalThermalNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 5, hidden_dims: list = [128, 64, 32]):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
