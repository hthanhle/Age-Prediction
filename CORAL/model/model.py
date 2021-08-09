"""
UOW, Wed Feb 24 23:37:42 2021
Dependencies: torch>=1.1, torchvision>=0.3.0
"""

import torch.nn as nn
import torch
import torchvision.models as models


class RegressionModel(nn.Module):
    def __init__(self, num_classes, backbone='mobilenet_v2', pretrained=True):
        super(RegressionModel, self).__init__()

        # Load a pretrained backbone
        self.backbone = getattr(models, backbone)(pretrained=pretrained)

        # Customize the last linear layer (i.e. dense) for each model
        if backbone == 'resnext101_32x8d' or backbone == 'resnet34':
            dim_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(dim_feats, 1)

        if backbone == 'densenet121':
            dim_feats = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(dim_feats, 1)

        if backbone == 'mobilenet_v2':
            dim_feats = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(dim_feats, 1)

        self.linear_1_bias = nn.Parameter(torch.zeros(num_classes).float())

    def forward(self, x):
        # Perform a feed-forward through the backbone
        logits = self.backbone(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas
