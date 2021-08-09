import torch.nn as nn
import pretrainedmodels
import torch
import pretrainedmodels.utils
import torchvision.models as models


# List of the pretrained backbone can be found at:
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/__init__.py

class ClassificationModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet34", pretrained=True):
        super(ClassificationModel, self).__init__()

        # Load a pretrained backbone
        self.backbone = getattr(models, backbone)(pretrained=pretrained)

        # Customize the last linear layer (i.e. dense) for each model
        if backbone == 'resnext101_32x8d' or backbone == 'resnet34':
            dim_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(dim_feats, num_classes)

        if backbone == 'densenet121':
            dim_feats = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(dim_feats, num_classes)

        if backbone == 'mobilenet_v2':
            dim_feats = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        # Perform a feed-forward through the backbone
        x = self.backbone(x)
        x = torch.relu(x)
        return x


class RegressionModel(nn.Module):
    def __init__(self, num_classes, backbone='se_resnext50_32x4d', pretrained='imagenet'):
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
        x = self.backbone(x)
        x = x + self.linear_1_bias
        x = torch.sigmoid(x)
        return x


def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    print('Get model used')
    return model
