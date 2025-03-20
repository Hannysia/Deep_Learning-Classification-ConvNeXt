#model.py

import torch
import torch.nn as nn
from torchvision import models

IMAGES_SIZE=(224, 224)

class ConvNeXtTinyModel(nn.Module):
    def __init__(self, num_classes=31):
        super(ConvNeXtTinyModel, self).__init__()
        self.model = models.convnext_tiny(weights=None)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model_eval=ConvNeXtTinyModel()
model_eval.load_state_dict(torch.load('model.pt', weights_only=True))