# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12HJTZ3kKrm6pxhl7W4AMUhcbjqMzwSrI
"""

import torch
import torch.nn as nn
from torchvision import models


def initialize_model(num_classes, learning_rate):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=1e-4)

    return model, criterion, optimizer