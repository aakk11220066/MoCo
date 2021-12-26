#!/usr/bin/env python
# coding: utf-8

import torch
from types import MethodType


# Apply only feature extractor of RESNET50
def _ftr_extract(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def get_encoder():
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        
        # convert forward function to _ftr_extract in model to convert it into a feature extractor
        model.forward = MethodType(_ftr_extract, model)
        
        return model

# Test feature extractor
# model = get_encoder()
# model(torch.arange(64*3*7*7).reshape((64,3,7,7)).float()).dtype
