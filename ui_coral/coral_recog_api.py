#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/9/13 16:15
@description: A class that can call a trained model to achieve the classification
"""
import torch
from torchvision import transforms
from PIL import Image


class recog:
    def __init__(self,model,device):
        super(recog, self).__init__()
        self.device = device
        self.model = torch.load(model, map_location=self.device)
        self.model.to(self.device)
        self.classes = self.get_classes()
    def get_classes(self):
        classFile = "models/coral.names"
        with open(classFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        return classes

    def coral_recog(self, img):
        img = Image.fromarray(img)
        to_tensor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.401, 0.395, 0.406], [0.181, 0.176, 0.179])
        ])
        img_tensor = to_tensor(img)

        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        img_tensor = img_tensor.to(self.device)

        outputs = self.model(img_tensor)
        _, indices = torch.max(outputs, 1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0]
        perc = round(percentage[int(indices)].item(),4)
        indices = indices.cpu().numpy()[0]
        return self.classes[indices], perc


