import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

class perceptionLoss():
    def __init__(self, device):
        vgg = torchvision.models.vgg19(pretrained=True)
        vgg.eval()
        self.features = vgg.features.to(device)
        self.feature_layers = ['4', '9', '18', '27', '36']
        self.mse_loss = nn.MSELoss()

    def getfeatures(self, x):
        feature_list = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.feature_layers:
                feature_list.append(x)
        return feature_list

    def calculatePerceptionLoss(self, video_pd, video_gt):
        features_pd = self.getfeatures(video_pd.view(video_pd.size(0)*video_pd.size(2), video_pd.size(1), video_pd.size(3), video_pd.size(4)))
        features_gt = self.getfeatures(video_gt.view(video_gt.size(0)*video_gt.size(2), video_gt.size(1), video_gt.size(3), video_gt.size(4)))
        
        with torch.no_grad():
            features_gt = [x.detach() for x in features_gt]
        
        perceptual_loss = sum([self.mse_loss(features_pd[i], features_gt[i]) for i in range(len(features_gt))])
        return perceptual_loss