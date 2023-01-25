import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reparametrize(mu, logvar, factor=0.2):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + factor*std*eps

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes, ):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.layers = nn.Linear(in_dim, num_classes)

        self.fc = nn.Linear(2048 * 1, num_classes)

        self.p_logvar = nn.Sequential(nn.Linear(2048 * 1, 512),
                                      nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(2048 * 1, 512),
                                  nn.LeakyReLU())

    def forward(self, features, cam=None, l2d=None):
        scores = self.layers(features)
        if cam is not None:
            cam_s = F.conv2d(cam, self.fc.weight.view(self.fc.out_features, cam.size(1), 1, 1)) + self.fc.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return scores,cam_s
        
        if l2d is not None:
            logvar = self.p_logvar(features)
            mu = self.p_mu(features)
            if l2d == 'Train':
                x = reparametrize(mu, logvar)
            elif l2d == "Test":
                x = mu
            scores = self.layers(x)
            return scores, mu, logvar
            
        return scores


class Masker(nn.Module):
    def __init__(self, in_dim=2048, num_classes=2048, middle =8192, k = 1024):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
       mask = self.bn(self.layers(f))
       z = torch.zeros_like(mask)
       for _ in range(self.k):
           mask = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=False)
           z = torch.maximum(mask,z)
       return z

## Added
# From https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/
class ProjectionHead(nn.Module):
    def __init__(self,embedding_dim,projection_dim=256,dropout=0.2):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x