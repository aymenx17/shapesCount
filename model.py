




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from resnet import resnet34


class AttentionModule(nn.Module):
    """ A neural module that takes a feature map, attends to the features, and
    produces an attention.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.dim = dim

    def forward(self, feats):
        out = F.relu(self.conv1(feats))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))
        return out


class QueryModule(nn.Module):
    """ A neural module that takes as input a feature map and an attention and produces a feature
    map as output.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        return out



class Classifier(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ShapeRecognizer(nn.Module):
    def __init__(self):
        super(ShapeRecognizer, self).__init__()

        self.resnet = resnet34(True)

        # square modules
        self.square_attn = AttentionModule(dim=256)
        self.query = QueryModule(dim=256)
        self.classifier = Classifier(dim=256)
        # circle modules
        self.circle_attn = AttentionModule(dim=256)

        # triangle modules
        self.triangle_attn = AttentionModule(dim=256)


    def forward(self, x):

        # feature analysis
        f = self.resnet(x)

        # square classifier
        sa = self.square_attn(f)
        sq = self.query(f,sa)
        sc = self.classifier(sq)

        # circle classifier
        ca = self.circle_attn(f)
        cq = self.query(f,ca)
        cc = self.classifier(cq)

        # triangle classifier
        ta = self.triangle_attn(f)
        tq = self.query(f,ta)
        tc = self.classifier(tq)

        return sc, cc, tc, sa, ca, ta
