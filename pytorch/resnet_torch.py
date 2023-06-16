#!/usr/bin/env python
import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.dilation = dilation
        if stride > 1:
            if dilation > 1:
                stride = 1
            self.downsample = nn.Conv3d(in_channels, channels, kernel_size=1, stride=stride, dilation=dilation, padding='valid', bias=False)
            self.bn0 = nn.BatchNorm3d(channels)
            self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=(3,3,3), stride=stride, dilation=dilation, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=(3,3,3), stride=stride, padding='same', bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3,3,3), stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.stride > 1:
            residual = self.downsample(residual)
            residual = self.bn0(residual)
        x += residual
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.dilation = dilation
        if stride > 1:
            self.downsample = nn.Conv3d(in_channels, channels*self.expansion, kernel_size=1, stride=stride, bias=False)
            self.bn0 = nn.BatchNorm3d(channels*self.expansion)
        elif in_channels != channels*self.expansion:
            self.downsample = nn.Conv3d(in_channels, channels*self.expansion, kernel_size=1, stride=1, bias=False)
            self.bn0 = nn.BatchNorm3d(channels*self.expansion)
        else:
            self.downsample = None
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=(1,1,1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()
        if stride > 1 or dilation > 1:
            self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3,3,3), stride=stride, padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3,3,3), stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv3 = nn.Conv3d(channels, channels*self.expansion, kernel_size=(1,1,1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels*self.expansion)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x





class ResNet3D18(nn.Module):
    def __init__(self, in_channels=3, dropout=0.3, stride=2, dilation=1):
        super(ResNet3D18, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(7,7,7), stride=2, padding='valid', bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(3,3,3), stride=2, dilation=1, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=stride, dilation=dilation),
            BasicBlock(128, 128))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=stride, dilation=dilation),
            BasicBlock(256, 256))
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=stride, dilation=dilation),
            BasicBlock(512, 512))
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dense1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.classify = nn.Linear(32, 1)

    def forward(self, x, policy=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.dropout(x)
        x = self.classify(x)

        return x


class ResNet3D50(nn.Module):
    def __init__(self, in_channels=3, dropout=0.3):
        super(ResNet3D50, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(7,7,7), stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(3,3,3), stride=2, dilation=1, padding=1)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64))
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, stride=2),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 128))
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, stride=1, dilation=2),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256))
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, stride=1, dilation=4),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        #self.dense1 = nn.Linear(512*self.expansion, 128)
        #self.dropout = nn.Dropout(dropout)
        #self.dense2 = nn.Linear(128, 64)
        #self.dense3 = nn.Linear(64, 32)
        self.classify = nn.Linear(512*self.expansion, 1)

    def forward(self, x, policy=None):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.dense1(x)
        #x = self.dropout(x)
        #x = self.dense2(x)
        #x = self.dropout(x)
        #x = self.dense3(x)
        #x = self.dropout(x)
        x = self.classify(x)

        return x


            






