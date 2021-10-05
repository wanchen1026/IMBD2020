import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 downsample=False, preactivate=True):
        super(ResBlock, self).__init__()

        self.residual = []
        self.shortcut = []
        if preactivate:
            self.residual.extend([nn.BatchNorm1d(in_channels), nn.ReLU()])
            self.shortcut.extend([nn.BatchNorm1d(in_channels), nn.ReLU()])

        self.residual.extend([
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
        ])
        self.shortcut.extend([
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
        ])

        if downsample:
            self.residual.append(nn.AvgPool1d(2))
            self.shortcut.append(nn.AvgPool1d(2))

        self.residual = nn.Sequential(*self.residual)
        self.shortcut = nn.Sequential(*self.shortcut)

    def forward(self, inputs):
        outputs = self.residual(inputs) + self.shortcut(inputs)
        return outputs


class ResEncoderFront(nn.Module):
    def __init__(self, in_channels, ch):
        super().__init__()
        self.encoder = nn.Sequential(
            ResBlock(in_channels, ch, downsample=True, preactivate=False),
            ResBlock(ch * 1, ch * 2, downsample=True),
            ResBlock(ch * 2, ch * 4),
            ResBlock(ch * 4, ch * 4),
            nn.BatchNorm1d(ch * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        outputs = torch.flatten(outputs, 1)
        return outputs


class ResEncoderOther(nn.Module):
    def __init__(self, in_channels, ch):
        super().__init__()
        self.encoder = nn.Sequential(
            ResBlock(in_channels, ch, downsample=True, preactivate=False),
            ResBlock(ch * 1, ch * 2, downsample=True),
            ResBlock(ch * 2, ch * 4, downsample=True),
            ResBlock(ch * 4, ch * 4, downsample=True),
            ResBlock(ch * 4, ch * 4),
            nn.BatchNorm1d(ch * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        outputs = torch.flatten(outputs, 1)
        return outputs


class GroupClassifier(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super(GroupClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 7),
        )

    def forward(self, inputs):
        outputs = self.classifier(inputs)
        outputs = torch.flatten(outputs, 1)
        return outputs


class PieceClassifier(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super(PieceClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 32),
        )

    def forward(self, inputs):
        outputs = self.classifier(inputs)
        outputs = torch.flatten(outputs, 1)
        return outputs
