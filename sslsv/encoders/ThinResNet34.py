import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

class AudioPreEmphasis(nn.Module):

    def __init__(self, coeff=0.97):
        super().__init__()

        self.w = torch.FloatTensor([-coeff, 1.0]).unsqueeze(0).unsqueeze(0)

    def forward(self, audio):
        audio = audio.unsqueeze(1)
        audio = F.pad(audio, (1, 0), 'reflect')
        return F.conv1d(audio, self.w.to(audio.device)).squeeze(1)


class ResNetBlock(nn.Module):

    def __init__(self, in_size, out_size, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_size,
            out_size,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_size)

        self.conv2 = nn.Conv2d(
            out_size,
            out_size,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_size)

        self.se = SELayer(out_size)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, X):
        residual = X
        if self.downsample: residual = self.downsample(residual)

        Z = self.conv1(X)
        Z = self.relu(Z)
        Z = self.bn1(Z)

        Z = self.conv2(Z)
        Z = self.bn2(Z)
        Z = self.se(Z)
        
        Z += residual
        Z = self.relu(Z)
        return Z


class SELayer(nn.Module):

    def __init__(self, in_size, reduction=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_size, in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_size // reduction, in_size),
            nn.Sigmoid()
        )

    def forward(self, X):
        b, c, _, _ = X.size()

        Y = self.fc(self.avg_pool(X).view(b, c)).view(b, c, 1, 1)
        return X * Y


class SAP(nn.Module):

    def __init__(self, out_size, dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(out_size, dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, out_size, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, X):
        b, c, h, w = X.size()

        X = X.reshape(b, -1, w)
        W = self.attention(X)
        return torch.sum(W * X, dim=2)


class ThinResNet34(nn.Module):

    def __init__(self, encoded_dim=1024, n_mels=40):
        super().__init__()

        self.features_extractor = nn.Sequential(
            AudioPreEmphasis(),
            MelSpectrogram(
                n_fft=512,
                win_length=400,
                hop_length=160,
                window_fn=torch.hamming_window,
                n_mels=n_mels
            )
        )
        self.instance_norm = nn.InstanceNorm1d(n_mels)

        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(32)

        self.block1 = self.__make_block(3, 32, 32, 1)
        self.block2 = self.__make_block(4, 32, 64, 2)
        self.block3 = self.__make_block(6, 64, 128, 2)
        self.block4 = self.__make_block(3, 128, 256, 2)

        sap_out_size = int(n_mels / 8 * 256)
        self.sap = SAP(sap_out_size)

        self.fc = nn.Linear(sap_out_size, encoded_dim)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_block(self, num_layers, in_size, out_size, stride):
        layers = []
        layers.append(ResNetBlock(in_size, out_size, stride))
        for i in range(1, num_layers):
            layers.append(ResNetBlock(out_size, out_size, 1))
        return nn.Sequential(*layers)

    def forward(self, X):
        # X shape: (B, T) = (B, 200)

        with torch.no_grad():
            X = self.features_extractor(X) + 1e-6
            X = X.log()
            X = self.instance_norm(X)
            X = X.unsqueeze(1)
            # X shape: (B, H, W, C) = (B, 1, 40, 200)

        Z = self.conv(X)
        Z = self.relu(Z)
        Z = self.bn(Z)

        Z = self.block1(Z)
        Z = self.block2(Z)
        Z = self.block3(Z)
        Z = self.block4(Z)

        Z = self.sap(Z)
        Z = self.fc(Z)

        return Z