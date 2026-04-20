import torch
import torch.nn as nn
from config import NUM_CLASSES


class MultiScaleResidualBlock(nn.Module):
    """
    Bloc multi-échelle avec connexion résiduelle légère.
    Plus stable que le bloc purement concat+proj.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        b1 = out_ch // 4
        b2 = out_ch // 4
        b3 = out_ch // 4
        b4 = out_ch - (b1 + b2 + b3)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, b1, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(b1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, b2, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(b2),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, b3, kernel_size=(3, 11), padding=(1, 5), bias=False),
            nn.BatchNorm2d(b3),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, b4, kernel_size=(5, 15), padding=(2, 7), bias=False),
            nn.BatchNorm2d(b4),
            nn.ReLU(inplace=True)
        )

        self.proj = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch else
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

        out = self.proj(out)
        out = out + self.shortcut(x)
        return self.act(out)


class CNNBiGRU(nn.Module):
    """
    Entrée : [B, 1, F, T]
    Sortie : logits [B, num_classes]
    """
    def __init__(self, num_classes=NUM_CLASSES, gru_hidden=96, gru_layers=1):
        super().__init__()

        self.features = nn.Sequential(
            MultiScaleResidualBlock(1, 32),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.08),

            MultiScaleResidualBlock(32, 64),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.10),

            MultiScaleResidualBlock(64, 128),
            nn.MaxPool2d((2, 1)),   # préserve mieux l’axe temps
            nn.Dropout2d(0.12),

            MultiScaleResidualBlock(128, 160),
            nn.Dropout2d(0.15)
        )

        # Compression fréquentielle uniquement
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.bigru = nn.GRU(
            input_size=160,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if gru_layers == 1 else 0.2
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)          # [B, 160, F', T']
        x = self.freq_pool(x)         # [B, 160, 1, T']
        x = x.squeeze(2)              # [B, 160, T']
        x = x.permute(0, 2, 1)        # [B, T', 160]
        self.bigru.flatten_parameters()  # ← ajouter

        seq, _ = self.bigru(x)        # [B, T', 2H]

        # Mean pooling temporel
        x = seq.mean(dim=1)           # [B, 2H]

        return self.classifier(x)