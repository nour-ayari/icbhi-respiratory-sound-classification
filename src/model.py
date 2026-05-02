

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel


class CustomAST(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        pretrained_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        dropout: float = 0.4,
        freeze_ast: bool = True,
        # Dimensions attendues par ce modèle AST pré-entraîné (patch 10x10)
        # mel_bins=128, max_length=1024  → position embeddings figés à ces valeurs
        ast_mel_bins: int = 128,
        ast_max_length: int = 1024,
    ):
        super().__init__()

        self.ast_mel_bins   = ast_mel_bins
        self.ast_max_length = ast_max_length

        # ── Backbone ────────────────────────────────────────────
        self.ast = ASTModel.from_pretrained(pretrained_name)
        hidden_size = self.ast.config.hidden_size  # 768

        # ── Tête de classification ───────────────────────────────
        # 2 couches comme le repo de référence mais avec LayerNorm
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        if freeze_ast:
            self.freeze_backbone()

    # ── Gestion du freeze ────────────────────────────────────────
    def freeze_backbone(self):
        """Gèle tout le backbone AST (phase 1 : warmup classifier)."""
        for p in self.ast.parameters():
            p.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int = 4):
        """
        Phase 2 : dégèle les n dernières couches Transformer + layernorm.
        Garde le reste frozen pour ne pas perdre les features AudioSet.
        """
        # D'abord tout regeler
        for p in self.ast.parameters():
            p.requires_grad = False

        # Dégeler les n dernières couches encoder
        for layer in self.ast.encoder.layer[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        # LayerNorm final toujours entraînable
        if hasattr(self.ast, "layernorm"):
            for p in self.ast.layernorm.parameters():
                p.requires_grad = True

    def get_trainable_params(self) -> dict:
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_M": total / 1e6, "trainable_M": trainable / 1e6}

    # ── Forward ──────────────────────────────────────────────────
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise x vers [B, ast_mel_bins, ast_max_length].

        Ton preprocess produit [B, 1, 128, T] avec T variable (~157 frames).
        L'AST MIT attend [B, 128, 1024].

        STRATÉGIE : pad/crop en dimension temporelle uniquement (pas de resize
        bilinéaire complet qui détruit les patterns fréquentiels).
        On garde les 128 mel bins intacts.
        """
        # [B, 1, 128, T] → [B, 128, T]
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Vérif bins mel
        if x.shape[1] != self.ast_mel_bins:
            # Rare, mais on resize seulement la dim fréquence si nécessaire
            x = F.interpolate(
                x.unsqueeze(1),
                size=(self.ast_mel_bins, x.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # Pad ou crop en dimension temporelle UNIQUEMENT
        T = x.shape[-1]
        if T < self.ast_max_length:
            # Pad cyclique (reproduit le signal audio court)
            repeats = (self.ast_max_length // T) + 1
            x = x.repeat(1, 1, repeats)[:, :, : self.ast_max_length]
        elif T > self.ast_max_length:
            # Crop centré
            start = (T - self.ast_max_length) // 2
            x = x[:, :, start : start + self.ast_max_length]

        # ASTModel attend [B, time, mel] = [B, 1024, 128]
        # Notre preprocess produit [B, mel, time] = [B, 128, 1024] → transposer
        x = x.permute(0, 2, 1)  # [B, 1024, 128]

        # Normaliser pour correspondre à la distribution ASTFeatureExtractor
        # (le backbone a été pré-entraîné avec mean≈0, std≈0.5)
        # Nos spectros sont en dB brut (mean≈-61, std≈19) → incompatible
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std  = x.std(dim=(-2, -1), keepdim=True).clamp(min=1.0)  # min=1.0 fp16-safe
        x = (x - mean) / (2.0 * std)  # std≈0.5, mean≈0 comme ASTFeatureExtractor

        return x  # [B, 1024, 128]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)          # [B, 128, 1024]
        outputs = self.ast(x)               # last_hidden_state: [B, seq, 768]

        # Mean pooling sur la séquence (plus stable que CLS seul)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # [B, 768]
        logits = self.classifier(embeddings)                # [B, num_classes]
        return logits