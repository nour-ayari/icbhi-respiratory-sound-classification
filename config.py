import torch, os

# ── Paths ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "ICBHI_final_database")
SPLIT_FILE = os.path.join(BASE_DIR, "data", "ICBHI_challenge_train_test.txt")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")
RES_DIR    = os.path.join(BASE_DIR, "results")
FIG_DIR    = os.path.join(BASE_DIR, "figures")

# ── Audio ─────────────────────────────────────────────
TARGET_SR  = 16000
TARGET_SEC = 5         # ← remettre 8s, 5s perd trop d'info
TARGET_LEN = TARGET_SR * TARGET_SEC
N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 512
FMIN       = 50
FMAX       = 8000

# ── Modèle ────────────────────────────────────────────
NUM_CLASSES       = 4
CNN_DIM           = 256
AST_DIM           = 768
FREEZE_AST_LAYERS = 10

# ── Entraînement ──────────────────────────────────────
EPOCHS       = 25        # ← une seule définition
BATCH_SIZE   = 16
ACCUM_STEPS  = 2
LR           = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA  = 1.5
RHO_SAM      = 0.01

# ── Augmentation ──────────────────────────────────────
USE_SPECAUGMENT = True
USE_MIXUP       = True
P_PITCH         = 0.4   # ← réactiver
P_MIXUP         = 0.3   # ← réactiver modérément
RARE_CLASSES    = [2, 3]

# ── Divers ────────────────────────────────────────────
LABEL_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']
SEED        = 42

# ── Device ────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {DEVICE}")