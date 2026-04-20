import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F
from config import USE_SPECAUGMENT, USE_MIXUP, P_MIXUP
import random

freq_mask = T.FrequencyMasking(freq_mask_param=20)
time_mask = T.TimeMasking(time_mask_param=30)

class ICBHIDataset(Dataset):
    def __init__(self, npz_path, train=False):
        data       = np.load(npz_path)
        self.X     = data["X"]    # [N, 128, 251] déjà prétraité
        self.Y     = data["Y"]    # [N, 4] one-hot
        self.L     = data["L"]    # [N] labels entiers
        self.train = train
        self.class_to_indices = {
            int(c): np.where(self.L == c)[0].tolist()
            for c in np.unique(self.L)
        }

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        mel    = torch.tensor(self.X[idx]).unsqueeze(0)  # [1,128,251]
        target = torch.tensor(self.Y[idx])
        label  = torch.tensor(self.L[idx], dtype=torch.long)

        # Mixup train only
        if self.train and USE_MIXUP and np.random.rand() < P_MIXUP:
            if int(label) == 1 and 2 in self.class_to_indices:
                idx2 = random.choice(self.class_to_indices[2])
            elif int(label) == 2 and 1 in self.class_to_indices:
                idx2 = random.choice(self.class_to_indices[1])
            else:
                idx2 = random.randint(0, len(self.X)-1)
            lam    = np.random.beta(0.4, 0.4)
            mel    = lam * mel    + (1-lam) * torch.tensor(self.X[idx2]).unsqueeze(0)
            target = lam * target + (1-lam) * torch.tensor(self.Y[idx2])

        # SpecAugment train only
        if self.train and USE_SPECAUGMENT:
            mel = time_mask(mel)
            mel = freq_mask(mel)

        return mel, target, label


def get_datasets():
    train = ICBHIDataset("data/preprocessed/train.npz", train=True)
    test  = ICBHIDataset("data/preprocessed/test.npz",  train=False)
    print(f"Train: {len(train)} | Test: {len(test)}")
    return train, test