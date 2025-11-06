import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset
import torch
import os

class EEGDataset(Dataset):
    def __init__(self, args=None, X_path=None, y_path=None):
        """
        EEG dataset loader supporting both default and pre-split paths.
        """
        self.args = args
        self.dataset_name = getattr(args, "dataset_name", "unknown")

        # ✅ If explicit npy paths are provided
        if X_path is not None and y_path is not None:
            print(f"📂 Loading dataset directly from provided paths:\n  X: {X_path}\n  y: {y_path}")
            X = np.load(X_path)
            y = np.load(y_path)
        else:
            # Default legacy path
            base_path = os.path.join("./data", self.dataset_name)
            X = np.load(os.path.join(base_path, 'X.npy'))
            y = np.load(os.path.join(base_path, 'labels.npy'))

        print(f"original data shape: {X.shape}, labels shape: {y.shape}")

        # Label normalization
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        print(f"preprocessed data shape: {X.shape}, labels shape: {y.shape}")

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )
