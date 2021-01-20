import torch
from torch.utils.data import Dataset

class ExperimentData(Dataset):
    def __init__(self, X, y, c):
        super().__init__()
        """
        PyTorch data loader for experimental data
        X : array-like, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,)
            Target values (checkout amount) for regression.
        c : array-like, shape (n_samples,)
            Target values (class labels) for classification.
        """
        self.X = X
        self.y = y
        self.c = c

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx], self.c[idx]

class WeightedExperimentData(ExperimentData):
    def __init__(self, X, y, c, w):
        super().__init__(X, y, c)
        """
        w : array-like, shape (n_samples,)
            Class weight for classification.
        """
        self.w = w

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx], self.c[idx], self.w[idx]

class TreatedWeightedExperimentData(WeightedExperimentData):
    def __init__(self, X, y, c, w, g):
        super().__init__(X, y, c, w)
        """
        g : array-like, shape (n_samples,)
            Binary treatment assignment.
        """
        self.g = g

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx], self.c[idx], self.w[idx], self.g[idx]
