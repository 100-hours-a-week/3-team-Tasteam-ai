import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _get_continuous_feature_count(root: str) -> int:
    path = Path(root) / "feature_sizes.txt"
    if not path.exists():
        return 13
    sizes = np.loadtxt(str(path), delimiter=",", dtype=np.int64)
    if sizes.ndim == 0:
        sizes = np.array([sizes])
    n = 0
    for s in sizes:
        if s == 1:
            n += 1
        else:
            break
    return n if n > 0 else 13


def _get_total_field_count(root: str) -> int:
    path = Path(root) / "feature_sizes.txt"
    if not path.exists():
        return 39
    sizes = np.loadtxt(str(path), delimiter=",", dtype=np.int64)
    if sizes.ndim == 0:
        sizes = np.array([sizes])
    return len(sizes)


class CriteoDataset(Dataset):
    """
    Criteo/Tasteam 전처리 결과용 Dataset.
    - feature_sizes.txt로 연속형 개수·전체 필드 수 추론.
    - train.txt에 sample_weight 컬럼이 있으면 (Xi, Xv, y, sample_weight) 반환.
    """
    def __init__(self, root, train=True, use_val_file=False):
        self.root = root
        self.train = train
        self.continous_features = _get_continuous_feature_count(root)
        self.total_fields = _get_total_field_count(root)

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        if self.train:
            train_path = os.path.join(root, "train.txt")
            data = pd.read_csv(train_path)
            n_feat = self.total_fields
            self.train_data = data.iloc[:, :n_feat].values
            self.target = data.iloc[:, n_feat].values
            if data.shape[1] > n_feat + 1:
                self.sample_weight = data.iloc[:, n_feat + 1].values.astype(np.float32)
            else:
                self.sample_weight = None
        else:
            test_path = os.path.join(root, "val.txt" if use_val_file else "test.txt")
            if not os.path.exists(test_path):
                test_path = os.path.join(root, "test.txt")
            data = pd.read_csv(test_path)
            n_feat = self.total_fields
            self.test_data = data.iloc[:, :n_feat].values
            self.test_has_label = data.shape[1] > n_feat
            if self.test_has_label:
                self.test_target = data.iloc[:, n_feat].values
            else:
                self.test_target = None

    def __getitem__(self, idx):
        n = self.continous_features
        if self.train:
            dataI = self.train_data[idx, :]
            targetI = self.target[idx]
            Xi_cont = np.zeros_like(dataI[:n])
            Xi_cat = dataI[n:]
            Xi = torch.from_numpy(
                np.concatenate((Xi_cont, Xi_cat)).astype(np.int32)
            ).unsqueeze(-1)
            Xv_cat = np.ones_like(dataI[n:])
            Xv_cont = dataI[:n]
            Xv = torch.from_numpy(
                np.concatenate((Xv_cont, Xv_cat)).astype(np.float32)
            )
            if self.sample_weight is not None:
                sw = torch.tensor(self.sample_weight[idx], dtype=torch.float32)
                return Xi, Xv, targetI, sw
            return Xi, Xv, targetI
        else:
            dataI = self.test_data[idx, :]
            Xi_cont = np.zeros_like(dataI[:n])
            Xi_cat = dataI[n:]
            Xi = torch.from_numpy(
                np.concatenate((Xi_cont, Xi_cat)).astype(np.int32)
            ).unsqueeze(-1)
            Xv_cat = np.ones_like(dataI[n:])
            Xv_cont = dataI[:n]
            Xv = torch.from_numpy(
                np.concatenate((Xv_cont, Xv_cat)).astype(np.float32)
            )
            if self.test_has_label and self.test_target is not None:
                return Xi, Xv, self.test_target[idx]
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)


def has_sample_weight(root: str) -> bool:
    """train.txt에 sample_weight 컬럼이 있는지."""
    path = Path(root) / "train.txt"
    if not path.exists():
        return False
    df = pd.read_csv(path, nrows=1)
    n = _get_total_field_count(root)
    return df.shape[1] > n + 1
