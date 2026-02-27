import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _get_continuous_feature_count(root: str) -> int:
    """feature_sizes.txt에서 연속형 피처 개수(값이 1인 필드 수)를 반환. 없으면 Criteo 호환 13."""
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


class CriteoDataset(Dataset):
    """
    Criteo/Tasteam 전처리 결과용 Dataset.
    feature_sizes.txt가 있으면 연속형 개수를 자동 추론(Tasteam 호환).
    """
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.continous_features = _get_continuous_feature_count(root)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(os.path.join(root, 'train.txt'))
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, 'test.txt'))
            # test.txt도 마지막 컬럼이 라벨일 수 있음
            self.test_data = data.iloc[:, :-1].values

    def __getitem__(self, idx):
        n = self.continous_features
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            Xi_cont = np.zeros_like(dataI[:n])
            Xi_cat = dataI[n:]
            Xi = torch.from_numpy(np.concatenate((Xi_cont, Xi_cat)).astype(np.int32)).unsqueeze(-1)
            Xv_cat = np.ones_like(dataI[n:])
            Xv_cont = dataI[:n]
            Xv = torch.from_numpy(np.concatenate((Xv_cont, Xv_cat)).astype(np.float32))
            return Xi, Xv, targetI
        else:
            dataI = self.test_data[idx, :]
            Xi_cont = np.zeros_like(dataI[:n])
            Xi_cat = dataI[n:]
            Xi = torch.from_numpy(np.concatenate((Xi_cont, Xi_cat)).astype(np.int32)).unsqueeze(-1)
            Xv_cat = np.ones_like(dataI[n:])
            Xv_cont = dataI[:n]
            Xv = torch.from_numpy(np.concatenate((Xv_cont, Xv_cat)).astype(np.float32))
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)