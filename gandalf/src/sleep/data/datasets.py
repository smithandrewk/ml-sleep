import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


def load_eeg_label_pair(data_path, dataset, id, condition, zero_pad=True, window_size=9):
    X, y = torch.load(
        os.path.join(data_path, dataset, f'{id}_{condition}.pt'),
        weights_only=False,
    )
    if zero_pad:
        pad = torch.zeros(window_size // 2, X.shape[1])
        X = torch.cat([pad, X, pad])
    return X, y


class WindowedDataset(Dataset):
    def __init__(self, X, y, window_size=9):
        pad = torch.zeros(window_size // 2, X.shape[1])
        self.X = torch.cat([pad, X, pad])
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx:idx + 9].flatten(), self.y[idx]


class EpochedDataset(Dataset):
    def __init__(self, data_path, dataset, id, condition, robust=True, downsampled=True):
        if robust:
            ds = 'pt_ekyn_robust_50hz' if downsampled else 'pt_ekyn_robust'
        else:
            ds = dataset
        X, y = torch.load(
            os.path.join(data_path, ds, f'{id}_{condition}.pt'),
            weights_only=False if not robust else True,
        )
        self.X = X
        self.y = y
        self.id = id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx:idx + 1], self.y[idx]


class SequencedDataset(Dataset):
    def __init__(self, data_path, id, condition, sequence_length, robust=True, downsampled=True, stride=1):
        self.sequence_length = sequence_length
        self.stride = stride
        self.id = id

        if robust:
            ds = 'pt_ekyn_robust_50hz' if downsampled else 'pt_ekyn_robust'
            X, y = torch.load(os.path.join(data_path, ds, f'{id}_{condition}.pt'), weights_only=True)
        else:
            X, y = torch.load(os.path.join(data_path, 'pt_ekyn', f'{id}_{condition}.pt'), weights_only=False)

        self.num_features = X.shape[1]
        self.num_classes = y.shape[1]
        self.X = torch.cat([
            torch.zeros(sequence_length // 2, self.num_features),
            X,
            torch.zeros(sequence_length // 2, self.num_features),
        ]).unsqueeze(1)
        self.y = torch.cat([
            torch.zeros(sequence_length // 2, self.num_classes),
            y,
            torch.zeros(sequence_length // 2, self.num_classes),
        ])

    def __len__(self):
        return (self.X.shape[0] - self.sequence_length + 1) // self.stride

    def __getitem__(self, idx):
        idx = self.stride * idx + self.sequence_length // 2
        half = self.sequence_length // 2
        return self.X[idx - half:idx + half + 1], self.y[idx]


def get_epoched_dataloader(data_path, ids, conditions, batch_size=512, shuffle=True, robust=True):
    return DataLoader(
        dataset=ConcatDataset([
            EpochedDataset(data_path, 'pt_ekyn', id=id, condition=cond, robust=robust, downsampled=True)
            for id in ids for cond in conditions
        ]),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
    )


def get_sequenced_dataloader(data_path, ids, conditions, sequence_length=3, batch_size=512, shuffle=True, robust=True):
    return DataLoader(
        dataset=ConcatDataset([
            SequencedDataset(data_path, id=id, condition=cond, sequence_length=sequence_length, robust=robust)
            for id in ids for cond in conditions
        ]),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
    )
