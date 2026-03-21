#!/usr/bin/env python3
"""Profile training throughput at different batch sizes for Gandalf."""
import sys
import time

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, '../gandalf/src')
from sleep.models import Gandalf
from sleep.data.splits import get_leave_one_out_folds
from sleep.data.datasets import load_eeg_label_pair

N_BATCHES = 20  # batches to time per batch size
BATCH_SIZES = [32, 64, 128, 256, 512]
CONFIG_PATH = '../gandalf/configs/gandalf_loo.yaml'

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load data once
data_path = cfg['data']['data_path']
dataset = cfg['data']['dataset']
conditions = cfg['data']['conditions']
window_size = cfg['model']['sequence_length']

folds = get_leave_one_out_folds(data_path)
train_ids, _ = folds[0]

print('Loading data...')
subjects = [
    load_eeg_label_pair(data_path, dataset, id=id, condition=cond, zero_pad=True, window_size=window_size)
    for id in train_ids for cond in conditions
]
Xs = [s[0] for s in subjects]
ys = [s[1] for s in subjects]
n_samples = sum(len(y) for y in ys)
eps = len(ys[0])

train_idx, _ = train_test_split(range(n_samples), test_size=0.1, random_state=0, shuffle=True)


class _SSDataset(torch.utils.data.Dataset):
    def __init__(self, Xs, ys, idx, epochs_per_subject):
        self.Xs, self.ys, self.idx = Xs, ys, idx
        self.eps = epochs_per_subject

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        i = self.idx[index]
        return (
            self.Xs[i // self.eps][(i % self.eps):(i % self.eps) + window_size].flatten(),
            self.ys[i // self.eps][i % self.eps],
        )


ds = _SSDataset(Xs, ys, train_idx, eps)

print(f'\nTotal training samples: {len(ds):,}')
print(f'Timing {N_BATCHES} batches per batch size\n')
print(f'{"batch_size":>10} {"batches/epoch":>13} {"sec/batch":>10} {"samples/sec":>12} {"est epoch":>10} {"est 180 epochs":>15} {"VRAM (MB)":>10}')
print('-' * 85)

for bs in BATCH_SIZES:
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    n_batches_per_epoch = len(loader)

    model = Gandalf(
        n_features=cfg['model']['n_features'],
        lstm_hidden=cfg['model']['lstm_hidden'],
        n_classes=cfg['model']['n_classes'],
        sequence_length=cfg['model']['sequence_length'],
    )
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])

    # Warmup
    it = iter(loader)
    X, y = next(it)
    X, y = X.to(device), y.to(device)
    model(X)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Time N_BATCHES
    times = []
    for i, (X, y) in enumerate(loader):
        if i >= N_BATCHES:
            break
        X, y = X.to(device), y.to(device)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times)
    samples_per_sec = bs / avg_time
    epoch_time = avg_time * n_batches_per_epoch
    total_time = epoch_time * 180

    vram = torch.cuda.max_memory_allocated() // (1024 * 1024) if device == 'cuda' else 0
    torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None

    def fmt_time(s):
        if s < 60:
            return f'{s:.0f}s'
        elif s < 3600:
            return f'{s/60:.1f}m'
        else:
            return f'{s/3600:.1f}h'

    print(f'{bs:>10} {n_batches_per_epoch:>13} {avg_time:>10.3f} {samples_per_sec:>12.0f} {fmt_time(epoch_time):>10} {fmt_time(total_time):>15} {vram:>10}')

    del model, optimizer
    torch.cuda.empty_cache() if device == 'cuda' else None
