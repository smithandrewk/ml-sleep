#!/usr/bin/env python3
"""Train Gandalf (ResNet-LSTM) for sleep staging with leave-one-out CV."""
import argparse
import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from sleep.models import Gandalf
from sleep.data.splits import get_leave_one_out_folds
from sleep.data.datasets import load_eeg_label_pair, WindowedDataset
from sleep.training.train import training_loop, dev_loop
from sleep.training.evaluate import evaluate, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='Train Gandalf sleep staging model')
    parser.add_argument('-c', '--config', type=str, default='configs/gandalf_loo.yaml')
    parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('-d', '--device', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seed everything
    seed = cfg['training']['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'

    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = os.path.join(cfg['output']['experiments_path'], f'gandalf_{args.fold}')
    run_dir = os.path.join(exp_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Data
    data_path = cfg['data']['data_path']
    dataset = cfg['data']['dataset']
    conditions = cfg['data']['conditions']
    window_size = cfg['model']['sequence_length']

    folds = get_leave_one_out_folds(data_path)
    train_ids, test_ids = folds[args.fold]

    subjects = [
        load_eeg_label_pair(data_path, dataset, id=id, condition=cond, zero_pad=True, window_size=window_size)
        for id in train_ids for cond in conditions
    ]
    Xs = [s[0] for s in subjects]
    ys = [s[1] for s in subjects]

    n_samples = sum(len(y) for y in ys)
    train_idx, dev_idx = train_test_split(
        range(n_samples), test_size=cfg['training']['dev_split'],
        random_state=cfg['training']['random_seed'], shuffle=True,
    )

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

    eps = len(ys[0])  # epochs per subject
    batch_size = cfg['training']['batch_size']
    g = torch.Generator()
    g.manual_seed(seed)
    trainloader = DataLoader(_SSDataset(Xs, ys, train_idx, eps), batch_size=batch_size, shuffle=True, generator=g)
    devloader = DataLoader(_SSDataset(Xs, ys, dev_idx, eps), batch_size=batch_size, shuffle=True)
    print(f'trainloader: {len(trainloader)} batches | devloader: {len(devloader)} batches')

    # Model
    model = Gandalf(
        n_features=cfg['model']['n_features'],
        lstm_hidden=cfg['model']['lstm_hidden'],
        n_classes=cfg['model']['n_classes'],
        sequence_length=cfg['model']['sequence_length'],
    )
    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])

    # Training
    best_dev_loss = float('inf')
    unimproved = 0
    patience = cfg['training']['patience']
    loss_tr, loss_dev = [], []

    pbar = tqdm(range(cfg['training']['epochs']))
    for epoch in pbar:
        loss_tr.append(training_loop(model, trainloader, criterion, optimizer, device))
        loss_dev.append(dev_loop(model, devloader, criterion, device))

        if loss_dev[-1] < best_dev_loss:
            best_dev_loss = loss_dev[-1]
            best_epoch = epoch
            unimproved = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
        else:
            unimproved += 1
            if patience and unimproved >= patience:
                break

        pbar.set_description(
            f'train: {loss_tr[-1]:.4f} | dev: {loss_dev[-1]:.4f} | '
            f'best: {best_dev_loss:.4f} | stop: {unimproved}'
        )

        # Save loss plots
        for suffix, data in [('last_30', (-30, None)), ('all', (None, None))]:
            plt.figure()
            plt.plot(loss_tr[data[0]:data[1]], label='train')
            plt.plot(loss_dev[data[0]:data[1]], label='dev')
            plt.legend()
            plt.savefig(os.path.join(run_dir, f'loss_{suffix}.jpg'))
            plt.close()

        torch.save(model.state_dict(), os.path.join(run_dir, f'{epoch}.pt'))

    # Final eval on dev set
    _, y_true, y_pred = evaluate(devloader, model, criterion, device)
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(run_dir, 'cm_last_dev.jpg'))

    # Save final model + config
    torch.save(model.state_dict(), os.path.join(exp_dir, 'last_model.pt'))
    run_config = {
        **cfg,
        'fold': args.fold,
        'run_id': run_id,
        'best_dev_loss': best_dev_loss,
        'best_epoch': best_epoch,
        'total_epochs': len(loss_tr),
        'parameters': params,
    }
    for d in [exp_dir, run_dir]:
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump(run_config, f, indent=2)

    print(f'Done. Best dev loss: {best_dev_loss:.4f} at epoch {best_epoch}')


if __name__ == '__main__':
    main()
