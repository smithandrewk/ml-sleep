#!/usr/bin/env python3
"""Evaluate trained Gandalf models across all LOO folds."""
import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from sleep.models import Gandalf
from sleep.data.splits import get_leave_one_out_folds
from sleep.data.datasets import load_eeg_label_pair, WindowedDataset
from sleep.training.evaluate import evaluate, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gandalf across LOO folds')
    parser.add_argument('-c', '--config', type=str, default='configs/gandalf_loo.yaml')
    parser.add_argument('-m', '--models-dir', type=str, default=None, help='Directory containing gandalf_N/ folders')
    parser.add_argument('-f', '--fold', type=int, default=None, help='Evaluate single fold (default: all)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Save results figure to path')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    data_path = cfg['data']['data_path']
    dataset = cfg['data']['dataset']
    conditions = cfg['data']['conditions']
    models_dir = args.models_dir or cfg['output']['experiments_path']

    folds = get_leave_one_out_folds(data_path)
    fold_range = [args.fold] if args.fold is not None else range(len(folds))

    criterion = torch.nn.CrossEntropyLoss()
    reports = []

    for fold_idx in tqdm(fold_range, desc='Folds'):
        train_ids, test_ids = folds[fold_idx]
        model_path = os.path.join(models_dir, f'gandalf_{fold_idx}', 'best_model.pt')
        if not os.path.exists(model_path):
            print(f'Skipping fold {fold_idx}: no model at {model_path}')
            continue

        model = Gandalf(
            n_features=cfg['model']['n_features'],
            lstm_hidden=cfg['model']['lstm_hidden'],
            n_classes=cfg['model']['n_classes'],
            sequence_length=cfg['model']['sequence_length'],
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.to(device)

        for condition in conditions:
            X, y = load_eeg_label_pair(data_path, dataset, id=test_ids[0], condition=condition, zero_pad=False)
            testloader = DataLoader(WindowedDataset(X, y), batch_size=cfg['evaluation']['batch_size'], shuffle=False)
            loss, y_true, y_pred = evaluate(testloader, model, criterion, device)
            report = classification_report(y_true, y_pred, output_dict=True)
            report['fold'] = fold_idx
            report['condition'] = condition
            reports.append(report)

    if not reports:
        print('No models found to evaluate.')
        return

    # Summary
    df = pd.DataFrame({
        'precision': [r['macro avg']['precision'] for r in reports],
        'recall': [r['macro avg']['recall'] for r in reports],
        'f1-score': [r['macro avg']['f1-score'] for r in reports],
        'fold': [r['fold'] for r in reports],
        'condition': [r['condition'] for r in reports],
    })
    print(df.describe())

    long_df = df.melt(id_vars=['fold', 'condition'], var_name='metric', value_name='value')
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='metric', y='value', data=long_df)
    plt.title('LOO CV: Macro Avg Metrics')
    if args.output:
        plt.savefig(args.output)
        print(f'Saved to {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
