import os
import torch
from torch import nn
from torch.utils.data import DataLoader,ConcatDataset,TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,classification_report
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def ekyn_ids():
    return sorted(set([filename.split('_')[0] for filename in os.listdir(f'../pt_ekyn_robust_50hz')]))
    
def moving_average(data, window_size=10):
    """Compute the moving average of a list."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def get_dataloaders(train_ids,test_ids,path_to_dataset,batch_size=32,num_workers=0):
    dataset = ConcatDataset([TensorDataset(*torch.load(f'{path_to_dataset}/{id}_{condition}.pt',weights_only=False)) for id in train_ids for condition in ['PF','Vehicle']])
    labels = torch.tensor([data[1].argmax().item() for data in dataset])
    class_counts = torch.bincount(labels)
    class_weights = 1. / class_counts.float()
    weights = class_weights[labels]
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),num_workers=num_workers)

    unweighted_trainloader = DataLoader(ConcatDataset([TensorDataset(*torch.load(f'{path_to_dataset}/{id}_{condition}.pt',weights_only=False)) for id in train_ids for condition in ['PF','Vehicle']]),batch_size=batch_size)
    unweighted_testloader = DataLoader(ConcatDataset([TensorDataset(*torch.load(f'{path_to_dataset}/{id}_{condition}.pt',weights_only=False)) for id in test_ids for condition in ['PF','Vehicle']]),batch_size=batch_size)
    print('train samples',len(trainloader)*batch_size,train_ids)
    print('test samples',len(unweighted_testloader)*batch_size,test_ids)
    return trainloader,unweighted_trainloader,unweighted_testloader

def evaluate(dataloader,model,criterion,device):
    with torch.no_grad():
        p = torch.vstack([torch.hstack([model(Xi.to(device)),yi.to(device)]) for Xi,yi in dataloader]).cpu()
        p = torch.hstack([p,p[:,:3].softmax(dim=1).argmax(axis=1).unsqueeze(1)])
        logits = p[:,:3]
        y_true = p[:,3:6].argmax(axis=1)
        y_pred = p[:,6:]
        f1 = f1_score(y_true,y_pred,average='macro')
        loss = criterion(logits,y_true).item()
        report = classification_report(y_pred=y_pred,y_true=y_true,output_dict=True)
    return loss,f1,report

def update_plot(epoch, config, loss_offset,  window_size):
    trainlossi = config["trainlossi"][loss_offset:]
    testlossi = config["testlossi"][loss_offset:]
    trainf1i = config["trainf1i"][loss_offset:]
    testf1i = config["testf1i"][loss_offset:]
    trainf1p = config["trainf1p"][loss_offset:]
    trainf1s = config["trainf1s"][loss_offset:]
    trainf1w = config["trainf1w"][loss_offset:]
    testf1p = config["testf1p"][loss_offset:]
    testf1s = config["testf1s"][loss_offset:]
    testf1w = config["testf1w"][loss_offset:]
    best_dev_loss = config["best_dev_loss"]
    best_dev_loss_epoch = config["best_dev_loss_epoch"]-loss_offset
    best_dev_f1 = config["best_dev_f1"]
    best_dev_f1_epoch = config["best_dev_f1_epoch"]-loss_offset

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))  # Adjusted height for better aspect ratio
    # Define colors for train and test
    colors = {
        'Train': '#007AFF',  # Apple Blue
        'Test': '#FF9500'    # Apple Orange
    }
    calm_warm_palette = ['#007AFF', '#40E0D0', '#34C759']  # Apple inspired colors

    # First subplot - Loss
    ax1.plot(trainlossi, label='Train Loss', color=colors['Train'], alpha=0.4, linewidth=1.5)
    ax1.plot(testlossi, label='Test Loss', color=colors['Test'], alpha=0.4, linewidth=1.5)
    
    # Moving average for loss
    if len(trainlossi) > window_size:
        ax1.plot(range(window_size-1, len(trainlossi)), moving_average(trainlossi, window_size), label='Train Loss MA', color=colors['Train'], linestyle='--', linewidth=1.5)
    if len(testlossi) > window_size:
        ax1.plot(range(window_size-1, len(testlossi)), moving_average(testlossi, window_size), label='Test Loss MA', color=colors['Test'], linestyle='--', linewidth=1.5)
    ax1.axhline(best_dev_loss, color='#FF4500', linestyle='--', linewidth=1.5)
    ax1.axvline(best_dev_loss_epoch, color='#FF4500', linestyle='--', linewidth=1.5)
    ax1.set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax1.grid(True, linestyle=':', color='#F5F5F5', alpha=0.7)

    # Second subplot - F1 Score
    ax2.plot(trainf1i, label='Train F1', color=colors['Train'], alpha=0.4, linewidth=1.5)
    ax2.plot(testf1i, label='Test F1', color=colors['Test'], alpha=0.4, linewidth=1.5)
    # Moving average for F1
    if len(trainf1i) > window_size:
        ax2.plot(range(window_size-1, len(trainf1i)), moving_average(trainf1i, window_size), label='Train F1 MA', color=colors['Train'], linestyle='--', linewidth=1.5)
    if len(testf1i) > window_size:
        ax2.plot(range(window_size-1, len(testf1i)), moving_average(testf1i, window_size), label='Test F1 MA', color=colors['Test'], linestyle='--', linewidth=1.5)
    ax2.axhline(best_dev_f1, color='#FF4500', linestyle='--', linewidth=1.5)
    ax2.axvline(best_dev_f1_epoch, color='#FF4500', linestyle='--', linewidth=1.5)
    ax2.set_title('F1 Score Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.grid(True, linestyle=':', color='#F5F5F5', alpha=0.7)

    # Third subplot - F1 Scores by class
    labels = ['Paradoxical', 'Slow Wave', 'Wakefulness']
    for label, color, train_data, test_data in zip(labels, calm_warm_palette, [trainf1p, trainf1s, trainf1w], [testf1p, testf1s, testf1w]):
        if len(train_data) > window_size:
            ax3.plot(range(window_size-1, len(train_data)), moving_average(train_data, window_size), color=color, linestyle='-', linewidth=1.5, label=f'{label} (Train)')
        if len(test_data) > window_size:
            ax3.plot(range(window_size-1, len(test_data)), moving_average(test_data, window_size), color=color, linestyle='--', linewidth=1.5, label=f'{label} (Test)')

    ax3.set_title('Performance by Sleep Stage', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epochs', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax3.grid(True, linestyle=':', color='#F5F5F5', alpha=0.7)

    # Adjust layout to prevent overlapping with increased margins
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # This increases the left margin and shifts the legend to the right
