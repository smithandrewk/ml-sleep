import torch
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate(dataloader, model, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_y_true = []
    all_y_pred = []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1)
        targets = y.argmax(dim=1) if y.dim() > 1 else y
        all_y_true.extend(targets.cpu().numpy())
        all_y_pred.extend(preds.cpu().numpy())
    avg_loss = total_loss / n_batches
    return avg_loss, np.array(all_y_true), np.array(all_y_pred)


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize='true'):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize=normalize, ax=ax)
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
