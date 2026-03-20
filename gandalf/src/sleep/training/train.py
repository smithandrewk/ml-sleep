import torch
from tqdm import tqdm


def training_loop(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def dev_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches
