#!/usr/bin/env python3
"""
Explore variance distributions across animals and conditions.

For each animal+condition recording:
  - Compute variance of each epoch (across 5000 samples)
  - Plot the distribution of these variances
  - Compare within-animal (PF vs Vehicle) vs between-animal similarity
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
import seaborn as sns

DATA_PATH = '/home/andrew/sleep/pt_ekyn'
OUT_DIR = '/home/andrew/sleep/ml-sleep/variance-signatures/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Load all variance distributions
files = sorted(os.listdir(DATA_PATH))
ids = sorted(set(f.split('_')[0] for f in files))
conditions = ['Vehicle', 'PF']

variances = {}  # (animal_id, condition) -> array of per-epoch variances

print('Computing per-epoch variances...')
for animal_id in ids:
    for cond in conditions:
        X, y = torch.load(os.path.join(DATA_PATH, f'{animal_id}_{cond}.pt'), weights_only=False)
        # X is (8640, 1, 5000) — variance across the 5000 samples per epoch
        v = X.squeeze(1).var(dim=1).numpy()
        variances[(animal_id, cond)] = v
        print(f'  {animal_id} {cond}: mean_var={v.mean():.2f}, std_var={v.std():.2f}')

# ============================================================
# Plot 1: Overlay PF vs Vehicle for each animal
# ============================================================
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
for i, animal_id in enumerate(ids):
    ax = axes[i // 4, i % 4]
    for cond, color in zip(conditions, ['tab:blue', 'tab:orange']):
        v = variances[(animal_id, cond)]
        ax.hist(v, bins=80, alpha=0.5, label=cond, color=color, density=True)
    ax.set_title(animal_id, fontsize=10)
    if i == 0:
        ax.legend(fontsize=8)
fig.suptitle('Per-epoch variance distributions: PF vs Vehicle within each animal', fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/within_animal_pf_vs_vehicle.png', dpi=150)
plt.close(fig)
print(f'Saved within-animal comparison')

# ============================================================
# Plot 2: All animals overlaid (one color per animal)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ci, cond in enumerate(conditions):
    ax = axes[ci]
    for animal_id in ids:
        v = variances[(animal_id, cond)]
        ax.hist(v, bins=80, alpha=0.3, density=True, label=animal_id)
    ax.set_title(f'{cond} — all animals', fontsize=12)
    ax.set_xlabel('Epoch variance')
    ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/all_animals_overlay.png', dpi=150)
plt.close(fig)
print(f'Saved all-animals overlay')

# ============================================================
# Plot 3: Pairwise distance matrix (Wasserstein + KS)
# ============================================================
keys = [(aid, cond) for aid in ids for cond in conditions]
n = len(keys)
W = np.zeros((n, n))
KS = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        vi = variances[keys[i]]
        vj = variances[keys[j]]
        W[i, j] = W[j, i] = wasserstein_distance(vi, vj)
        KS[i, j] = KS[j, i] = ks_2samp(vi, vj).statistic

labels = [f'{k[0]}_{k[1][:3]}' for k in keys]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

ax = axes[0]
im = ax.imshow(W, cmap='viridis')
ax.set_xticks(range(n))
ax.set_xticklabels(labels, rotation=90, fontsize=5)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=5)
ax.set_title('Wasserstein distance')
fig.colorbar(im, ax=ax, shrink=0.7)

ax = axes[1]
im = ax.imshow(KS, cmap='viridis')
ax.set_xticks(range(n))
ax.set_xticklabels(labels, rotation=90, fontsize=5)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=5)
ax.set_title('KS statistic')
fig.colorbar(im, ax=ax, shrink=0.7)

fig.suptitle('Pairwise distance between variance distributions', fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/pairwise_distances.png', dpi=150)
plt.close(fig)
print(f'Saved pairwise distance matrices')

# ============================================================
# Quantify: within-animal vs between-animal distances
# ============================================================
within = []
between = []

for i, ki in enumerate(keys):
    for j, kj in enumerate(keys):
        if j <= i:
            continue
        if ki[0] == kj[0]:  # same animal, different condition
            within.append(W[i, j])
        else:
            between.append(W[i, j])

print(f'\n=== Wasserstein distances ===')
print(f'Within-animal (PF vs Vehicle):  mean={np.mean(within):.4f}, std={np.std(within):.4f}, n={len(within)}')
print(f'Between-animal:                 mean={np.mean(between):.4f}, std={np.std(between):.4f}, n={len(between)}')
print(f'Ratio (between/within):         {np.mean(between)/np.mean(within):.2f}x')

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(within, bins=20, alpha=0.6, label=f'Within-animal (n={len(within)})', density=True)
ax.hist(between, bins=40, alpha=0.6, label=f'Between-animal (n={len(between)})', density=True)
ax.set_xlabel('Wasserstein distance')
ax.set_title('Within-animal vs Between-animal variance distribution distance')
ax.legend()
fig.savefig(f'{OUT_DIR}/within_vs_between.png', dpi=150)
plt.close(fig)
print(f'Saved within vs between comparison')

# ============================================================
# Plot 5: Can we classify animal identity from variance dist?
# Simple: use log-variance histogram as feature vector, run kNN
# ============================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

n_bins = 50
features = []
animal_labels = []

for animal_id in ids:
    for cond in conditions:
        v = variances[(animal_id, cond)]
        hist, _ = np.histogram(np.log1p(v), bins=n_bins, density=True)
        features.append(hist)
        animal_labels.append(animal_id)

X_feat = np.array(features)
y_labels = np.array(animal_labels)

# LOO CV with kNN
knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
scores = cross_val_score(knn, X_feat, y_labels, cv=len(y_labels))  # LOO
print(f'\n=== Animal identification from variance distribution ===')
print(f'1-NN LOO accuracy: {scores.mean():.1%} ({int(scores.sum())}/{len(scores)})')
