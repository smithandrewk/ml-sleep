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
        print(f'  {animal_id} {cond}: mean_var={v.mean():.2e}, std_var={v.std():.2e}')

# ============================================================
# Plot 1: Overlay PF vs Vehicle for each animal
# ============================================================
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for i, animal_id in enumerate(ids):
    ax = axes[i // 4, i % 4]
    for cond, color in zip(conditions, ['tab:blue', 'tab:orange']):
        v = np.sort(variances[(animal_id, cond)])
        cdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, cdf, color=color, label=cond)
    ax.set_xlim(0, 2e-8)
    ax.set_title(animal_id, fontsize=10)
    if i == 0:
        ax.legend(fontsize=8)
fig.suptitle('CDF of per-epoch variance: PF vs Vehicle within each animal', fontsize=14)
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
        v = np.sort(variances[(animal_id, cond)])
        cdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, cdf, alpha=0.7, label=animal_id)
    ax.set_xlim(0, 2e-8)
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
print(f'Within-animal (PF vs Vehicle):  mean={np.mean(within):.2e}, std={np.std(within):.2e}, n={len(within)}')
print(f'Between-animal:                 mean={np.mean(between):.2e}, std={np.std(between):.2e}, n={len(between)}')
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
# Train on Vehicle, predict PF (and vice versa)
# ============================================================
from sklearn.neighbors import KNeighborsClassifier

n_bins = 50

# Build feature vectors: log-variance histogram per recording
def make_features(animal_ids, condition):
    X, y = [], []
    for aid in animal_ids:
        v = variances[(aid, condition)]
        hist, _ = np.histogram(np.log1p(v), bins=n_bins, density=True)
        X.append(hist)
        y.append(aid)
    return np.array(X), np.array(y)

print(f'\n=== Animal identification from variance distribution ===')
knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')

# Train on Vehicle, test on PF
X_train, y_train = make_features(ids, 'Vehicle')
X_test, y_test = make_features(ids, 'PF')
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
acc = (preds == y_test).mean()
print(f'Train Vehicle -> Predict PF:  {acc:.1%} ({int((preds == y_test).sum())}/{len(y_test)})')
for pred, true in zip(preds, y_test):
    marker = 'OK' if pred == true else 'MISS'
    print(f'  {true}: predicted {pred} [{marker}]')

# Train on PF, test on Vehicle
X_train, y_train = make_features(ids, 'PF')
X_test, y_test = make_features(ids, 'Vehicle')
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
acc = (preds == y_test).mean()
print(f'Train PF -> Predict Vehicle:  {acc:.1%} ({int((preds == y_test).sum())}/{len(y_test)})')
for pred, true in zip(preds, y_test):
    marker = 'OK' if pred == true else 'MISS'
    print(f'  {true}: predicted {pred} [{marker}]')
