import os
from random import seed, shuffle


def get_ekyn_ids(data_path):
    for dataset_name in ['pt_ekyn_robust_50hz', 'pt_ekyn']:
        dataset_path = os.path.join(data_path, dataset_name)
        if not os.path.exists(dataset_path):
            continue
        return sorted(list(set(
            f.split('_')[0] for f in os.listdir(dataset_path)
        )))
    return None


def get_leave_one_out_folds(data_path):
    ids = get_ekyn_ids(data_path)
    seed(0)
    shuffle(ids)
    folds = []
    for test_id in ids:
        train_ids = [x for x in ids if x != test_id]
        folds.append((train_ids, [test_id]))
    return folds
