import numpy as np
from tqdm import trange


def _grid_index(value: float, n_grids: int) -> int:
    clipped = min(max(abs(float(value)), 0.0), 0.999999)
    return int(clipped * n_grids) + 1


def adversarial_refine(model1_score_gap, model2_score_gap):
    model1_score_gap = np.asarray(model1_score_gap, dtype=float)
    model2_score_gap = np.asarray(model2_score_gap, dtype=float)

    if model1_score_gap.shape != model2_score_gap.shape:
        raise ValueError("score gap arrays must have the same shape")

    n_grids = 50
    keep = []

    score_to_idx = {}
    for i in range(1, n_grids + 1):
        for j in range(1, n_grids + 1):
            score_to_idx[(i, j)] = []
            score_to_idx[(-i, -j)] = []
            score_to_idx[(i, -j)] = []
            score_to_idx[(-i, j)] = []

    model1_zero_idx = []
    model2_zero_idx = []
    for i, (model1_gap, model2_gap) in enumerate(zip(model1_score_gap, model2_score_gap)):
        if model1_gap == 0 and model2_gap == 0:
            keep.append(i)
            continue
        if model1_gap == 0:
            model1_zero_idx.append(i)
            continue
        if model2_gap == 0:
            model2_zero_idx.append(i)
            continue

        model1_id = _grid_index(model1_gap, n_grids)
        model2_id = _grid_index(model2_gap, n_grids)
        if model1_gap > 0 and model2_gap > 0:
            score_to_idx[(model1_id, model2_id)].append(i)
        elif model1_gap < 0 and model2_gap < 0:
            score_to_idx[(-model1_id, -model2_id)].append(i)
        elif model1_gap > 0 and model2_gap < 0:
            score_to_idx[(model1_id, -model2_id)].append(i)
        elif model1_gap < 0 and model2_gap > 0:
            score_to_idx[(-model1_id, model2_id)].append(i)
        else:
            raise ValueError("unexpected score gap sign combination")

    for i in range(1, n_grids + 1):
        for j in range(1, n_grids + 1):
            idx = score_to_idx[(i, j)]
            op_idx = score_to_idx[(-i, -j)]
            if idx and op_idx:
                if len(idx) > len(op_idx):
                    keep.extend(op_idx)
                    keep.extend(list(np.random.choice(idx, len(op_idx), replace=False)))
                else:
                    keep.extend(idx)
                    keep.extend(list(np.random.choice(op_idx, len(idx), replace=False)))

            idx = score_to_idx[(i, -j)]
            op_idx = score_to_idx[(-i, j)]
            if idx and op_idx:
                if len(idx) > len(op_idx):
                    keep.extend(op_idx)
                    keep.extend(list(np.random.choice(idx, len(op_idx), replace=False)))
                else:
                    keep.extend(idx)
                    keep.extend(list(np.random.choice(op_idx, len(idx), replace=False)))

    zero_grid_count = 10

    if model1_zero_idx:
        pos_score_to_idx = {i: [] for i in range(zero_grid_count)}
        neg_score_to_idx = {i: [] for i in range(zero_grid_count)}
        for i in model1_zero_idx:
            gap = model2_score_gap[i]
            bucket = min(int(min(abs(gap), 0.999999) * zero_grid_count), zero_grid_count - 1)
            if gap > 0:
                pos_score_to_idx[bucket].append(i)
            else:
                neg_score_to_idx[bucket].append(i)
        for i in trange(zero_grid_count):
            pos_idx = pos_score_to_idx[i]
            neg_idx = neg_score_to_idx[i]
            if len(pos_idx) > len(neg_idx):
                keep.extend(neg_idx)
                keep.extend(list(np.random.choice(pos_idx, len(neg_idx), replace=False)))
            else:
                keep.extend(pos_idx)
                keep.extend(list(np.random.choice(neg_idx, len(pos_idx), replace=False)))

    if model2_zero_idx:
        pos_score_to_idx = {i: [] for i in range(zero_grid_count)}
        neg_score_to_idx = {i: [] for i in range(zero_grid_count)}
        for i in model2_zero_idx:
            gap = model1_score_gap[i]
            bucket = min(int(min(abs(gap), 0.999999) * zero_grid_count), zero_grid_count - 1)
            if gap > 0:
                pos_score_to_idx[bucket].append(i)
            else:
                neg_score_to_idx[bucket].append(i)
        for i in trange(zero_grid_count):
            pos_idx = pos_score_to_idx[i]
            neg_idx = neg_score_to_idx[i]
            if len(pos_idx) > len(neg_idx):
                keep.extend(neg_idx)
                keep.extend(list(np.random.choice(pos_idx, len(neg_idx), replace=False)))
            else:
                keep.extend(pos_idx)
                keep.extend(list(np.random.choice(neg_idx, len(pos_idx), replace=False)))

    return sorted(set(int(i) for i in keep))
