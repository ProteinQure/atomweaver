"""Shared geometry and element features for read-out fitting and inference."""

import torch


def geometry_features(coords: torch.Tensor, mask: torch.Tensor, elements: torch.Tensor):
    """Encode masked pair distances followed by five-way per-slot element one-hots.

    Inputs have shapes (N, S, 3), (N, S), and (N, S). Element IDs are database
    IDs (C=0, N=1, O=2, X=3); masked slots use the PAD column (4). Slot order,
    the -1 distance sentinel, and float32 output match the released heads.
    """
    count, slots = mask.shape
    upper = torch.triu_indices(slots, slots, 1)
    distances = torch.cdist(coords, coords)[:, upper[0], upper[1]]
    valid_pairs = mask[:, upper[0]] & mask[:, upper[1]]
    distances = torch.where(valid_pairs, distances, torch.full_like(distances, -1.0))
    element_ids = elements.clone()
    element_ids[~mask] = 4
    element_ids = element_ids.clamp(0, 4).long()
    one_hot = torch.zeros(count, slots, 5)
    one_hot.scatter_(2, element_ids.unsqueeze(2), 1.0)
    return torch.cat([distances, one_hot.reshape(count, -1)], 1).numpy()
