"""Sanity checks on prepared data that the validation split relies on."""

from __future__ import annotations

import numpy as np


def group_coherence(
    embeddings: np.ndarray, keys: np.ndarray, group_size: int
) -> dict[str, float | int]:
    """Test whether images with consecutive ``keys`` form groups of ``group_size``.

    The validation split holds out images in groups of consecutive stimulus
    ids, assuming each group is one object concept. If that is true, two
    neighbouring ids inside a group show the same concept and their embeddings
    are much more alike than two neighbouring ids on either side of a group
    boundary. Returns the mean cosine similarity (after removing the mean
    embedding) of both kinds of neighbour pairs and the fraction of boundaries
    whose similarity is below that of the pairs just inside the groups around it.
    """
    order = np.argsort(keys)
    keys = np.asarray(keys)[order]
    centred = np.asarray(embeddings, dtype=np.float64)[order]
    centred = centred - centred.mean(axis=0)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)
    adjacent = keys[1:] == keys[:-1] + 1
    similarity = (centred[1:] * centred[:-1]).sum(axis=1)
    boundary = adjacent & (keys[1:] // group_size != keys[:-1] // group_size)
    within = adjacent & ~boundary
    if not boundary.any() or not within.any():
        raise ValueError("need neighbouring ids both inside and across groups")

    # A boundary pair is "clean" if it is less alike than the pairs on both sides of it.
    index = np.flatnonzero(boundary)
    index = index[(index > 0) & (index < len(similarity) - 1)]
    index = index[within[index - 1] & within[index + 1]]
    clean = similarity[index] < np.minimum(similarity[index - 1], similarity[index + 1])
    return {
        "within_group_similarity": float(similarity[within].mean()),
        "across_boundary_similarity": float(similarity[boundary].mean()),
        "clean_boundary_fraction": float(clean.mean()) if len(clean) else float("nan"),
        "n_within_pairs": int(within.sum()),
        "n_boundary_pairs": int(boundary.sum()),
    }
