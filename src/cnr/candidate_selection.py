"""Select only from a shared candidate pool; invalid-pool fallback is explicit."""
import numpy as np


def select_candidates(groups, valid, scores, order):
    selected, fallback = [], []
    for group in groups:
        # Shared random priority makes both random selection and tied scores fair.
        candidates = np.asarray(group)[np.argsort(order[group])]
        eligible = candidates[np.asarray(valid)[candidates]]
        failed = len(eligible) == 0
        if failed:
            eligible = candidates
        chosen = eligible[0] if scores is None or failed else eligible[np.argmax(np.asarray(scores)[eligible])]
        selected.append(int(chosen)); fallback.append(failed)
    return np.array(selected), np.array(fallback)
