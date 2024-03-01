import numpy as np


def format_sequence(*arr):
    arr = [np.linspace(*s) if isinstance(s, tuple) and len(s) == 3 else s for s in arr]
    return np.unique(np.concatenate(arr)).astype("float").tolist()
