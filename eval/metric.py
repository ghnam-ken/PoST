import numpy as np
import math

def spatial_accuracy(ps1, ps2, thresh):
    ''' 
        Args) ps1, ps2 : normalized point sets
        Retern) acc: spatial accuracy
    '''
    assert len(ps1) == len(ps2), \
        f"length of given point sets are differenct: len(ps1)={len(ps1)}, len(ps2)={len(ps2)}"

    dists = (ps1 - ps2) ** 2
    dists = np.sum(dists, axis=-1)
    dists = np.sqrt(dists)

    acc = np.mean(dists <= thresh)
    return acc

def temporal_accuracy(ps1, ps2, prev_ps1, prev_ps2, thresh):
    ''' 
        Args) ps1, ps2 : normalized point sets
        Retern) acc: temporal accuracy
    '''
    assert len(ps1) == len(ps2), \
            f"length of given point sets are differenct: len(ps1)={len(ps1)}, len(ps2)={len(ps2)}"
    assert len(prev_ps1) == len(prev_ps2), \
            f"length of given point sets are differenct: len(prev_ps1)={len(prev_ps1)}, len(prev_ps2)={len(prev_ps2)}"

    dists_prev = ps1 - ps2
    dists_next = prev_ps1 - prev_ps2

    diffs = (dists_prev - dists_next) ** 2
    diffs = np.sum(diffs, axis=-1)
    diffs = np.sqrt(diffs)

    acc = np.mean(diffs <= thresh, axis=-1)
    acc = np.mean(acc)
    return acc
