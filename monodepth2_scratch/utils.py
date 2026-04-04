"""
Utility functions for Monodepth2.
"""

import os


def readlines(filename):
    """Read all lines from a text file, stripping whitespace."""
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def sec_to_hm_str(t):
    """Convert time in seconds to h:mm:ss string."""
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    h = t // 60
    return f"{h}:{m:02d}:{s:02d}"


def normalize_image(x):
    """Rescale image tensor to [0, 1] for visualisation."""
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d
