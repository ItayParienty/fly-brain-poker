"""The track centreline, for drawing and for estimating what a spot covers.

The bloons themselves do not use this: they follow the original game's
frame-by-frame tweens (original.py).  This is the same path, as 14 straight
segments, for anything that needs "how much track is within r of (x, y)".
"""
import math

from bloons import rules as R

_SEG = []                      # (x0, y0, dx, dy, length, cumulative start)
_acc = 0.0
for (x0, y0), (x1, y1) in zip(R.WAYPOINTS, R.WAYPOINTS[1:]):
    L = math.hypot(x1 - x0, y1 - y0)
    _SEG.append((x0, y0, (x1 - x0) / L, (y1 - y0) / L, L, _acc)); _acc += L
PATH_LENGTH = _acc


def point_at(s):
    """Pixel position s px along the track."""
    for x0, y0, dx, dy, L, start in _SEG:
        if s <= start + L:
            t = s - start
            return x0 + dx * t, y0 + dy * t
    x0, y0, dx, dy, L, start = _SEG[-1]
    return x0 + dx * L, y0 + dy * L


def distance_to_path(x, y):
    """Distance from a point to the track centreline."""
    best = float("inf")
    for x0, y0, dx, dy, L, _ in _SEG:
        t = max(0.0, min(L, (x - x0) * dx + (y - y0) * dy))
        best = min(best, math.hypot(x - (x0 + dx * t), y - (y0 + dy * t)))
    return best
