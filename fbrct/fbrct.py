import numpy as np


def column_mask_2d(shape, r=None, center=None):
    """
    :param shape: tuple
    :param r: Radius leave to None to compute automatically
    :param center: leave to None to compute automatically
    :return:
    """
    assert len(shape) == 2
    s1, s2 = shape

    if center is None:
        cx, cy = s1 // 2, s2 // 2
    else:
        cx, cy = center

    if r is None:
        r = np.min([cx, s1 - cx, cy, s2 - cy])

    def dist(x1, y1, x2, y2):
        return (x1 - x2) ** 2 + (y1 - y2) ** 2

    v = np.zeros(shape)
    for x in range(cx - r, cx + r):
        for y in range(cy - r, cy + r):
            if dist(cx, cy, x, y) <= r * r:
                v[x][y] = 1

    return v


def column_mask(shape, radius=None, center=None):
    """
    :param shape: tuple
    :param radius: leave to None to compute automatically
    :param center: leave to None to compute automatically
    :return:
    """
    assert len(shape) == 3
    s1, s2, s3 = shape

    col = np.empty(shape)
    col[..., 0] = column_mask_2d((s1, s2), radius, center)
    for k in range(1, s3):
        col[..., k] = col[..., 0]

    return col