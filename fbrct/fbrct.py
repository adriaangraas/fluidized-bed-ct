# -*- coding: utf-8 -*-
import numpy as np
import odl


def circle_mask_2d(shape, radius=None, center=None):
    """
    :param shape: tuple
    :param radius: leave to None to compute automatically
    :param center: leave to None to compute automatically
    :return:
    """

    def dist(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def make_circle(tiles, cx, cy, r):
        for x in range(cx - r, cx + r):
            for y in range(cy - r, cy + r):
                if dist(cx, cy, x, y) < r:
                    tiles[x][y] = 1

    s1, s2 = shape

    if center is None:
        cx, cy = int(s1 / 2), int(s2 / 2)
    else:
        cx, cy = center

    if radius is None:
        radius = np.min([
            cx,
            shape[0] - cx,
            cy,
            shape[1] - cy
        ])

    v = np.zeros(shape)
    make_circle(v, cx, cy, radius)
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

    for z in range(s3):
        col[:, :, z] = circle_mask_2d((s1, s2))

    return col


def reconstruct_filter(op: odl.Operator, p, u, niter=10, mask=None,
                       fn_filter=None, clip=(None, None)):
    """
    Iterative reconstruction based on
       R.F. Mudde, Time-resolved X-ray tomography of a fluidized bed

    input
       op - projection matrix or linear projection operator
       p - sinogram
       u - initial iterate
       options.niter - max iterations
               mask - image mask as array with same size as u
               beta - regularization parameter for median filter in [0,1]
               bounds - box constraints on values of u: [min,max]

     output
       u - final iterate
    """
    ones = op.range.element(np.ones(op.range.shape))
    C = odl.MultiplyOperator(np.divide(1, op.adjoint(ones)))

    ones = op.domain.element(np.ones(op.domain.shape))
    R = odl.MultiplyOperator(np.divide(1, op(ones)))

    if mask is not None:
        M = odl.MultiplyOperator(op.domain.element(mask), domain=op.domain, range=op.domain)
    else:
        M = odl.IdentityOperator(op.domain)

    for _ in range(niter):
        # SART update
        # v = u + C(M.adjoint(op.adjoint(R(p - op(M(u))))))

        # u += C*M.T*op.T*R*(p - op*M*u)
        u2 = M(u)

        v = op(u2)
        np.subtract(p, v, out=v.data)
        R(v, out=v)
        w = op.adjoint(v)
        M.adjoint(w, out=w)  # adjoint of a diagonal...
        C(w, out=w)
        np.add(u, w, out=u.data)

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure(29234)
        # plt.imshow(v.data)
        # plt.pause(.2)

        # box constraints
        if clip != (None, None):
            np.clip(u.data, *clip, out=u.data)

        # correction step (median filter)
        if fn_filter is not None:
            u.data[:] = fn_filter(u.data)

        # for example:
        #     u[:] = beta * medians_3d(u) + (1 - beta) * u
        # or
        #     u[:] = beta*medians(u) + (1-beta)*v  % one-step-late update


def medians_2d(u):
    assert u.ndim == 2

    u_padded = np.pad(u, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    w = np.ones_like(u)

    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            patch = u_padded[i:i + 2, j:j + 2]
            w[i, j] = np.median(patch.flat)

    return w


def medians_3d(u):
    assert u.ndim == 3

    # u = padarray(reshape(u,n,n,n),[1,1,1],0,'both')
    u_padded = np.pad(u, ((1, 1), (1, 1), (1, 1)),
                      mode='constant', constant_values=0)
    w = np.ones_like(u)

    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            for k in range(u.shape[2]):
                patch = u_padded[i:i + 2, j:j + 2, k:k + 2]
                w[i, j, k] = np.median(patch.flat)

    return w
