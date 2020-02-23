# -*- coding: utf-8 -*-
import numpy as np
import odl


def circle_mask_2d(shape, radius=None, center=None, value=1.):
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


def reconstruct_filter(op: odl.Operator, p, u, niter=10, mask=None, mask_val=0.,
                       fn_filter=None, clip=(None, None), iter_start=0, iter_save=50, save_name="recon"):
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
    if mask is not None:
        M = odl.MultiplyOperator(op.domain.element(mask), domain=op.domain, range=op.domain)
    else:
        M = odl.IdentityOperator(op.domain)

    detector_ones = op.range.element(np.ones(op.range.shape))
    backprojected_ones = op.adjoint(detector_ones)
    # a matrix that divides every voxel by the times it is hit by a ray
    # I don't know why but Numpy does not parse the where kwarg when an DiscreteLpElement is passed, bug?
    np.divide(1, backprojected_ones.data, where=backprojected_ones.data != 0, out=backprojected_ones.data)
    C = odl.MultiplyOperator(backprojected_ones)

    # a volume full of 1s
    volume_ones = op.domain.element(np.ones(op.domain.shape))
    projected_ones = op(volume_ones)
    # a matrix that divides every pixel by the times it is hit by a ray
    # I don't know why but Numpy does not parse the `where` kwarg when an DiscreteLpElement is passed, bug?
    np.divide(1, projected_ones.data, where=projected_ones.data != 0, out=projected_ones.data)
    R = odl.MultiplyOperator(projected_ones)



    for i in range(iter_start, niter + iter_start):
        print("Iter:" + str(i))

        # SART update
        # v = u + C(M.adjoint(op.adjoint(R(p - op(M(u))))))

        # u += C*M.T*op.T*R*(p - op*M*u)

        # Apply mask, if specified
        u2 = M(u)

        # Project
        v = op(u2)

        # Store the residual on the detector in v
        np.subtract(p, v, out=v.data)

        # Divide detector by the number of times it is hit by a ray
        R(v, out=v)

        # Backproject
        w = op.adjoint(v)

        # Divide volume voxels by the number of times they are hit by a ray
        C(w, out=w)

        # Apply mask again, if specified
        M(w, out=w)  # adjoint of a diagonal...

        # Update the solution
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

        # save
        if (i+1) % iter_save == 0:
            print(f"Saving {save_name}...")
            np.save(f"{save_name}_{i+1}", u.data)

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
