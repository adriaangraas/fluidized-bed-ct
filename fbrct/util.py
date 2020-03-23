import h5py
import odl
import numpy as np


def load_dataset_matlab(fname):
    f = h5py.File(fname, 'r')
    p, pref = f['p_aligned_T'], f['pref_aligned_T']

    # We want rows to be quickly accessible (row-major) so they have to be the last dim.
    assert pref.shape[1] == 1548 and p.shape[2] == 1548
    assert pref.shape[2] == 1524 and p.shape[3] == 1524

    return p, pref, p.shape

def uniform_angle_partition(dim=2, offset=0.0):
    """
    Return basically an array of three angles ODL-style :)
    :return:
    """
    if not (-2 * np.pi <= offset <= 2 * np.pi):
        raise ValueError()

    if dim == 2:
        apart = odl.uniform_partition(
            min_pt=0+offset, max_pt=2 * np.pi - (2 * np.pi / 3) + offset,
            shape=3,
            nodes_on_bdry=True)
    else:
        raise ValueError()

    return apart


def detector_partition_2d(recon_width_length, det_pixel_width):
    """
    Return a detector partition

    :param dim:
    :param recon_det_size:
    :param recon_det_shape:
    :return:
    """
    detector_size = recon_width_length * det_pixel_width
    # if recon_width_length is not None:
    #     detector_size = recon_width_length * det_pixel_width
    # else:
    #     detector_size = DETECTOR_SIZE[0]

    return odl.uniform_partition(
        -detector_size / 2,
        detector_size / 2,
        recon_width_length)


def detector_partition_3d(recon_width_length, recon_height_length,
                          det_pixel_width, det_pixel_height):
    """
    Return a limited-height, limited-width detector partition that is meant to make
    a partial reconstruction.
    """
    recon_det_size = np.array([
        recon_width_length * det_pixel_width,
        recon_height_length * det_pixel_height,
    ])

    recon_det_shape = [
        recon_width_length,
        recon_height_length,
    ]

    return odl.uniform_partition(
        min_pt=-recon_det_size / 2,
        max_pt=recon_det_size / 2,
        shape=recon_det_shape)


def plot_sino_range(p, start, end):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(p[0, 0, start:end, :])
    plt.colorbar()
    plt.show()


def plot_sino(sino, pause=None, vmin=None, vmax=None):
    import matplotlib.pyplot as plt

    if sino.shape[0] == 2:
        _, (ax1, ax2) = plt.subplots(1, 2, num='1')
        ax1.imshow(sino[0, ...].T, vmin=vmin, vmax=vmax)
        ax2.imshow(sino[1, ...].T, vmin=vmin, vmax=vmax)
    elif sino.shape[0] == 3:
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, num='1')
        ax1.imshow(sino[0, ...].T, vmin=vmin, vmax=vmax)
        ax2.imshow(sino[1, ...].T, vmin=vmin, vmax=vmax)
        ax3.imshow(sino[2, ...].T, vmin=vmin, vmax=vmax)
    else:
        raise ValueError("Unexpected shape of sinogram. Received:", sino.shape)

    if pause is None:
        plt.show()
    else:
        plt.pause(pause)


def plot_3d(y, vmin=0., vmax=None, pause=None):
    import matplotlib.pyplot as plt

    n, n, m = y.shape

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, num=2)
    ax1.set_title("$\hat{e}_3$ plane")
    ax1.imshow(y[..., m // 2].T, vmin=vmin, vmax=vmax)
    ax2.set_title("$\hat{e}_2$ plane")
    ax2.imshow(y[:, n // 2, :].T, vmin=vmin, vmax=vmax)
    ax3.set_title("$\hat{e}_1$ plane")
    ax3.imshow(y[n // 2, ...].T, vmin=vmin, vmax=vmax)
    if pause is None:
        plt.show()
    else:
        plt.pause(pause)

    # from mayavi import mlab
    # from skimage.measure import marching_cubes_lewiner
    #
    # verts, faces, normals, values = marching_cubes_lewiner(y, 0.025)
    #
    # mlab.triangular_mesh([vert[0] for vert in verts],
    #                      [vert[1] for vert in verts],
    #                      [vert[2] for vert in verts],
    #                     faces)
    # mlab.show()
