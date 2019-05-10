import odl
import matplotlib.pyplot as plt
import numpy as np
import h5py
import dyntomo
from bubblereactor import reconstruct_filter, plot_sino
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D


from matplotlib import rc

reco_3d = False
reco_screw = False
detectors = range(0,3)

f = h5py.File('/export/scratch1/adriaan/MatlabProjects/DynamicTomography/astra_scripts/fluidized_bed_1_python_2.mat',
              'r')
p, pref = f['p_aligned_T'], f['pref_aligned_T']


# p = f['p_aligned_T'].value
# pref = f['pref_aligned_T'].value

if reco_screw:
    # f_screw = h5py.File('/export/scratch1/adriaan/MatlabProjects/DynamicTomography/astra_scripts/fluidized_bed_1_python_screw.mat',
    #           'r')
    # pscrew = f_screw['BW_T'].value

    g = h5py.File('/export/scratch1/adriaan/MatlabProjects/DynamicTomography/astra_scripts/fluidized_bed_1_python_3.mat',
              'r')
    pscrew = g['pref_aligned_T'].value

    pscrew[0, :, :] = np.roll(pscrew[0, :, :], -57, axis=0)
    pscrew[1, :, :] = np.roll(pscrew[1, :, :], 10, axis=0)
    pscrew[1, :, :] = np.roll(pscrew[1, :, :], 0, axis=1)
    pscrew[2, :, :] = np.roll(pscrew[2, :, :], 5, axis=0)
    pscrew[2, :, :] = np.roll(pscrew[2, :, :], 0, axis=1)


    # pscrew[:, 500:, :] = 0
    # pscrew[:, :250, :] = 0

    # pscrew[1, :, :] = pscrew[0, :, :]
    # pscrew[2, :, :] = pscrew[0, :, :]
    # pscrew[1, :, :] = np.roll(pscrew[0, :, :], 20, axis=0)


T, nr_detectors, det_height, det_width = p.shape

if detectors is True:
    detectors = range(nr_detectors)
else:
    nr_detectors = len(detectors)

# We want rows to be quickly accessible (row-major) so they have to be the last dim.
assert pref.shape[1] == 1548 and p.shape[2] == 1548
assert pref.shape[2] == 1524 and p.shape[3] == 1524

src_rad = 93.7
det_rad = 53.4
det_size = [30, 30]  # [w, h]

if reco_screw:
    delta = 300
    mid = 774/2
    recon_height_ran = range(int(mid - delta), int(mid + delta))
    # recon_height_ran = range(p.shape[2])
else:
    recon_height_ran = range(50, 500)
    # recon_height_ran = range(p.shape[2])

recon_height_len = int(len(recon_height_ran))

recon_det_size = np.array([det_size[1], det_size[0] / pref.shape[1] * recon_height_len])  # [w, h]
recon_det_shape = [det_width, recon_height_len]

if nr_detectors is 2:
    apart = odl.uniform_partition(min_pt=0, max_pt=2 * np.pi - (4 * np.pi / 3), shape=2, nodes_on_bdry=True)
else:
    apart = odl.uniform_partition(min_pt=0, max_pt=2 * np.pi - (2 * np.pi / 3), shape=3, nodes_on_bdry=True)

n = 600  # reconstruction on a nxn grid
L = 10  # -L cm to L cm in the physical space
scale = 14  # median filter reconstruction
t = 15

if reco_3d:
    cone_dpart = odl.uniform_partition(min_pt=-recon_det_size / 2, max_pt=recon_det_size / 2, shape=recon_det_shape)
    cone_geometry = odl.tomo.ConeFlatGeometry(apart, cone_dpart, src_rad, det_rad)

    # scale amount of pixels and physical height according to the selected reconstruction range
    m = max(1, int(np.floor(n / det_width * recon_height_len)))
    H = L * m / n
    reco_space = odl.uniform_discr(min_pt=[-L, -L, -H], max_pt=[L, L, H], shape=[n, n, m])
    op = odl.tomo.RayTransform(reco_space, cone_geometry)

    if not reco_screw:
        pr = -(p[t, :, recon_height_ran, :] - pref[:, recon_height_ran, :]) / scale
    else:
        pr = (pscrew[:, recon_height_ran, :]) / scale
        pr = pr[detectors, :, :]
    pr = np.swapaxes(pr, 1, 2)

    # plot_sino(pr)
else:
    fanflat_dpart = odl.uniform_partition(-det_size[0] / 2, det_size[0] / 2, det_width)
    fanflat_geometry = odl.tomo.FanFlatGeometry(apart, fanflat_dpart, src_rad, det_rad)

    reco_space = odl.uniform_discr(min_pt=[-L, -L], max_pt=[L, L], shape=[n, n])
    op = odl.tomo.RayTransform(reco_space, fanflat_geometry)

    if not reco_screw:
        pr = -(p[0, :, mid, :] - pref[:, mid, :]) / scale
    else:
        pr = -pscrew[:, mid, :] / scale

# reconstruct
x = op.domain.element(np.zeros(op.domain.shape))
reconstruct_medianfilter(op, pr, x, niter=8, bounds=[0, 1])
y = x.data

# recon = op.adjoint
# y = recon(pr).data


if reco_3d:
    try:
        verts, faces, _, _ = measure.marching_cubes_lewiner(y)  # spacing=(0.1, 0.1, 0.1))
        fig = plt.figure(2398472309476)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
        ax.invert_zaxis()
        plt.pause(.1)
    except:
        print("No surfacelevel plot was produced.")
        pass

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, num='892348923')
    ax1.set_title("$\hat{e}_3$ plane")
    im1 = ax1.imshow(y[:, :, int(92)].T)
    ax2.set_title("$\hat{e}_2$ plane")
    ax2.imshow(y[:, int(300), :].T)
    ax3.set_title("$\hat{e}_1$ plane")
    ax3.imshow(y[int(322), :, :].T)
    plt.show()
else:
    plt.figure()
    plt.imshow(y)
    plt.show()
