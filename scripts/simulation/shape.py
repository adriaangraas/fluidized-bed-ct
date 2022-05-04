import itertools

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')
import matplotlib.ticker as tick

plt.rcParams.update({'figure.raise_window': False})
from matplotlib.colors import ListedColormap

from astrapy import *
from fbrct.reco import Reconstruction

import settings
from fbrct import column_mask


def geoms_perfect():
    geom_t0 = Geometry(
        tube_pos=[-settings.SOURCE_RADIUS, 0., 0.],
        det_pos=[settings.DETECTOR_RADIUS, 0., 0.],
        u_unit=[0, 1, 0],
        v_unit=[0, 0, 1],
        detector=Detector(
            settings.DETECTOR_ROWS,
            settings.DETECTOR_COLS,
            settings.DETECTOR_PIXEL_WIDTH,
            settings.DETECTOR_PIXEL_HEIGHT))

    rotate_inplace(geom_t0, yaw=.3 * 2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    return [rotate(geom_t0, yaw=a) for a in angles]


# can't have anisotropic voxels yet
iso_voxel_size = min(settings.APPROX_VOXEL_WIDTH, settings.APPROX_VOXEL_HEIGHT)
X = 5000
voxels_x, voxels_z, w, h = Reconstruction.compute_volume_dimensions(
    iso_voxel_size,
    iso_voxel_size,
    {'rows': settings.DETECTOR_ROWS, 'cols': settings.DETECTOR_COLS},
    nr_voxels_x=X)

vectors = settings.cate_to_astra(
    settings.calib_dir + "/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy")
geoms_real = settings.astra_to_rayve(vectors)
min_constraint = 0.0
max_constraint = 1.0


def write_bubble(vol, center: tuple, radius: int, value=1., grad=False,
                 gradpart=1.):
    for i, j, k in itertools.product(
        *[range(center[i] - radius, center[i] + radius) for i in
          range(vol.ndim)]):
        l, m, n = i - center[0], j - center[1], k - center[2]
        dist_squared = l * l + m * m + n * n
        radius_squared = radius * radius
        if dist_squared <= radius_squared:
            if grad:
                if gradpart != 1.:
                    if np.sqrt(dist_squared) >= gradpart * radius:
                        vol[i, j, k] = value * 1 / (1 - gradpart) * (
                                1 - np.sqrt(dist_squared) / radius)
                else:
                    f = (1 - dist_squared / radius_squared)
                    vol[i, j, k] = f * value + (1 - f) * vol[i, j, k]
            else:
                vol[i, j, k] = value


def write_noise(vol, sigma, clip=False):
    vol += np.random.normal(scale=sigma, size=vol.shape)
    if clip:
        vol[...] = np.clip(vol, 0., None)


def _vol_params(vxls, xmin, xmax, geoms):
    return vol_params([vxls, None, vxls],
                      [xmin, None, None],
                      [xmax, None, None],
                      # vox_sz=[0.1, 0.1, 0.1],
                      geometries=geoms)


# # single bubble, full intensity
# geoms = geoms_real
# vxls = 350
# s_rad = int(vxls / 5)
# s_center = np.array([0., 0.])
# prms = _vol_params(vxls, -3, 3, geoms)
# shp, vol_min, vol_max, vox_sz = prms
# vol = np.zeros(shp)
# pos = (vol.shape[0] // 2 + 20, vol.shape[1] // 2 + 20, vol.shape[2] // 2)
# write_bubble(vol, pos,
#              s_rad, 1.)
# nr_iters = 200

# # single bubble, half intensity
# geoms = geoms_real
# vxls = 350
# s_rad = int(vxls / 5)
# s_center = np.array([0., 0.])
# shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# vol = np.zeros(shp)
# write_bubble(vol, (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2),
#              s_rad, .5)
# nr_iters = 500

# single bubble, half intensity, realistic noise
# geoms = geoms_real
# vxls = 350
# s_rad = int(vxls / 5)
# s_center = np.array([0., 0.])
# shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# vol = np.zeros(shp)
# write_bubble(vol, (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2),
#              s_rad, .5)
# write_noise(vol, 0.09)

# # single bubble, grad intensity
# vxls = 350
# geoms = geoms_real
# s_rad = int(vxls / 5)
# s_center = np.array([0., 0.])
# shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# vol = np.zeros(shp)
# write_bubble(vol, (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2),
#              s_rad, .5, grad=False)
# nr_iters = 200

# # single bubble, half grad half homogeneous intensity
# vxls = 350
# geoms = geoms_real
# s_rad = int(vxls / 5)
# s_center = np.array([0., 0.])
# shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# vol = np.zeros(shp)
# write_bubble(vol, (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2),
#              s_rad, .5, halfgrad=True)
# nr_iters = 200

# # single bubble, half grad half homogeneous intensity
# vxls = 350
# geoms = geoms_real
# s_rad = int(vxls / 5)
# s_center = np.array([0., 0.])
# shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# vol = np.zeros(shp)
# write_bubble(vol, (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2),
#              s_rad, .5, grad=True, gradpart=.5)
# nr_iters = 200

# # two bubbles, rotated geometry "half hiding"
# geoms = geoms_real
# vxls=350
# shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# vol = np.zeros(shp)
# [rotate_inplace(g, yaw=2*np.pi/12) for g in geoms]
# s_rad = int(vxls / 8)
# pos = int(vxls / 100 * 35)
# s_center = -np.array([pos - vxls // 2, pos - vxls // 2]) * vox_sz[:2] # center of sphere
# write_bubble(vol, (pos, pos, vol.shape[2] // 2), s_rad, .8)
# pos = int(vxls / 100 * 65)
# write_bubble(vol, (pos, pos, vol.shape[2] // 2), s_rad, .8)
# nr_iters = 200

# # two bubbles, rotated geometry "perfect hiding"
# vxls=350
# geoms = geoms_real
# [rotate_inplace(g, yaw=2*np.pi/12) for g in geoms]
# # s_rad = int(vxls / 8)
# shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# vol = np.zeros(shp)
# s_rad = int(vxls / 8)
# pos = int(vxls / 100 * 35)
# # s_center = -np.array([pos - vxls // 2, 0]) * vox_sz[:2]
# write_bubble(vol, (pos, vxls // 2, vol.shape[2] // 2), s_rad, 1.0)
# s_rad = int(vxls // 12)
# pos = int(vxls / 100 * 65)
# s_center = -np.array([pos - vxls // 2, 0]) * vox_sz[:2]
# write_bubble(vol, (pos, vxls // 2, vol.shape[2] // 2), s_rad, 1.0)
# nr_iters = 200

# # noise
# vxls=350
# geoms = geoms_real
# [rotate_inplace(g, yaw=2*np.pi/12) for g in geoms]
# # s_rad = int(vxls / 8)
# shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# vol = np.zeros(shp)
# s_rad = int(vxls / 8)
# pos = int(vxls / 100 * 35)
# # s_center = -np.array([pos - vxls // 2, 0]) * vox_sz[:2]
# write_bubble(vol, (pos, vxls // 2, vol.shape[2] // 2), s_rad, .9)
# pos = int(vxls / 100 * 65)
# s_center = -np.array([pos - vxls // 2, 0]) * vox_sz[:2]
# write_bubble(vol, (pos, vxls // 2, vol.shape[2] // 2), s_rad, .9)
# single bubble, grad intensity
vxls = 350
geoms = geoms_real
s_rad = int(vxls / 10)
prms = _vol_params(vxls, -3, 3, geoms)
shp, vol_min, vol_max, vox_sz = prms
vol = np.zeros(shp)
s_center = np.array([0., 0.])
pos = (vol.shape[0] // 2 + 20, vol.shape[1] // 2 + 20, vol.shape[2] // 2)
# s_center = (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2)
write_noise(vol, 1 / 0.327 * 0.0882)  # as measured in a scanned volume
write_bubble(vol, pos, s_rad, 1.0, grad=False)
noiselessvol = np.zeros_like(vol)
write_bubble(noiselessvol, pos, s_rad, 1.0, grad=False)
# nr_iters = 60 # for volnoise, after 60 there is overfitting
nr_iters = 200  # for projnoise
# proj_noise_sigma = 0.16274985671  # from projnoise.py
# min_constraint = -(1 / 0.327 * 0.0882)  # allowing negative noise

# forward project
# from copy import deepcopy
# geoms_error = deepcopy(geoms)
# shift(geoms_error[0], [0.5, 0.5, 0.5])
# projs = fp(vol, geoms_error, vol_min, vol_max)

print(prms)
projs = fp(vol, geoms, vol_min, vol_max)
noiselessprojs = np.copy(projs)
noiselessprojsnoiselessvol = fp(noiselessvol, geoms, vol_min, vol_max)
projsnoiselessvol = np.copy(noiselessprojsnoiselessvol)

try:
    if proj_noise_sigma > 0:
        [write_noise(p, proj_noise_sigma) for p in projs]
        [write_noise(p, proj_noise_sigma) for p in projsnoiselessvol]
except NameError:
    pass

proj0 = projs[0]
projsnoiselessvol0 = projsnoiselessvol[0]
noiselessproj0 = noiselessprojs[0]
noiselessprojsnoiselessvol0 = noiselessprojsnoiselessvol[0]

from fbrct.util import plot_projs, plot_projlines

plot_projlines([projsnoiselessvol0,
                noiselessproj0,
                noiselessprojsnoiselessvol0],
               colors=[None, None, 'red'],
               labels=["Camera noise", "Bed", "Clean"],
               pixel_width=settings.DETECTOR_PIXEL_WIDTH
               )

plot_projs(np.asarray(projs),
           settings.DETECTOR_PIXEL_WIDTH,
           settings.DETECTOR_PIXEL_HEIGHT,
           675, 875)

# Note: there are some voxels that are inside the column mask, but cannot be
# seen on all three detectors!
# plt.figure()
# plt.imshow((col_mask * vol_mask == col_mask).astype(np.int)[..., 50])
# plt.show()
# assert np.all(vol_mask * col_mask == col_mask)  # fails

# prevent backprojection into "dead area" (that cannot be seen on all of the
# detectors)
# projs_mask = np.ones_like(projs)
# vol_mask = bp(projs_mask, geoms, vol.shape, vol_min, vol_max)
# 7.5 is a lowerbound intensity in the volume after backprojecting
# vol_mask = (vol_mask > 7.5).astype(np.float)
# vol_mask *= column_mask(shp)

# 5 cm inner radius column = 2.5 cm diameter
vol_mask = column_mask(shp, radius=int(2.5 / vox_sz[0]))
vol_mask2 = vol_mask == 0
mask_vol = np.ma.masked_where(~vol_mask2, vol_mask2)
my_cmap = ListedColormap(['black'])
my_cmap._init()
my_cmap._lut[:-1, -1] = 1.0


def plot_nicely(no, title, im, vmin, vmax, cmap, with_lines=True):
    extent = (-0.5 - (shp[0] // 2),
              (shp[0] // 2) - 0.5,
              -0.5 - (shp[1] // 2),
              (shp[1] // 2) - 0.5)
    plt.figure(no)
    plt.cla()
    plt.title(title)
    im = np.flipud(np.fliplr(np.swapaxes(im, 0, 1)))
    plt.imshow(im,
               vmin=vmin, vmax=vmax,
               cmap=cmap, origin='lower',
               extent=extent, interpolation='none')
    ax = plt.gca()
    # ax.axis('square')
    ax.set_xlim(xmin=-shp[0] // 2, xmax=shp[0] // 2)
    ax.set_ylim(ymin=-shp[1] // 2, ymax=shp[1] // 2)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1. / vox_sz[0]))  # cm
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1. / vox_sz[0] / 10))  # mm
    ax.yaxis.set_major_locator(plt.MultipleLocator(1. / vox_sz[1]))  # cm
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1. / vox_sz[1] / 10))  # mm

    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('$y$ (cm)')

    def x_fmt(x, y):
        return "{:.1f}".format(x * vox_sz[0])  # millis

    def y_fmt(x, y):
        return "{:.1f}".format(x * vox_sz[1])  # millis

    ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    plt.imshow(mask_vol[..., vol.shape[2] // 2], alpha=1, cmap=my_cmap,
               origin='lower', extent=extent, interpolation='none')

    if with_lines:
        for i, (g, ls) in enumerate(zip(geoms, (':', '-.', '--'))):
            # if i == 1:
            if True:
                S = g.tube_position[:2]
                D = g.detector_position[:2]

                ax.axline(S / vox_sz[:2], D / vox_sz[:2],
                          color='red', linewidth=0.5, ls=ls,
                          label=f"Source {i}")

                SD = D - S
                SC = s_center - S  # source-sphere vector

                # unit vector from sphere to source-det line
                CZ = np.inner(SD, SC) / np.inner(SD, SD) * SD - SC
                nCZ = CZ / np.linalg.norm(CZ)

                # outer point on the ball with radius r, being hit by SD
                P = s_center + s_rad * np.array(vox_sz[:2]) * nCZ

                ax.axline(S / vox_sz[:2], P / vox_sz[:2],
                          color='white', linewidth=1.0, ls=ls,
                          label=f"p1")

                P = s_center - s_rad * np.array(vox_sz[:2]) * nCZ
                ax.axline(S / vox_sz[:2], P / vox_sz[:2],
                          color='white', linewidth=1.0, ls=ls,
                          label=f"p2")

        # plt.legend()
    plt.tight_layout()


plot_nicely(
    0,
    "Groundtruth",
    vol[..., vol.shape[2] // 2],
    0, 1.5, 'gray',
    with_lines=False)

gt_error = []
gt_error_increase = []


def fcall(i, x, y_tmp):
    global gt_error
    global gt_error_increase
    global vol
    gt_error += [np.linalg.norm(x.get() - vol)]
    if len(gt_error) > 1:
        gt_error_increase += [gt_error[-1] - gt_error[-2]]
    else:
        gt_error_increase += [0.]

    # x += cp.random.normal(scale=0.01, size=x.shape)

    plt.figure(3)
    plt.cla()
    plt.plot(gt_error)
    plt.pause(.01)

    plt.figure(4)
    plt.cla()
    plt.plot(gt_error_increase)
    plt.pause(.01)

    plot_nicely(1, "Error",
                vol[..., vol.shape[2] // 2] - x[..., x.shape[2] // 2].get(),
                vmin=-1., vmax=1., cmap='RdBu')
    plt.pause(0.01)

    plot_nicely(2, "Reconstruction",
                x[..., x.shape[2] // 2].get(), vmin=max(-.5, min_constraint),
                vmax=1, cmap='viridis')
    plt.pause(0.01)


vol2 = sirt_experimental(projs, geoms, vol.shape, vol_min, vol_max,
                         iters=nr_iters,
                         mask=vol_mask,
                         min_constraint=min_constraint,
                         max_constraint=max_constraint,
                         callback=fcall,
                         x_0=np.ones_like(vol) * 0.5,
                         )

filename = "out_shape_script.npy"

infodict = {"timeframe": nr_iters,
            "name": "Numerical script outcome",
            "reference_dir": None,
            "volume": vol2,
            "algorithm": 'sirt',
            "nr_iters": nr_iters,
            "voxel_size": vox_sz,
            "vol_params": prms,
            "geometry": geoms,
            "gt": vol}

print(f"Saving {filename}...")
# os.makedirs(os.path.dirname(filename), exist_ok=True)
np.save(filename, infodict, allow_pickle=True)

import pyqtgraph as pq

pq.image(np.transpose(vol2))
plt.figure()
plt.pause(1000)
for sl in range(5, vol2.shape[-1]):
    plt.cla()
    plt.imshow(vol2[..., sl], vmin=-0.5, vmax=1.5)
    plt.pause(.101)
plt.close()
