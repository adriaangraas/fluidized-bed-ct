import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.ticker as tick

plt.rcParams.update({'figure.raise_window': False})
from astrapy import *
from fbrct.reco import Reconstruction
from fbrct import astra_to_astrapy, cate_to_astra, column_mask
from fbrct.util import plot_projs

from scripts import settings


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

    rotate_(geom_t0, yaw=.3 * 2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    return [rotate(geom_t0, yaw=a) for a in angles]


iso_voxel_size = min(settings.APPROX_VOXEL_WIDTH, settings.APPROX_VOXEL_HEIGHT)
X = 5000
voxels_x, voxels_z, w, h = Reconstruction.compute_volume_dimensions(
    iso_voxel_size,
    iso_voxel_size,
    {'rows': settings.DETECTOR_ROWS, 'cols': settings.DETECTOR_COLS},
    nr_voxels_x=X)
det = {'rows': settings.DETECTOR_ROWS,
       'cols': settings.DETECTOR_COLS,
       'pixel_width': settings.DETECTOR_PIXEL_WIDTH,
       'pixel_height': settings.DETECTOR_PIXEL_HEIGHT}
vectors = cate_to_astra(
    settings.calib_dir + "/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    det)
geoms = astra_to_astrapy(vectors, det)
min_constraint = 0.0
max_constraint = 1.0
nr_iters = 2000
proj_noise_sigma = None


def write_bubble(vol, center: tuple, radius: int, value=1., grad=False,
                 gradpart=1.):
    """Draw an artifical bubble in the volume."""

    for i, j, k in itertools.product(
        *[range(center[i] - radius, center[i] + radius) for i in
          range(vol.ndim)]):
        l, m, n = i - center[0], j - center[1], k - center[2]
        dist_squared = l * l + m * m + n * n
        radius_squared = radius * radius
        if dist_squared <= radius_squared:
            if grad:
                if gradpart != 1.:
                    core = (1 - gradpart) * radius
                    if np.sqrt(dist_squared) >= core:
                        # in the gradient part
                        remaining = np.sqrt(dist_squared) - core
                        gradrad = gradpart * radius
                        vol[i, j, k] = value * (1 - remaining / gradrad)
                    else:
                        # in the core part
                        vol[i, j, k] = value
                else:
                    f = 1 - dist_squared / radius_squared
                    vol[i, j, k] = f * value + (1 - f) * vol[i, j, k]
            else:
                vol[i, j, k] = value


def write_vol_noise(vol, sigma, clip=False):
    vol += np.random.normal(scale=sigma, size=vol.shape)
    if clip:
        vol[...] = np.clip(vol, 0., None)


def _vol_params(vxls, xmin, xmax, geoms):
    return vol_params([vxls, None, vxls],
                      [xmin, None, None],
                      [xmax, None, None],
                      geometries=geoms)


# # # single bubble, full intensity
# vxls = 300
# s_rad = int(vxls / 5)
# s_center = np.array([0., 0.])
# prms = _vol_params(vxls, -2.5, 2.5, geoms)
# shp, vol_min, vol_max, vox_sz = prms
# vol = np.zeros(shp)
# pos = (vol.shape[0] // 2 + 20, vol.shape[1] // 2 + 20, vol.shape[2] // 2)
# write_bubble(vol, pos, s_rad, 1.)
# # filename = "out_single_bubble.npy"
# # proj_noise_sigma = 0.16274985671  # from projnoise.py
# # filename = "../../out/out_noise.npy"

# # single bubble, 95% intensity
# # astrapy.rotate_(geoms[0], yaw=.5 * np.pi)
# # astrapy.shift_(geoms[0], [.5, .5, .5])
# vxls = 300
# s_rad = int(vxls / 5)  # = 1. cm exactly for a -2.5, 2.5 volume
# s_center = np.array([0., 0.])
# prms = _vol_params(vxls, -2.5, 2.5, geoms)
# shp, vol_min, vol_max, vox_sz = prms
# vol = np.zeros(shp)
# write_bubble(vol, (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2),
#              s_rad, 0.95)
# filename = "out_95gf.npy"

# single bubble, half intensity, realistic noise
# geoms = geoms_real
# vxls = 350
# s_rad = int(vxls / 5)
# s_center = np.array([0., 0.])
# shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# vol = np.zeros(shp)
# write_bubble(vol, (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2),
#              s_rad, .5)
# write_vol_noise(vol, 0.09)

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

# single bubble, half grad half homogeneous intensity
vxls = 300
s_rad = int(vxls / 5)
s_center = np.array([0., 0.])
prms = _vol_params(vxls, -2.5, 2.5, geoms)
shp, vol_min, vol_max, vox_sz = prms
vol = np.zeros(shp)
write_bubble(vol, (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2),
             s_rad, .95, grad=True, gradpart=.50)
filename = "out_95gf_50grad.npy"

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
# vxls = 300
# [astrapy.rotate_(g, yaw=2*np.pi/12) for g in geoms]
# # s_rad = int(vxls / 8)
# prms = _vol_params(vxls, -3, 3, geoms)
# shp, vol_min, vol_max, vox_sz = prms
# vol = np.zeros(shp)
# s_rad = int(vxls / 8)
# pos = int(vxls / 100 * 35)
# # s_center = -np.array([pos - vxls // 2, 0]) * vox_sz[:2]
# write_bubble(vol, (pos, vxls // 2, vol.shape[2] // 2), s_rad, .95)
# s_rad = int(vxls // 12)
# pos = int(vxls / 100 * 65)
# s_center = -np.array([pos - vxls // 2, 0]) * vox_sz[:2]
# write_bubble(vol, (pos, vxls // 2, vol.shape[2] // 2), s_rad, .95)
# filename = "out_double_bubble.npy"

# # # noise
# # vxls=350
# # geoms = geoms_real
# # [rotate_inplace(g, yaw=2*np.pi/12) for g in geoms]
# # # s_rad = int(vxls / 8)
# # shp, vol_min, vol_max, vox_sz = _vol_params(vxls, -3, 3, geoms)
# # vol = np.zeros(shp)
# # s_rad = int(vxls / 8)
# # pos = int(vxls / 100 * 35)
# # # s_center = -np.array([pos - vxls // 2, 0]) * vox_sz[:2]
# # write_bubble(vol, (pos, vxls // 2, vol.shape[2] // 2), s_rad, .9)
# # pos = int(vxls / 100 * 65)
# # s_center = -np.array([pos - vxls // 2, 0]) * vox_sz[:2]
# # write_bubble(vol, (pos, vxls // 2, vol.shape[2] // 2), s_rad, .9)
# # single bubble, grad intensity
# vxls = 300
# s_rad = int(vxls / 10)
# prms = _vol_params(vxls, -2.5, 2.5, geoms)
# shp, vol_min, vol_max, vox_sz = prms
# vol = np.zeros(shp)
# s_center = np.array([0., 0.])
# pos = (vol.shape[0] // 2 + 20, vol.shape[1] // 2 + 20, vol.shape[2] // 2)
# # s_center = (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2)
# # write_vol_noise(vol, 1 / 0.327 * 0.0882)  # as measured in a scanned volume
# write_bubble(vol, pos, s_rad, 1.)
# noiselessvol = np.zeros_like(vol)
# write_bubble(noiselessvol, pos, s_rad, 1., grad=False)
# # nr_iters = 60 # for volnoise, after 60 there is overfitting
# proj_noise_sigma = 0.16274985671  # from projnoise.py
# # min_constraint = -(1 / 0.327 * 0.0882)  # allowing negative noise
# filename = "out_noise.npy"

projs = fp(vol, geoms, vol_min, vol_max)

if False:  # if True, runs and saves noise statistics
    from copy import deepcopy


    class RunningStats:
        def __init__(self):
            self.n = 0
            self.old_m = 0
            self.new_m = 0
            self.old_s = 0
            self.new_s = 0

        def push(self, x):
            self.n += 1

            if self.n == 1:
                self.old_m = self.new_m = x
                self.old_s = 0
            else:
                self.new_m = self.old_m + (x - self.old_m) / self.n
                self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

                self.old_m = self.new_m
                self.old_s = self.new_s

        def mean(self):
            return self.new_m if self.n else 0.0

        def variance(self):
            return self.new_s / (self.n - 1) if self.n > 1 else 0.0

        def standard_deviation(self):
            return np.sqrt(self.variance())


    projs_gt = deepcopy(projs)
    stats = RunningStats()
    for j in range(100):
        projs = deepcopy(projs_gt)
        if proj_noise_sigma is not None:
            [write_vol_noise(p, proj_noise_sigma) for p in projs]

        # 5 cm inner radius column = 2.5 cm diameter
        vol_mask = column_mask(shp, radius=int(2.5 / vox_sz[0]))
        vol_mask_booleans = vol_mask == 0
        vol2 = sirt_experimental(
            projs, geoms, vol.shape, vol_min, vol_max,
            iters=nr_iters,
            mask=vol_mask,
            min_constraint=min_constraint,
            max_constraint=max_constraint,
        )
        stats.push(vol2)

        if j == 0:
            infodict = {"name": "Numerical script outcome",
                        "reference_dir": None,
                        "volume": vol2,
                        "algorithm": 'sirt',
                        "nr_iters": nr_iters,
                        "voxel_size": vox_sz,
                        "vol_params": prms,
                        "geometry": geoms,
                        "gt": vol}
            print("Saving first instance for plotting...")
            np.save("statistics_firstinstance.npy", infodict,
                    allow_pickle=True)

        infodict = {"name": "Numerical script outcome",
                    "reference_dir": None,
                    "volume": stats.mean(),
                    "algorithm": 'sirt',
                    "nr_iters": nr_iters,
                    "voxel_size": vox_sz,
                    "vol_params": prms,
                    "geometry": geoms,
                    "gt": vol,
                    "stats_n": stats.n}
        print("Saving mean...")
        np.save("statistics_mean.npy", infodict, allow_pickle=True)

        infodict = {"name": "Numerical script outcome",
                    "reference_dir": None,
                    "volume": stats.variance(),
                    "algorithm": 'sirt',
                    "nr_iters": nr_iters,
                    "voxel_size": vox_sz,
                    "vol_params": prms,
                    "geometry": geoms,
                    "gt": vol,
                    "stats_n": stats.n}

        print("Saving variance...")
        np.save("statistics_variance.npy", infodict, allow_pickle=True)

# add noise when this is defined
if proj_noise_sigma is not None:
    [write_vol_noise(p, proj_noise_sigma) for p in projs]

if True:  # if True, plot projections
    from plotting import plt, CM

    plot_projs(
        np.asarray(projs)[:, 675:875],
        # settings.DETECTOR_PIXEL_WIDTH,
        # settings.DETECTOR_PIXEL_HEIGHT,
        figsize=(6 * CM, 7.0 * CM),
    )

# globals required for the `plot_nicely` function.
vol_mask = column_mask(shp, r=int(2.5 / vox_sz[0]))  # 5 cm inner column
vol_mask_booleans = vol_mask == 0
mask_vol = np.ma.masked_where(~vol_mask_booleans, vol_mask_booleans)
my_cmap = ListedColormap(['black'])
my_cmap._init()
my_cmap._lut[:-1, -1] = 1.0


def plot_nicely(no, title, im, vmin, vmax, cmap, with_lines=True,
                with_mask=True):
    """A plotting heliper during reconstruction that draws the source-detector
    axis and mask."""
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
    if with_mask:
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
    0, 1.5,
    'gray',
    with_lines=False)

gt_error = []
gt_error_increase = []


def callback_during_sirt(i, x, y_tmp):
    if i % 100 == 0:
        global gt_error
        global gt_error_increase
        global vol
        gt_error += [np.linalg.norm(x.get() - vol)]
        if len(gt_error) > 1:
            gt_error_increase += [gt_error[-1] - gt_error[-2]]
        else:
            gt_error_increase += [0.]

        plt.figure(3)
        plt.cla()
        plt.plot(gt_error)
        plt.pause(.01)

        plt.figure(4)
        plt.cla()
        plt.plot(gt_error_increase)
        plt.pause(.01)

        plot_nicely(1001, "Error",
                    vol[..., vol.shape[2] // 2] - x[
                        ..., x.shape[2] // 2].get(),
                    vmin=-1., vmax=1., cmap='RdBu')
        plt.pause(0.01)

        plot_nicely(1002, "Reconstruction",
                    x[..., x.shape[2] // 2].get(),
                    vmin=max(-.5, min_constraint),
                    vmax=1, cmap='viridis',
                    with_lines=False, with_mask=False)
        plt.pause(0.01)


vol2 = sirt_experimental(
    projs, geoms, vol.shape, vol_min, vol_max,
    iters=nr_iters,
    mask=vol_mask,
    min_constraint=min_constraint,
    max_constraint=max_constraint,
    callback=callback_during_sirt,
)

infodict = {"name": "Numerical script outcome",
            "reference_dir": None,
            "volume": vol2,
            "algorithm": 'sirt',
            "nr_iters": nr_iters,
            "voxel_size": vox_sz,
            "vol_params": prms,
            "geometry": geoms,
            "gt": vol}
print(f"Saving {filename}...")
np.save(filename, infodict, allow_pickle=True)

if False:  # if True, plot with pyqtgraph
    import pyqtgraph as pq

    pq.image(np.transpose(vol2))
    plt.figure()
    plt.pause(1000)
    for sl in range(5, vol2.shape[-1]):
        plt.cla()
        plt.imshow(vol2[..., sl], vmin=-0.5, vmax=1.5)
        plt.pause(.101)
    plt.close()
