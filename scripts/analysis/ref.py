import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from fbrct import loader
from settings import *

plt.rcParams.update({'figure.raise_window': False})


def _do_stats(pixels: np.ndarray, plot=False):
    # statistics on noise
    pixels = pixels.flatten()
    mu, sigma = scipy.stats.norm.fit(pixels)
    print(f"Mu: {mu}\n Sigma: {sigma}")

    if plot:
        plt.figure()
        plt.hist(pixels, bins=100, density=True)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
        plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
        plt.show()

    return mu, sigma


data_dir_23 = "/export/scratch3/adriaan/evert/data/2021-08-23"
data_dir_24 = "/export/scratch3/adriaan/evert/data/2021-08-24"
projs_dir = f'{data_dir_24}/pre_proc_10mm_23mm_foamballs_horizontal'
# projs_dir = f'{data_dir_24}/pre_proc_3x10mm_foamballs_vertical_wall'
# full_dir = f'{data_dir_23}/pre_proc_Full_30degsec'
full_dir = f'{data_dir_24}/pre_proc_Full_30degsec'
# empty_dir = f'{data_dir_23}/pre_proc_Empty_30degsec'
empty_dir = f'{data_dir_24}/pre_proc_Empty_30degsec'
darks_dir = f'{data_dir_24}/pre_proc_Dark'
ref_dir = f'{data_dir_23}/pre_proc_25Lmin'
ref_dirs = (
    f'{data_dir_23}/pre_proc_19Lmin',
    f'{data_dir_23}/pre_proc_20Lmin',
    f'{data_dir_23}/pre_proc_22Lmin',
    f'{data_dir_23}/pre_proc_25Lmin',
)

cams = (1,)
darks = loader.load(darks_dir, range(1, 1000), cameras=cams)
darksavg = np.average(darks, axis=0)
k = np.reshape(darksavg, (1, *darksavg.shape))

n = 1000
full = loader.load(full_dir, range(1, n), cameras=cams) - darksavg
full_mean = np.average(full, axis=0)

# # std-dev shows strong evidence of high signal deviation outside of the column
# full_dev = np.average(full - full_mean, axis=0)
# # better evidence is obtained for computing the std in vertical regions of the col
# # in the center of the col the std is 107 photons:
# _do_stats((full - full_mean)[:, :, 10:1000, 270:280])
# # on the edge of the col the std is 130 photons:
# _do_stats((full - full_mean)[:, :, 10:1000, 90:100])
# # in the background the std is 166 photons:
# _do_stats((full - full_mean)[:, :, 10:1000, 10:80])
# # this shows that the noise is correlated with the signal (as expected)

empty = loader.load(empty_dir, range(1, n), cameras=cams) - darksavg
empty_mean = np.average(empty, axis=0)
ref = loader.load(ref_dir, range(1, 1200), cameras=cams) - darksavg

refs = []
for r in ref_dirs:
    refs.append(
        loader.load(r, range(1, 1200), cameras=cams) - darksavg
    )

meas_ref = np.log(ref / empty_mean) * 5.0 / 0.51
ref_mean = np.average(meas_ref, axis=0)
# ref_mean = np.median(meas_ref, axis=0)

from numpy.polynomial import Polynomial

# plot polynomials at different heights
# rows = (refs[0].shape[2] // 2,)  # 774
rows = (1000,)
# rows = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100)
col = refs[0].shape[3] // 2
exps = (0, 2, 3)
labels = ('19 L/min', '22 L/min', '25 L/min')
qu = 3
# plt.figure(1)
plt.figure(2)
i = 0
for x in exps:
    for c in (0,):
        # for p1 in range(qu, ref_mean.shape[-2] - qu):
        for p1 in rows:
            p2 = col
            wu = refs[x][:, c, p1 - qu:p1 + qu + 1, p2].flatten()

            hist, bin_edges = np.histogram(wu, bins=100,
                                           # range=(5500, 5600),
                                           density=True)
            bin_width = bin_edges[1] - bin_edges[0]
            bin_centers = bin_edges[:100] + bin_width/2
            pol = Polynomial.fit(bin_centers, hist, deg=20)
            xx, yy = pol.linspace(1000)
            mu = xx[np.argmax(yy)]

            # plt.figure(1)
            # plt.plot(xx, yy, label=p1)

            plt.figure(2)
            # plt.plot(xx, yy, color='black')
            plt.hist(wu, bins=100, range=(5500, 7500), density=True,
                     label=f"{labels[i]} - {int(np.round(mu))} photons")
            i += 1

            plt.axvline(x=mu, color='r', linewidth=1)

            print(mu)

plt.xlabel('Photon count')
plt.ylabel('Probability density')
plt.legend(frameon=False)
plt.tight_layout()
# Save figure
plt.savefig('reference_packing.png', dpi=300,
            bbox_inches='tight')
plt.show()

row = 600
col = 300
qu = 0
for c in (0, 1, 2,):
    # for p1 in range(qu, ref_mean.shape[-2] - qu):
    for p1 in range(row - 1, row + 1):
        # for p2 in range(ref_mean.shape[-1]):
        for p2 in range(col - 1, col + 1):
            wu = meas_ref[:, c, p1 - qu:p1 + qu + 1, p2].flatten()

            hist, bin_edges = np.histogram(wu, bins=100, density=True)
            pol = Polynomial.fit(bin_edges[:100], hist, deg=20)
            xx, yy = pol.linspace()
            mu = xx[np.argmax(yy)]

            plt.figure()
            plt.hist(wu, bins=100, density=True)
            # plt.plot(bin_edges[:100], pol)
            plt.plot(xx, yy)
            plt.show()

            # plt.figure()
            # plt.hist(wu, bins=100, density=True)
            # x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            # plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
            # plt.show()

            ref_mean[0, p1, p2] = mu
            print(mu)

# ref_gt = np.log(ref_mean) * 5.0 / 0.51
# meas_err_ref = meas_ref - ref_gt
# plt.figure()
# plt.imshow(meas_ref[100, 0])
# plt.show()
plt.figure()
plt.imshow(meas_ref[100, 0])

plt.figure()
plt.imshow(ref_mean[0][30:1200])
plt.show()
t = np.sqrt(np.average(meas_ref - ref_mean, axis=0) ** 2)
plt.figure()
plt.imshow(t[0][30:1200])
plt.figure()
plt.imshow(meas_ref[0, 0] - ref_mean[0])
# plt.figure()
# plt.imshow(ref_gt[0])
# plt.show()
# plt.figure()
# plt.imshow(meas_err_ref[100, 0])
# plt.show()

full_mean_x = np.copy(full_mean)
full_mean_x[0, 625] = np.average(full_mean_x[0, 575:675], axis=0)


def _alpha(a, b, c=1.):
    plt.figure()
    plt.plot(np.log(full_mean_x / empty_mean)[0, 625] * 5 / .51,
             label="full bed")
    # plt.plot(np.log(ref_mean * a + b)[0, 625])
    # X = ref_mean * a + b
    # X[:, :, 112:443] *= c
    plt.plot(c * ref_mean[0, 625], label="ref bed")
    plt.legend()
    plt.show()


def _beta(a, b, c):
    plt.figure()
    plt.plot(np.log(full_mean_x / empty_mean)[0, 625], label='full')
    X = np.average(np.log(ref * a + b), axis=0)
    X[:, :, 112:443] *= c
    plt.plot(np.log(X / empty_mean)[0, 625], label='ref')
    plt.legend()
    plt.show()

# data_dir_23 = "/export/scratch3/adriaan/evert/data/2021-08-23"
# data_dir_24 = "/export/scratch3/adriaan/evert/data/2021-08-24"
#
# # scan = FluidizedBedScan(
# #     f'{data_dir_23}/pre_proc_{lmin}Lmin',
# #     liter_per_min=lmin,
# #     # ref_dir=f'{data_dir}/pre_proc_Empty_30degsec',
# #     # ref_ran=range(1, 100),
# #     # ref_dir=f'{data_dir}/pre_proc_{lmin}Lmin',
# #     # ref_ran=range(1000, 1299),
# #     # darks_dir=f'{data_dir}/pre_proc_Darkfield',
# #     ref_dir=f'{data_dir_23}/pre_proc_Full_30degsec',
# #     ref_ran=range(1, 10),
# #     geometry=calib,
# #     geometry_scaling_factor=1. / 1.012447804,
# #     # geometry=f'{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
# #     cameras=(1, 2, 3),
# #     ref_lower_density=False,
# #     # mask_scan=mask_scan()
# #     timeframes=range(1100, 1101),  # nice "bubble" in the top
# #     col_inner_diameter=5.0,
# #     ref_normalization_dir=f'{data_dir_23}/pre_proc_Empty_30degsec',
# #     ref_normalization_ran=range(1, 100),
# # )
#
# # n = 1000
# # cams = (1,)
# # darks_dir = f'{data_dir_24}/pre_proc_Dark'
# # darks = loader.load(darks_dir, range(1, 1000), cameras=cams)
# # darksavg = np.average(darks, axis=0)
# # k = np.reshape(darksavg, (1, *darksavg.shape))
# #
# # full_dir = f'{data_dir_23}/pre_proc_Full_30degsec'
# # empty_dir = f'{data_dir_23}/pre_proc_Empty_30degsec'
# # full = loader.load(full_dir, range(1, n), cameras=cams) - darks
# # empty = loader.load(empty_dir, range(1, n), cameras=cams) - darks
# # empty_mean = np.average(empty, axis=0)
# # full_mean = np.average(full, axis=0)
# #
# # plt.figure()
# # plt.imshow((full_mean - empty_mean)[0])
# # plt.show()
#
# cams = (1,)
#
# projs_dir = f'{data_dir_23}/pre_proc_19Lmin'
# sino = loader.load(projs_dir, range(1, 1200), cameras=cams)
# sino_mean = np.average(sino, axis=0)
#
# row = 395
# t = 300
#
# # plt.figure()
# # plt.imshow(sino[t, 0, row-100:row+100], label='2 bubbles')
# # # plt.imshow(sino[t, 0], label='2 bubbles')
# # plt.axhline(y=100, color='r', linestyle='-')
# # plt.show()
#
# # k = np.log(sino)
# z = np.zeros((sino.shape[-2], sino.shape[-1]))
# # for p1 in range(sino.shape[-2]):
# for p1 in range(row - 100, row + 100):
#     for p2 in range(sino.shape[-1]):
#         mu, sigma = scipy.stats.norm.fit(sino[:, 0, p1, p2])
#         z[p1, p2] = mu
#         print(mu)
#
#         # j = np.histogram(k[:, 0, p1, p2], bins=50)
#         # max = np.argmax(j[1], axis=0)
#         # z[p1, p2] = j[1][max]
#
# plt.figure()
# plt.imshow(z)
# plt.show()
#
# plt.figure()
# plt.plot(np.log(sino[t, 0, row]), label='sinogram with 2 bubbles')
# plt.plot(z[row], label='mode')
# for x in [100]:
#     sino_median = np.median(sino[t - x:t + x], axis=0)
#     plt.plot(np.log(sino_median[0, row]), 'k',
#              label=f'median ref {2 * x} window')
#
# plt.legend()
# plt.show()
