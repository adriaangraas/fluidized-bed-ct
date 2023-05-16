from loader import preprocess
from plotting import CM, COLUMNWIDTH, plt
import numpy as np
import scipy.stats

from fbrct import loader
from settings import *

plt.rcParams.update({'figure.raise_window': False})
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

data_dir_23 = "/export/scratch3/adriaan/evert/data/2021-08-23"
data_dir_24 = "/export/scratch3/adriaan/evert/data/2021-08-24"
darks_dir = f'{data_dir_24}/pre_proc_Dark'
projs_dir = f'{data_dir_24}/pre_proc_10mm_23mm_foamballs_horizontal'
full_dir = f'{data_dir_23}/pre_proc_Full_30degsec'
empty_dir = f'{data_dir_23}/pre_proc_Empty_30degsec'

cams = (1,)
n = 1001
vmax = 3.5

# darks = loader.load(darks_dir, range(1, 1000), cameras=cams)
# darksavg = np.average(darks, axis=0)
# k = np.reshape(darksavg, (1, *darksavg.shape))

sino = loader.load(projs_dir, range(1, n), cameras=cams)
full = loader.load(full_dir, range(1, n), cameras=cams)
empty = loader.load(empty_dir, range(1, n), cameras=cams)

row = 625
sino_mean = np.average(sino, axis=0)
full_mean = np.average(full, axis=0)
empty_mean = np.average(empty, axis=0)

# projs_dir = f'{data_dir_23}/pre_proc_19Lmin'
# cams = (1,)
# ref = loader.load(projs_dir, range(1, 1200), cameras=cams)
# refmedian = np.median(ref, axis=0)[0]
# refmean = np.average(ref, axis=0)[0]

# noise = ref - refmean
# for col in range(120, 440, 20):
#     pixels = noise[:, 0, 200:1000, col].flatten()
#     mu, sigma = scipy.stats.norm.fit(pixels)
#     print(f"{col} Mu: {mu}\n Sigma: {sigma}")
#
# print("ROI: 280-40, 280+40")
# pixels = noise[:, 0, 200:1000, 280-40:280+40].flatten()
# mu, sigma = scipy.stats.norm.fit(pixels)
# print(f"Mu: {mu}\n Sigma: {sigma}")
#
# plt.figure()
# for i in range(0, 1200, 10):
#     plt.plot(ref[i, 0, row], 'k', alpha=0.1)
#
# plt.plot(refmedian[row], 'r', label='Median')
# plt.plot(refmean[row], 'y', label='Mean')
# # plt.plot(refmode[row], 'b', label='Mode')
# plt.legend()
# plt.show()

def x_fmt(x, y):
    return "{:.0f}".format(x * DETECTOR_PIXEL_WIDTH)


def y_fmt(x, y):
    return "{:.0f}".format(x * DETECTOR_PIXEL_HEIGHT
                           + row * DETECTOR_PIXEL_HEIGHT)


# sino = np.log(sino)
# sino_mean = np.log(sino_mean)
# full_mean = np.log(full_mean)

# plt.figure()
# plt.plot(sino[1, 0, row], 'r', label='Signal of 23mm and 10mm bubbles')
# plt.plot(sino_mean[0, row], 'k', label='Groundtruth (1000 averages)')
#
# # plt.plot(full[1, 0, row], label='full column')
# # plt.plot(full_mean[0, row], 'b', label='Reference')
#
# # plt.plot(empty[1, 0, row], label='empty column')
# plt.plot(sino_mean[0, row] - full_mean[0, row], 'k',
#          label='empty column (1000 averages)')
# plt.plot(empty_mean[0, row] - full_mean[0, row], 'k',
#          label='empty column (1000 averages)')
#
# ax = plt.gca()
# ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
# ax.xaxis.set_major_locator(
#     plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH))  # cm
# ax.xaxis.set_minor_locator(
#     plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH / 10))  # cm
# ax.set(xlabel='Width (cm)', ylabel='Photon count')
# ax.label_outer()
# plt.legend()
# plt.show()

fig, axs = plt.subplots(
    nrows=3, ncols=1,
    figsize=(COLUMNWIDTH * CM, 10 * CM))
for ax in axs:
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))  # cm
    ax.yaxis.set_major_locator(plt.MultipleLocator(100))  # cm

    # ax.xaxis.set_major_locator(
    #     plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH))  # cm
    # ax.xaxis.set_minor_locator(
    #     plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH / 10))  # cm
    # ax.yaxis.set_major_locator(
    #     plt.MultipleLocator(1. / DETECTOR_PIXEL_HEIGHT))  # cm
    # ax.yaxis.set_minor_locator(
    #     plt.MultipleLocator(1. / DETECTOR_PIXEL_HEIGHT / 10))  # cm
    # ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
    # ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    # ax.set(xlabel='Width (cm)', ylabel='Height (cm)')
    ax.label_outer()

m, M = 75, 85
meas = np.copy(sino[1, 0, row - m:row + M])
axs[0].set_title('$I$')
axs[0].imshow(meas,
              label='2 bubbles',
              cmap='gray'
              )
# axs[0].axhline(y=100, color='r', linestyle='-')

# axs[1].set_title('Groundtruth (1000 averages)')
# axs[1].imshow(sino_mean[0, row - 100:row + 100])
# axs[1].axhline(y=100, color='k', linestyle='-')

ref = np.copy(full_mean[0, row - m:row + M])
axs[1].set_title("$I_{full}$")
axs[1].imshow(ref,
              cmap='gray'
              )
# axs[1].axhline(y=100, color='b', linestyle='-')

refempty = empty_mean[0, row - m:row + M]
axs[2].set_title("$I_{empty}$")
axs[2].imshow(refempty,
              cmap='gray'
              )

# y = preprocess(
#     sino[1, 0, row - 100:row + 100],
#     full_mean[0, row - 100:row + 100],
#     ref_lower_density=False,
#     scaling_factor=1.)
#
# axs[3].set_title("$y$")
# axs[3].imshow(y, cmap='gray_r')

plt.tight_layout(w_pad=.0, h_pad=0.)
plt.savefig('raw_projs.pdf')
plt.show()
plt.close()


LINEWIDTH = .5
fig, axs = plt.subplots(nrows=2, ncols=1,
                        figsize=(COLUMNWIDTH * CM, 8 * CM),
                        gridspec_kw={
                           'width_ratios': [1],
                           'height_ratios': [5, 3]})
"""
First axis
"""
ax = axs[0]
# ax.plot(sino[1, 0, row], 'r', label='$i$', linewidth=lw)
ax.fill_between(
    x=np.array(range(sino[:, 0, row].shape[-1])),
    y1=sino[:, 0, row].min(axis=0),
    y2=sino[:, 0, row].max(axis=0),
    color='r',
    alpha=.1,
    linewidth=0,
)
stddev = np.sqrt(
    np.sum(
        (sino[:, 0, row] - sino_mean[0, row])**2,
        axis=0
    ) / (sino[:, 0, row].shape[0] - 1)  # Bessel's correction
)
ax.fill_between(
    x=np.array(range(sino[:, 0, row].shape[-1])),
    y1=sino_mean[0, row] - stddev,
    y2=sino_mean[0, row] + stddev,
    color='r',
    linewidth=0,
    label='$I$',
    alpha=1.)
ax.plot(
    sino_mean[0, row],
    'k',
    linewidth=LINEWIDTH)
ax.label_outer()
# plt.plot(full[1, 0, row], label='full column')
ax.plot(full_mean[0, row], 'b', label="$I_{full}$", linewidth=LINEWIDTH)
# plt.plot(empty[1, 0, row], label='empty column')
# plt.plot(empty_mean[0, row], 'k', label='empty column (1000 averages)')
ax.xaxis.set_major_locator(plt.MultipleLocator(100))  # pixels
ax.set(ylabel='Photon count')
ax.legend(frameon=False, loc='lower left')

"""
Second axis
"""
ax = axs[1]
y = preprocess(
    np.copy(sino[1, 0, row]),
    full_mean[0, row],
    ref_lower_density=False,
    scaling_factor=1.)
ys = np.array([preprocess(
    np.copy(sino[i, 0, row]),
    full_mean[0, row],
    ref_lower_density=False,
    scaling_factor=1.) for i in range(sino.shape[0])])
y_mean = np.average(ys, axis=0)
# ax.plot(y, label="$y$", linewidth=lw)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ax.fill_between(
    x=np.array(range(ys.shape[-1])),
    y1=ys.min(axis=0),
    y2=ys.max(axis=0),
    color=colors[0],
    linewidth=0,
    alpha=.1
)
variance = (np.sum(
        (ys - y_mean)**2,
        axis=0
    ) / (ys.shape[0] - 1)) # Bessel's correction
stddev = np.sqrt(variance)
ax.fill_between(
    x=np.array(range(sino[:, 0, row].shape[-1])),
    y1=y_mean - stddev,
    y2=y_mean + stddev,
    color=colors[0],
    linewidth=0,
    alpha=1.,
    label='$y$'
)
ax.plot(y_mean, 'k', linewidth=LINEWIDTH)
ax.set(ylabel='Gas fraction projection')
ax.legend(frameon=False, loc='upper left')
ax.label_outer()

ax = axs[1].twinx()
snr = 10 * np.log10(y_mean**2/variance)
ax.plot(snr, 'k--', linewidth=LINEWIDTH, label='SNR')
ax.set(ylabel='SNR (dB)')
ax.legend(frameon=False, loc='upper right')

fig.align_ylabels(axs[:])
plt.tight_layout(w_pad=.0, h_pad=0.5)
plt.savefig('noise.pdf')
plt.show()
plt.pause(5.)