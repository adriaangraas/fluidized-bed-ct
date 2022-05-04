import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import scipy.stats

from fbrct import loader
from settings import *

plt.rcParams.update({'figure.raise_window': False})

data_dir_23 = "/export/scratch3/adriaan/evert/data/2021-08-23"
data_dir_24 = "/export/scratch3/adriaan/evert/data/2021-08-24"
darks_dir = f'{data_dir_24}/pre_proc_Dark'
projs_dir = f'{data_dir_24}/pre_proc_10mm_23mm_foamballs_horizontal'
full_dir = f'{data_dir_23}/pre_proc_Full_30degsec'
empty_dir = f'{data_dir_23}/pre_proc_Empty_30degsec'

cams = (1,)
n = 100

darks = loader.load(darks_dir, range(1, 1000), cameras=cams)
darksavg = np.average(darks, axis=0)
k = np.reshape(darksavg, (1, *darksavg.shape))
# plt.figure()
# plt.imshow(k[0])
# plt.show()

sino = loader.load(projs_dir, range(1, n), cameras=cams)
full = loader.load(full_dir, range(1, n), cameras=cams)
empty = loader.load(empty_dir, range(1, n), cameras=cams)

row = 625
sino_mean = np.average(sino, axis=0)
full_mean = np.average(full, axis=0)
empty_mean = np.average(empty, axis=0)

projs_dir = f'{data_dir_23}/pre_proc_19Lmin'
cams = (1,)
ref = loader.load(projs_dir, range(1, 1200), cameras=cams)
refmedian = np.median(ref, axis=0)[0]
refmean = np.average(ref, axis=0)[0]
# refmode = np.zeros((ref.shape[-2], ref.shape[-1]))
# for p1 in range(row - 100, row + 100):
#     for p2 in range(ref.shape[-1]):
#         mu, sigma = scipy.stats.norm.fit(ref[:, 0, p1, p2])
#         refmode[p1, p2] = mu
#         print(mu)
#
# plt.figure()
# plt.imshow(refmode)
# plt.show()
# plt.figure()
# axs[3].set_title('Reference 2')
# axs[3].imshow(refmode[row - 100:row + 100])
# axs[3].axhline(y=100, color='b', linestyle='-')
#
# plt.show()

noise = ref - refmean
for col in range(120, 440, 20):
    pixels = noise[:, 0, 200:1000, col].flatten()
    mu, sigma = scipy.stats.norm.fit(pixels)
    print(f"{col} Mu: {mu}\n Sigma: {sigma}")

print("ROI: 280-40, 280+40")
pixels = noise[:, 0, 200:1000, 280-40:280+40].flatten()
mu, sigma = scipy.stats.norm.fit(pixels)
print(f"Mu: {mu}\n Sigma: {sigma}")

plt.figure()
for i in range(0, 1200, 10):
    plt.plot(ref[i, 0, row], 'k', alpha=0.1)

plt.plot(refmedian[row], 'r', label='Median')
plt.plot(refmean[row], 'y', label='Mean')
# plt.plot(refmode[row], 'b', label='Mode')
plt.legend()
plt.show()

def x_fmt(x, y):
    return "{:.0f}".format(x * DETECTOR_PIXEL_WIDTH)


def y_fmt(x, y):
    return "{:.0f}".format(x * DETECTOR_PIXEL_HEIGHT
                           + row * DETECTOR_PIXEL_HEIGHT)


sino = np.log(sino)
sino_mean = np.log(sino_mean)
full_mean = np.log(full_mean)

plt.figure()
plt.plot(sino[1, 0, row], 'r', label='Signal of 23mm and 10mm bubbles')
plt.plot(sino_mean[0, row], 'k', label='Groundtruth (1000 averages)')

# plt.plot(full[1, 0, row], label='full column')
# plt.plot(full_mean[0, row], 'b', label='Reference')

# plt.plot(empty[1, 0, row], label='empty column')
plt.plot(sino_mean[0, row] - full_mean[0, row], 'k',
         label='empty column (1000 averages)')
plt.plot(empty_mean[0, row] - full_mean[0, row], 'k',
         label='empty column (1000 averages)')

ax = plt.gca()
ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
ax.xaxis.set_major_locator(
    plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH))  # cm
ax.xaxis.set_minor_locator(
    plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH / 10))  # cm
ax.set(xlabel='Width (cm)', ylabel='Photon count')
ax.label_outer()
plt.legend()
plt.show()

fig, axs = plt.subplots(3)
for ax in axs:
    ax.xaxis.set_major_locator(
        plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH))  # cm
    ax.xaxis.set_minor_locator(
        plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH / 10))  # cm
    ax.yaxis.set_major_locator(
        plt.MultipleLocator(1. / DETECTOR_PIXEL_HEIGHT))  # cm
    ax.yaxis.set_minor_locator(
        plt.MultipleLocator(1. / DETECTOR_PIXEL_HEIGHT / 10))  # cm
    ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    ax.set(xlabel='Width (cm)', ylabel='Height (cm)')
    ax.label_outer()

axs[0].set_title('Signal')
axs[0].imshow(sino[1, 0, row - 100:row + 100], label='2 bubbles')
axs[0].axhline(y=100, color='r', linestyle='-')

axs[1].set_title('Groundtruth (1000 averages)')
axs[1].imshow(sino_mean[0, row - 100:row + 100])
axs[1].axhline(y=100, color='k', linestyle='-')

axs[2].set_title('Reference')
axs[2].imshow(full_mean[0, row - 100:row + 100])
axs[2].axhline(y=100, color='b', linestyle='-')

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(sino[1, 0, row], 'r', label='Signal of 23mm and 10mm bubbles')
plt.plot(sino_mean[0, row], 'k', label='Groundtruth (1000 averages)')

# plt.plot(full[1, 0, row], label='full column')
plt.plot(full_mean[0, row], 'b', label='Reference')

# plt.plot(empty[1, 0, row], label='empty column')
# plt.plot(empty_mean[0, row], 'k', label='empty column (1000 averages)')

ax = plt.gca()
ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
ax.xaxis.set_major_locator(
    plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH))  # cm
ax.xaxis.set_minor_locator(
    plt.MultipleLocator(1. / DETECTOR_PIXEL_WIDTH / 10))  # cm
ax.set(xlabel='Width (cm)', ylabel='Photon count')
ax.label_outer()
plt.legend()
plt.show()

# sino_err = np.abs((sino - sino_mean)[:, 0, 200:1000, 200:400])
#
# # plt.figure()
# # plt.imshow(sino_err[100])
# # plt.show()
#
# # plt.figure()
# avgs = []
# for i in range(300):
# # for i in range(sino.shape[0]):
#     avg = np.abs(np.average(sino[:i+1, 0, 200:1000, 200:400], axis=0)
#                  - sino_mean[0, 200:1000, 200:400])
#     # plt.clf()
#     # ax = plt.gca()
#     # ax.imshow(avg)
#     avgs.append(np.average(avg.flatten()))
#     # plt.pause(.0001)
#     print(i)
#
# # plt.show()
#
# sino = sino - k
# sino_mean = np.average(sino, axis=0)
# sino_err = np.abs((sino - sino_mean)[:, 0, 200:1000, 200:400])
# avgs2 = []
# for i in range(300):
#     avg = np.abs(np.average(sino[:i + 1, 0, 200:1000, 200:400], axis=0)
#                  - sino_mean[0, 200:1000, 200:400])
#     avgs2.append(np.average(avg.flatten()))
#     print(i)
#
# plt.figure()
# plt.plot(avgs)
# plt.plot(avgs2)
# plt.show()
#
# plt.figure()
# plt.semilogy(avgs)
# plt.semilogy(avgs2)
# plt.show()
