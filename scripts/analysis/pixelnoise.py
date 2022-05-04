import matplotlib.pyplot as plt
import numpy as np
import scipy.stats.distributions

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
        # plt.plot(x, scipy.stats.poisson.pmf(x, int(mu)), 'bo',
        #          ms=8, label='poisson pmf')
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
ref_dir = f'{data_dir_23}/pre_proc_19Lmin'

cams = (1, 2,)
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

meas_ref = np.log(ref / empty_mean) * 5.0 / 0.51
ref_mean = np.average(meas_ref, axis=0)
ref_mean = np.median(meas_ref, axis=0)

row = 200
for p1 in range(row - 100, row + 100):
    for p2 in range(ref_mean.shape[-1]):
        mu, sigma = scipy.stats.norm.fit(meas_ref[:, 0, p1, p2])
        ref_mean[0, p1, p2] = mu
        print(mu)


# ref_gt = np.log(ref_mean) * 5.0 / 0.51
# meas_err_ref = meas_ref - ref_gt
# plt.figure()
# plt.imshow(meas_ref[100, 0])
# plt.show()
plt.figure()
plt.imshow(ref_mean[0])
plt.show()
plt.figure()
plt.imshow(np.average(meas_ref[:, 0] - ref_mean[0], axis=0))
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
    plt.plot(np.log(full_mean_x / empty_mean)[0, 625] * 5 / .51, label="full bed")
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

ref_magic = ref_mean * 1.05
ref_magic /= 5.0 / 0.51
ref_magic = np.exp(ref_magic)
ref_magic *= empty_mean

plt.figure()
plt.imshow(ref_magic[0])
plt.show()
plt.figure()
plt.imshow(full_mean[0])
plt.show()
plt.figure()
plt.imshow(ref_magic[0] - full_mean[0])
plt.show()

# ref using full:
sino = loader.load(projs_dir, range(1, n), cameras=cams) - darksavg

plt.figure()
plt.imshow(np.log(sino[100,0] / ref_magic[0]))
plt.figure()
plt.imshow(np.log(sino[100,0] / full_mean[0]))
plt.figure()
plt.imshow(np.log(sino[100,0] / ref_magic[0]) - np.log(sino[100,0] / full_mean[0]))
plt.show()

# preproc_sino = np.log(sino / ref_magic) * 5.0 / 0.51
preproc_sino = np.log(sino / full_mean) * 5.0 / 0.51
# preproc_sino = np.log(sino) * 5.0 / 0.51
sino_mean = np.average(sino, axis=0)
# meas_mean = np.log(sino_mean / full_mean) * 5.0 / 0.51  # effect of noise on detector
meas_mean = np.average(preproc_sino, axis=0)  # or, effect of bed in volume
noise = preproc_sino - meas_mean

# # ref using the average fluidized bed we just found
# sino = loader.load(projs_dir, range(1, n), cameras=cams) - darksavg
# preproc_sino = np.log(sino) * 5.0 / 0.51 - ref_gt
# sino_mean = np.average(sino, axis=0)
# meas_gt = np.log(sino_mean) * 5.0 / 0.51 - ref_gt
# meas_err = preproc_sino - meas_gt


# 10 mm phantom side cannot be observed (although it is probalby there)
c = 0
avgnoise = np.average(noise[:, c], axis=0)
t = np.sqrt(np.average(np.abs(noise[:, c] - avgnoise[0])**2, axis=0))
plt.figure()
plt.imshow(t)
plt.show()
mu, sigma = _do_stats(noise[:, 0, 641:641 + 1, 168:168 + 1])
plt.figure()
signal = meas_mean[c]
sigma = t
decibel = 10 * np.log10(signal**2/sigma**2)
plt.imshow(decibel)
plt.show()

# big bubble is observed with SNR ~12 dB
mu, sigma = _do_stats(noise[:, 0, 635 - 10:635 + 10, 316 - 10:316 + 10])
signal = np.average(meas_mean[0, 635-10:635+10, 316-10:316+10])
print(f"Signal: {signal}")
print("SNR: ", 10 * np.log10(signal/sigma), " dB")
plt.figure()
avgnoise = np.average(noise[:, 0], axis=0)
t = np.sqrt(np.average(np.abs(noise[:, 0] - avgnoise[0])**2, axis=0))
plt.imshow(t)

# small bubble ~7 dB
mu, sigma = _do_stats(noise[:, 0, 619 - 10:619 + 10, 200 - 10:200 + 10])
signal = np.average(meas_mean[0, 619-10:619+10, 200-10:200+10])
print(f"Signal: {signal}")
print("SNR: ", 10 * np.log10(signal/sigma), " dB")

# background
mu, sigma = _do_stats(noise[:, 0, 716 - 50:716 + 50, 306 - 50:306 + 50])
signal = np.average(meas_mean[0, 716-50:716+50, 306-50:306+50])
# SNR cannot be computed on ~0 valued signal

# special case: empty column noise dB
preproc_sino_2 = np.log(empty / full_mean) * 5.0 / 0.51
meas_mean_2 = np.average(preproc_sino_2, axis=0)  # or, effect of bed in volume
noise_2 = preproc_sino_2 - meas_mean_2
mu, sigma = _do_stats(noise_2[:, 0, 716 - 50:716 + 50, 206 - 50:206 + 50])

plt.figure()
plt.imshow(np.average(noise, axis=0)[0])
plt.show()

plt.figure()
plt.imshow(preproc_sino[0, 0])
plt.show()
plt.figure()
plt.imshow(meas_mean[0])
plt.show()
plt.figure()
plt.imshow(noise[0, 0])
plt.show()

# statistics on noise
pixels = noise[:, 0, 200:1000, 200:400].flatten()
mu, sigma = scipy.stats.norm.fit(pixels)
print(f"Mu: {mu}\n Sigma: {sigma}")
plt.figure()
plt.hist(pixels, bins=100, density=True)
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
plt.show()

# statistics on bed
pixels = (meas_gt)[:, 800:1100, 200:400].flatten()
mu, sigma = scipy.stats.norm.fit(pixels)
print(f"Mu: {mu}\n Sigma: {sigma}")
plt.figure()
plt.hist(pixels, bins=100, density=True)
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
plt.show()