import itertools
import os
import re
import warnings
from typing import Sequence
from joblib import Memory

import numpy as np
from tifffile import tifffile
# import imageio
from tqdm import tqdm

PROJECTION_FILE_REGEX = "camera ([1-3])/img_([0-9]{1,6})\.tif$"

import pathlib
path = pathlib.Path(__file__).parent.resolve()
cachedir = str(path.parent / "cache")
memory = Memory(cachedir, verbose=0)

def _collect_fnames(
    path: str,
    regex: str = PROJECTION_FILE_REGEX,
):
    if not os.path.exists(path):
        raise IOError(f"The path to {path} does not seem to exist.")

    results, results_filenames = [], []
    regex = re.compile(regex)
    print(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            full_filename = os.path.join(root, file)
            match = regex.search(full_filename)
            if match is not None:
                groups = match.groups()

                tmp_result = [None] * len(groups)
                for i, group in enumerate(groups):
                    if not group.isdigit():
                        raise Exception(
                            "The regex captured a non-digit, cannot proceed."
                        )

                    tmp_result[i] = int(group)

                results.append(tmp_result)
                results_filenames.append(full_filename)

    return results, results_filenames


def load(
    path,
    time_range: range = None,
    regex: str = PROJECTION_FILE_REGEX,
    dtype=np.float32,
    verbose: bool = True,
    cameras: Sequence = (1, 2, 3),
    detector_rows: Sequence = None,
):
    """Load a stack of data from disk using a pattern."""
    results, results_filenames = _collect_fnames(path, regex)
    # Check the results for continuity in the subsequences range
    lists = list(zip(*results))

    first_cam = cameras[0]
    amount_matches = lists[0].count(first_cam)
    assert amount_matches > 0
    for cam in cameras:
        assert amount_matches == lists[0].count(cam)

    if time_range is None:
        time_range = range(np.min(lists[1]), np.max(lists[1]) + 1)

    # load into results
    im_shape = list(tifffile.imread(results_filenames[0]).shape)
    if detector_rows is None:
        rows = slice(0, im_shape[0])
    else:
        rows = slice(detector_rows.start, detector_rows.stop)
    ims = np.zeros((len(time_range), len(cameras), *im_shape), dtype=dtype)

    # make a dictionary with in the first key the detector, and second key
    # the timestep
    nested_dict = {i: {} for i in cameras}
    for i, ((cam_id, t), filename) in enumerate(
        zip(results, results_filenames)):
        if cam_id in cameras:
            nested_dict[cam_id][t] = filename

    # check if wanted timesteps are in the dict, and load them
    arr = []
    for t_i, t in enumerate(tqdm(time_range)) if verbose else enumerate(
        time_range):
        for d_i, d in enumerate(nested_dict.keys()):
            if not t in nested_dict[d]:
                raise FileNotFoundError(
                    f"Could not find timestep {t} from "
                    f"detector {d} in directory {path}."
                )
            ims[t_i, d_i, rows] = \
            tifffile.imread(nested_dict[d][t], maxworkers=1)[
                detector_rows
            ]

    return np.ascontiguousarray(ims)


@memory.cache
def reference_via_mode(data, qu=1, deg=20, nr_bins=100, nr_linspace=1000):
    """For some reason `numpy` is faster (about 2x)."""
    xp = np
    data = xp.asarray(data)

    if qu != 0:
        # edges are unreachable and are estimated by mean
        ref_mode = xp.mean(data, axis=0)
    else:
        ref_mode = xp.zeros(data.shape[1:])

    cams = range(data.shape[1])
    rows = range(qu, ref_mode.shape[-2] - qu)
    cols = range(ref_mode.shape[-1])
    # rows = range(200, 300)  # handy for quick visualization
    # cols = range(200, 300)
    total = len(rows) * len(cols) * len(cams)
    for cam, row, col in tqdm(itertools.product(cams, rows, cols),
                              total=total):
        pixeldata = data[:, cam, row - qu:row + qu + 1, col].flatten()
        bin_values, bin_edges = xp.histogram(
            pixeldata,
            bins=nr_bins,
            # range=(5500, 5600),
            density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:100] + bin_width / 2

        if xp != np:
            # assuming its cupy
            import cupy as cp
            assert xp == cp
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = xp.polyfit
                pol = fit(bin_centers, bin_values, deg=deg)
                xx = xp.linspace(xp.min(bin_centers),
                                 xp.max(bin_centers),
                                 nr_linspace)
                yy = xp.polyval(pol, xx)
        else:
            fit = np.polynomial.Polynomial.fit
            pol = fit(bin_centers, bin_values, deg=deg)
            xx, yy = pol.linspace(nr_linspace)

        mode_distr = xx[xp.argmax(yy)]
        ref_mode[cam, row, col] = mode_distr

        # import matplotlib.pyplot as plt
        # plt.figure(2)
        # plt.cla()
        # plt.hist(pixeldata, bins=nr_bins,
        #          # range=(5500, 7500),
        #          density=True)
        # plt.axvline(x=mode_distr, color='r', linewidth=1)
        # plt.pause(1.)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(ref_mode[0])
    # plt.show()
    return ref_mode


def _isfinite(a):
    m = a.min()
    M = a.max()
    assert np.isfinite(m)
    assert np.isfinite(M)


def _apply_darkfields(dark, meas):
    dark = np.median(dark, axis=0)
    dark = np.expand_dims(dark, 0)
    dark = np.clip(dark, 0.0, None, out=dark)

    meas[:] -= dark
    np.clip(meas, 0.0, None, out=meas)
    _isfinite(meas)
    return


def compute_bed_density(empty, ref, L: float, nr_bins=1000) -> float:
    # not the most efficient, but otherwise duplicated code
    # computing log(ref/meas) / log(ref)
    # should be normalized between 0 and 1
    np.clip(empty, 1.0, None, out=empty)
    bed = np.log(empty / ref, where=ref != 0)
    _isfinite(bed)

    np.clip(bed, 0.0, None, out=bed)

    sum = 0.0
    for i in range(bed.shape[0]):
        counts, bins = np.histogram(bed[i].flatten(), bins=nr_bins)
        counts[:100] = 0.0
        max_value = bins[np.argmax(counts)]  # corresponds to inner diam
        print(f"Proj {i}: mode of column statistic: ", max_value)
        print(f"Proj {i}: density approx.: ", 1 / L * max_value)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(bed.flatten(), bins=1000)
        # plt.show()
        sum += (1 / L) * max_value

    avg = sum / bed.shape[0]  # average density over nr. projs/cams
    print("Average density: ", avg)
    return avg


def preprocess(
    meas,
    ref=None,
    scaling_factor=1.0,
    dtype=np.float32,
    ref_lower_density=False,
):
    """

    :param meas: Projection images (unreferenced)
    :param dark: Darks are always averaged, then subtracted from projs, refs
    :param ref: Refs are always clipped from 0
    :param dtype:
    :param ref_lower_density:
    :return:
    """
    if ref is not None:
        if ref_lower_density:
            # we want to measure lower density, so need to multiply by -1
            # -(log(meas) - log(ref)) = log(ref/meas)
            np.divide(ref, meas, out=meas, where=meas != 0)
            _isfinite(ref)
        else:
            np.divide(meas, ref, out=meas, where=ref != 0)
            _isfinite(meas)

        np.clip(meas, 1.0, None, out=meas)

    np.log(meas, out=meas)
    np.multiply(meas, scaling_factor, out=meas)
    _isfinite(meas)
    return meas.astype(dtype)


def projection_numbers(path, cam=None) -> list:
    """Returns projection numbers of camera `cam`."""
    if cam is None:
        cam = 1  # TODO: check if results are consistent amongst cams?

    results, results_fnames = _collect_fnames(path)
    return sorted([r[1] for r in results if r[0] == cam])


def load_first_projection_number(path) -> int:
    results, results_fnames = _collect_fnames(path)
    # Check the results for continuity in the subsequences range
    lists = list(zip(*results))
    return sorted(lists[1])[0]
