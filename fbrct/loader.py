import os
import re
from typing import Sequence

import numpy as np
# from tifffile import tifffile
import imageio
from tqdm import tqdm

PROJECTION_FILE_REGEX = 'camera ([1-3])/img_([0-9]{1,6})\.tif$'


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
                            "The regex captured a non-digit, cannot proceed.")

                    tmp_result[i] = int(group)

                results.append(tmp_result)
                results_filenames.append(full_filename)

    return results, results_filenames


def load(path,
         time_range: range = None,
         regex: str = PROJECTION_FILE_REGEX,
         dtype=np.float32,
         verbose: bool = True,
         cameras: Sequence = (1, 2, 3)):
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
    im_shape = imageio.imread(results_filenames[0]).shape
    ims = np.empty((len(time_range), len(cameras), *im_shape), dtype=dtype)

    # make a dictionary with in the first key the detector, and second key
    # the timestep
    nested_dict = {i: {} for i in cameras}
    for i, ((cam_id, t), filename) in enumerate(
        zip(results, results_filenames)):
        if cam_id in cameras:
            nested_dict[cam_id][t] = filename

    # check if wanted timesteps are in the dict, and load them
    for t_i, t in enumerate(tqdm(time_range)) if verbose else enumerate(
        time_range):
        for d_i, d in enumerate(nested_dict.keys()):
            if not t in nested_dict[d]:
                raise FileNotFoundError(f"Could not find timestep {t} from "
                                        f"detector {d} in directory {path}.")

            ims[t_i, d_i, ...] = imageio.imread(nested_dict[d][t])

    return np.ascontiguousarray(ims)


def preprocess(meas, dark=None, ref=None, dtype=np.float32,
               ref_lower_density=False, ref_mode='static',
               ref_normalization=None):
    """

    :param meas: Projection images (unreferenced)
    :param dark: Darks are always averaged, then subtracted from projs, refs
    :param ref: Refs are always clipped from 0, and taken mediam
    :param dtype:
    :param ref_lower_density:
    :return:
    """
    def _isfinite(a):
        m = a.min()
        M = a.max()
        assert np.isfinite(m)
        assert np.isfinite(M)

    if dark is not None:
        dark = np.median(dark, axis=0)
        dark = np.expand_dims(dark, 0)
        dark = np.clip(dark, 0., None, out=dark)

        meas[:] -= dark
        np.clip(meas, 0., None, out=meas)
        _isfinite(meas)

        if ref is not None:
            ref[:] -= dark
            np.clip(ref, 0., None, out=ref)
            _isfinite(ref)
        if ref_normalization is not None:
            ref_normalization[:] -= dark
            np.clip(ref_normalization, 0., None, out=ref_normalization)
            _isfinite(ref_normalization)


    # log(meas) - log(ref) = log(meas/ref)
    if ref is not None:
        if ref_mode == 'static':
            ref = np.mean(ref, axis=0)
        elif ref_mode == 'reco':
            assert len(ref) == len(meas)
        else:
            raise NotImplementedError()

        # bed = ref - ref_normalization
        # from scipy.signal import savgol_filter
        # for i, s in enumerate(bed):
        #     bed[i, 50:1000, :] = savgol_filter(s[50:1000, :], 521, 1, axis=-2, mode='mirror')
        #     bed[i, :, :] = savgol_filter(s[:, :], 5, 3, axis=-1, mode='mirror')
        #
        # # import matplotlib.pyplot as plt
        # # plt.figure()
        # # plt.imshow(bed[0])
        # # plt.show()
        # # plt.figure()
        # # plt.imshow(ref[0])
        # # plt.figure()
        # qref = ref_normalization + bed
        # # plt.imshow(qref[0])
        # # plt.show()
        # ref = qref

        if ref_lower_density:
            # we want to measure lower density, so need to multiply by -1
            # -(log(meas) - log(ref)) = log(ref/meas)
            np.divide(ref, meas, out=meas, where=meas != 0)
            _isfinite(ref)
        else:
            np.divide(meas, ref, out=meas, where=ref != 0)
            _isfinite(meas)

        np.clip(meas, 1., None, out=meas)

    np.log(meas, out=meas)
    _isfinite(meas)

    if ref_normalization is not None:
        if np.isscalar(ref_normalization):
            max_value = ref_normalization
        else:
            if ref_mode == 'static':
                ref_normalization = np.mean(ref_normalization, axis=0)
            elif ref_mode == 'reco':
                assert len(ref_normalization) == len(meas)
            else:
                raise NotImplementedError()

            # not the most efficient, but otherwise duplicated code
            # computing log(ref/meas) / log(ref)
            # should be normalized between 0 and 1
            np.clip(ref_normalization, 1., None, out=ref_normalization)
            bed = np.log(ref_normalization / ref, where=ref != 0)
            _isfinite(bed)
            np.clip(bed, 0., None, out=bed)

            for j in range(bed.shape[0]):
                counts, bins = np.histogram(bed[j].flatten(), bins=1000)
                counts[:100] = 0.
                max_value = bins[np.argmax(counts)]
                print(f"Camera {j}: mode of empty column statistic: ", max_value)

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.hist(bed.flatten(), bins=1000)
                # plt.show()

        # bed[bed < 0.01] = 0.
        # bed = -np.log(ref_normalization)
        np.multiply(meas[:, j], 5.0 / max_value, out=meas[:, j])
        np.clip(meas[:, j], 0, 5.0, out=meas[:, j])

        # plt.figure()
        # plt.imshow(meas[0,0])
        # plt.show()
        # meas[:] = np.multiply(bed, 5. / max_value)  # artificially reco bed


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