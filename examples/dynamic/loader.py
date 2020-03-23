import os
import re

import numpy as np
from tifffile import tifffile

# TODO: Not here
# DARK_DIR = "pre_proc_Dark"
# BRIGHT_DIR = "pre_proc_Empty_FOV2"
# FULLS_DIR = "pre_proc_Full_FOV2"

CAMERA_DIR_REGEX = "camera ([1-3]{1})"
PROJECTION_FILE_GLOB_PATTERN = "camera ?/img_*.tif"
PROJECTION_FILE_REGEX = 'camera ([1-3]{1})/img_([0-9]{1,6})\.tif$'


def load(path, time_range: range = None, regex=PROJECTION_FILE_REGEX, dtype=np.float32):
    """Load a stack of data from disk using a pattern."""
    if not os.path.exists(path):
        raise IOError("The path to the root directory not seem to exist.")

    results, results_filenames = [], []

    regex = re.compile(regex)
    for root, dirs, files in os.walk(path):
        for file in files:
            full_filename = os.path.join(root, file)
            match = regex.search(full_filename)
            if match is not None:
                groups = match.groups()

                tmp_result = [None] * len(groups)
                for i, group in enumerate(groups):
                    if not group.isdigit():
                        raise Exception("The regex captured a non-digit, cannot proceed.")

                    tmp_result[i] = int(group)

                results.append(tmp_result)
                results_filenames.append(full_filename)

    # Check the results for continuity in the subsequences range
    lists = list(zip(*results))
    amount_matches = lists[0].count(1)
    assert amount_matches == lists[0].count(2) == lists[0].count(3)
    assert amount_matches > 0

    if time_range is None:
        time_range = range(np.min(lists[1]), np.max(lists[1]) + 1)

    # load into results
    im_shape = tifffile.imread(results_filenames[0]).shape
    ims = np.empty((len(time_range), 3, *im_shape), dtype=dtype)

    # make a dictionary with in the first key the detector, and second key
    # the timestep
    nested_dict = {1: {}, 2: {}, 3: {}}
    for i, ((camera_id, t), filename) in enumerate(zip(results, results_filenames)):
        nested_dict[camera_id][t] = filename

    # check if wanted timesteps are in the dict, and load them
    for t in time_range:
        for d in nested_dict.keys():
            if not t in nested_dict[d]:
                raise FileNotFoundError(f"Could not find timestep {t} from "
                                        f"detector {d} in directory {path}.")

            ims[t - time_range.start, d - 1, ...] = tifffile.imread(nested_dict[d][t])

    return np.ascontiguousarray(ims)


    # # compute gas fraction (according to Evert)
    # np.divide(Imeas, Ifull, out=Imeas)
    # np.log(Imeas, out=Imeas)
    # np.divide(Ibright, Ifull, out=Ibright)
    # np.log(Ibright, out=Ibright)

# TODO(Adriaan): try median reference

def load_referenced_projs_from_fulls(proj_path, full_path, t_range=None):
    # Full column as reference
    Iref = np.mean(load(full_path), axis=0)

    Imeas = load(proj_path, t_range)
    np.divide(Imeas, Iref, out=Imeas)
    np.clip(Imeas, 1.0, None, out=Imeas)
    np.log(Imeas, out=Imeas)
    return Imeas.astype(np.float32)


def load_referenced_projs_from_time_average(proj_path, t_range):
    Imeas = load(proj_path, t_range)

    # Creating Iref because fulls/brights are not well aligned
    # idea, use the first projection image (or a bunch of averages)
    # as a column reference, then set the sign of the bubbles positively
    Iref = load(PROJECTION_FILE_REGEX, proj_path, range(t_range[0], t_range[0]+100))
    Iref = np.min(Iref, axis=0)
    Iref = np.clip(Iref, 0, None)

    np.divide(Imeas, Iref, out=Imeas)
    np.clip(Imeas, 1.0, None, out=Imeas)
    np.log(Imeas, out=Imeas)
    return np.ascontiguousarray(Imeas.astype(np.float32))


