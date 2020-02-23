import re
import os
import numpy as np
from tifffile import tifffile


class Loader:
    """Loader and preprocessor for projection TIFFs"""

    CAMERA_DIR_REGEX = "camera ([1-3]{1})"
    PROJECTION_FILE_GLOB_PATTERN = "camera ?/img_*.tif"
    PROJECTION_FILE_REGEX = 'camera ([1-3]{1})/img_([0-9]{1,6})\.tif$'
    DARK_DIR = "pre_proc_Dark"
    BRIGHT_DIR = "pre_proc_Empty_FOV2"
    FULLS_DIR = "pre_proc_Full_FOV2"

    def __init__(self, root_path):
        if not os.path.exists(root_path):
            raise IOError("The path to the root directory not seem to exist.")

        self.path = root_path

    def _load(self, regex_pattern, path, time_range: range = None):
        """Load a stack of data from disk using a pattern."""
        results, results_filenames = [], []

        regex = re.compile(regex_pattern)
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
        ims = np.empty((len(time_range), 3, *im_shape))

        for i, ((camera_id, t), filename) in enumerate(zip(results, results_filenames)):
            if t in time_range:
                ims[t - time_range.start, camera_id - 1, ...] = tifffile.imread(filename)

        return ims

    def darks(self, t_range: range = None):
        dark_path = os.path.join(self.path, self.DARK_DIR)

        ims = self._load(self.PROJECTION_FILE_REGEX,
                         dark_path,
                         t_range)

        return ims

    def brights(self, t_range: range = None):
        bright_path = os.path.join(self.path, self.BRIGHT_DIR)

        ims = self._load(self.PROJECTION_FILE_REGEX,
                         bright_path,
                         t_range)

        return ims

    def fulls(self, t_range: range = None):
        fulls_path = os.path.join(self.path, self.FULLS_DIR)

        ims = self._load(self.PROJECTION_FILE_REGEX,
                         fulls_path,
                         t_range)

        return ims

    def projs(self, dir, t_range: range = None):
        # Idark = np.mean(self.darks(), axis=0)
        # Ibright = np.mean(self.brights(), axis=0)
        # Ifull = np.mean(self.fulls(), axis=0)

        proj_path = os.path.join(self.path, dir)
        Imeas = self._load(self.PROJECTION_FILE_REGEX, proj_path, t_range)

        # Creating Iref because fulls/brights are not well aligned
        # idea, use the first projection image (or a bunch of averages)
        # as a column reference, then set the sign of the bubbles positively
        Iref = self._load(self.PROJECTION_FILE_REGEX, proj_path, range(3, 100))
        Iref = np.min(Iref, axis=0)
        Iref = np.clip(Iref, 0, None)

        # # compute gas fraction (according to Evert)
        # np.divide(Imeas, Ifull, out=Imeas)
        # np.log(Imeas, out=Imeas)
        # np.divide(Ibright, Ifull, out=Ibright)
        # np.log(Ibright, out=Ibright)

        np.divide(Imeas, Iref, out=Imeas)
        np.clip(Imeas, 1.0, None, out=Imeas)
        np.log(Imeas, out=Imeas)
        return Imeas.astype(np.float32)
