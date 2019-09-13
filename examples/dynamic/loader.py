import re
import warnings
import os
import imageio as im
import glob
import numpy as np
import skimage
import matplotlib.pyplot as plt
from tifffile import tifffile

from examples.settings import *


class Loader:
    CAMERA_DIR_REGEX = "camera ([1-3]{1})"
    PROJECTION_FILE_GLOB_PATTERN = "camera ?/img_*.tif"
    PROJECTION_FILE_REGEX = 'camera ([1-3]{1})/img_([0-9]{1,6})\.tif$'
    DARK_DIR = "pre_proc_Dark"
    BRIGHT_DIR = "pre_proc_Empty_FOV2"
    FULLS_DIR = "pre_proc_Full_FOV2"

    def __init__(self, path):
        if not os.path.exists(path):
            raise IOError("The path to the root directory not seem to exist.")

        self.path = path

    def _load(self, pattern, path, time_range: range = None):
        """
        This load function returns a multidim array of loaded files,
        guaranteeing the following consistency:
        - all subarrays have equal dimension
        - no files are missing
        - the file data have equal number of hor/vert pixels

        :param pattern:
        :param path:
        :param subsequences:
        :return:
        """
        results = list()
        results_filenames = list()

        regex = re.compile(pattern)
        for root, dirs, files in os.walk(path):
            for file in files:
                full_filename = os.path.join(root, file)
                match = regex.search(full_filename)
                if match is not None:
                    groups = match.groups()

                    tmp_result = [None] * len(groups)
                    for i, group in enumerate(groups):
                        # check if numbers are captured
                        if not group.isdigit():
                            raise Exception(
                                "The regex captured a non-digit, I'm not sure how to index this.")

                        tmp_result[i] = int(group)

                    results.append(tmp_result)
                    results_filenames.append(full_filename)

        # Check the results for continuity in the subsequences range
        lists = list(zip(*results))
        amount_matches = lists[0].count(1)
        assert amount_matches == lists[0].count(2) == lists[0].count(3)
        assert amount_matches > 0

        #
        if time_range is None:
            time_range = range(np.min(lists[1]), np.max(lists[1]) + 1)

        # load into results
        im_shape = tifffile.imread(results_filenames[0]).shape
        ims = np.empty((len(time_range), 3, *im_shape))

        for (camera_id, proj_id), filename in zip(results, results_filenames):
            ims[proj_id, camera_id-1, ...] = tifffile.imread(filename)

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

    def projs(self, dir = "pre_proc_1_65lmin_83mm_FOV2", t_range: range = None):
        Idark = np.mean(self.darks(t_range), axis=0)
        #@todo I should use m

        Ibright = np.mean(self.brights(t_range), axis=0)
        Ifull = np.mean(self.fulls(t_range), axis=0)

        proj_path = os.path.join(self.path, dir)
        Imeas = self._load(self.PROJECTION_FILE_REGEX, proj_path, t_range)

        # compute gas fraction
        np.divide(Imeas, Ifull, out=Imeas)
        np.log(Imeas, out=Imeas)

        np.divide(Ibright, Ifull, out=Ibright)
        np.log(Ibright, out=Ibright)

        np.divide(Imeas, Ibright, out=Imeas)

        return Imeas

        # # camera_dir_regex = re.compile(self.CAMERA_DIR_REGEX)
        # # camera_dirs = list(filter(lambda f: regex.match(f), os.listdir(dark_path)))
        # # camera_nrs = list(map(lambda camera: int(re.match(self.CAMERA_DIR_REGEX, camera).group(1)),
        # #                       camera_dirs))
        #

        # # all pairs (camera_id, proj_id)
        # pairs = list(map(
        #     lambda path: (int(regex.search(path).group(1)), int(regex.search(path).group(1))),
        #     list(all_files)))
        #
        # # [[1,1,1,..., 2,2,..., 3,3,...], [8,64,3..., ]]

        #
        # # all cameras should have an equal number of projections
        #
        # # for f in files:
        # #     path = os.path.join(self.path, f)
        # #     ims.append(np.asarray(im.imread(path), dtype=np.float32))
        #
        # # allocate
        # ims = [] * t
        # for camera, projection in pairs:
        #
        # return ims
