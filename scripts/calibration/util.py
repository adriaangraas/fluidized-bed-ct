import copy
from typing import Sequence

import numpy as np
from cate import xray
from cate.annotate import Annotator
from cate.astra import geom2astravec
from cate.param import ScalarParameter, VectorParameter
from cate.xray import markers_from_leastsquares_intersection, XrayOptimizationProblem
from fbrct.reco import Reconstruction
from fbrct.loader import load, preprocess


def _load_median_projs(path, t, fulls_path=None):
    if fulls_path is None:
        # TODO
        Imeas = load(path, range(t, t + 1))
        np.clip(Imeas, 1.0, None, out=Imeas)
        np.log(Imeas, out=Imeas)
        ims = Imeas.astype(np.float32)
        return ims[0]
    else:
        p = load_referenced_projs_from_fulls(path,
                                             fulls_path,
                                             t_range=range(t, t + 1),
                                             reference_method='median')
        return np.median(p, axis=0)


def _load_projs(path, t_range: Sequence, camera: int):
    p = load(path, time_range=t_range, cameras=[camera])
    return preprocess(p)


def annotated_data(
    projs_path: str,
    times: Sequence,
    entity_locations_class: type,
    fname: str,
    cameras: Sequence = (1, 2, 3),
    open_annotator: bool = False,
    vmin=None,
    vmax=None) -> list:
    """(Re)store marker projection coordinates from annotations

    :return list
        List of `dict`, each dict being a projection angle, and each item
        from the dictionary is a key-value pair of identifier and pixel
        location."""

    data = {c: [] for c in cameras}
    for cam in cameras:
        for t in times:
            # open a EntityLocations class for this file, in the
            points = entity_locations_class(f"resources/{fname}_cam{cam}.npy",
                                            t)
            if open_annotator:
                projs = _load_projs(projs_path, t_range=range(t, t + 1),
                                    camera=cam)
                projs = np.squeeze(projs)
                Annotator(points, projs, block=True, vmin=vmin, vmax=vmax)

            l = points.locations()
            data[cam].append(l)

    return data


def triangle_geom(src_rad, det_rad, rotation=False, shift=False,
                  fix_first_det=True, weight_decay=None):
    geoms = []
    for i, src_a in enumerate([0, 2 / 3 * np.pi, 4 / 3 * np.pi]):
        det_a = src_a + np.pi  # opposing
        src = src_rad * np.array([np.cos(src_a), np.sin(src_a), 0])
        det = det_rad * np.array([np.cos(det_a), np.sin(det_a), 0])
        if i == 0 and fix_first_det:
            print("Fixing first detector.")
            geom = xray.StaticGeometry(
                source=VectorParameter(src, weight_decay=weight_decay),
                # source=src,
                detector=det,
                roll=None,
                pitch=None,
                yaw=None,
            )
        else:
            geom = xray.StaticGeometry(
                source=VectorParameter(src, weight_decay=weight_decay),
                detector=VectorParameter(det, weight_decay=weight_decay),
                roll=ScalarParameter(None),
                pitch=ScalarParameter(None),
                yaw=ScalarParameter(None),
            )

        geoms.append(geom)

    if rotation:
        # transform the whole geometry by a global rotation, this is the
        # same as if the phantom rotated

        rotation_roll = ScalarParameter(0.)
        rotation_pitch = ScalarParameter(0.)
        rotation_yaw = ScalarParameter(0.)

        for i in range(len(geoms)):
            geoms[i] = xray.transform(geoms[i],
                                      rotation_roll,
                                      rotation_pitch,
                                      rotation_yaw)

    if shift:
        shift_param = VectorParameter(np.array([0., 0., 0.]))

        for i in range(len(geoms)):
            geoms[i] = xray.shift(geoms[i], shift_param)

    return geoms


def astra_reco_rotation_singlecamera(
    reco: Reconstruction,
    data,
    geoms,
    algo='FDK',
    voxels_x=400, iters=200):
    # detector_mid = DETECTOR_ROWS // 2
    # offset = DETECTOR_ROWS // 2 - 0
    # sinogram = p[0, :, recon_height_range,
    #            recon_width_range.start:recon_width_range.stop]
    # recon_height_range = range(detector_mid - offset, detector_mid + offset)
    # recon_width_range = range(DETECTOR_COLS)
    # recon_height_length = int(len(recon_height_range))
    # recon_width_length = int(len(recon_width_range))
    vectors = np.array([geom2astravec(g, reco.detector) for g in geoms])
    proj_id, proj_geom = reco.sino_gpu_and_proj_geom(data, vectors)
    vol_id, vol_geom = reco.backward(proj_id, proj_geom, algo=algo,
                                     voxels_x=voxels_x, iters=iters)
    return vol_id, vol_geom


def astra_reco(
    reco: Reconstruction,
    projs,
    geoms,
    algo='SIRT',
    voxels_x=400):
    """
    Reconstruction without rotation table:
     - 3 sources
     - 3 detectors
     - triangular set-up
    """
    # detector_mid = DETECTOR_ROWS // 2
    # offset = DETECTOR_ROWS // 2 - 0
    # sinogram = p[0, :, recon_height_range,
    #            recon_width_range.start:recon_width_range.stop]
    # recon_height_range = range(detector_mid - offset, detector_mid + offset)
    # recon_width_range = range(DETECTOR_COLS)
    # recon_height_length = int(len(recon_height_range))
    # recon_width_length = int(len(recon_width_range))
    assert len(geoms) == 3

    vectors = np.array([geom2astravec(g, reco.detector) for g in geoms])
    proj_id, proj_geom = reco.sino_gpu_and_proj_geom(projs, vectors)
    vol_id, vol_geom = reco.backward(proj_id, proj_geom, algo=algo,
                                     voxels_x=voxels_x)
    return vol_id, vol_geom


def astra_project(reco, vol_id, vol_geom, geoms):
    vectors = np.array([geom2astravec(g, reco.detector) for g in geoms])
    sino_id, proj_geom = reco.sino_gpu_and_proj_geom(
        0.,  # zero-sinogram
        vectors)

    proj_id = reco.forward(
        volume_id=vol_id,
        volume_geom=vol_geom,
        projection_geom=proj_geom,
    )
    return reco.sinogram(proj_id)


def astra_residual(reco, data, vol_id, vol_geom, geoms):
    """Projects then substracts a volume with `vol_id` and `vol_geom`
    onto projections from `projs_path` with `nrs`.

    Using `geoms` or `angles`. If geoms are perfect, then the residual will be
    zero. Otherwise it will show some geometry forward-backward mismatch.
    """
    reproj = astra_project(reco, vol_id, vol_geom, geoms)

    assert reproj.shape == data.shape, f"reproj shape is {reproj.shape} while" \
                                       f" data.shape is {data.shape}"
    return data - reproj


def plot_projections(res, vmin=None, vmax=None, title=None):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=1, ncols=res.shape[1])
    if title is not None:
        plt.title(title)

    if res.shape[1] == 1:
        plt.imshow(res[:, 0], vmin=vmin, vmax=vmax)
        plt.colorbar()
    else:
        for i in range(res.shape[1]):
            im = axs[i].imshow(res[:, i], vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=axs[i])

    plt.pause(.1)


def prep_projs(projs, crop_cols):
    def _crop(projs, crop_cols):
        return projs[:, :, crop_cols // 2:int(projs.shape[2] - crop_cols // 2)]

    # projs *= -1
    # projs += 9.5
    # np.clip(projs, -0.5, None, out=projs)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(projs[0, 0])
    # plt.show()
    projs = np.squeeze(projs)
    # projs = _crop(projs, crop_cols)
    projs = np.swapaxes(projs, 0, 1)
    return np.ascontiguousarray(projs)


def run_initial_marker_optimization(
    geoms, data, nr_iters: int = 20,
    max_nfev=10,
    plot=False, **kwargs):
    """Find points of the phantom that we have because of a high-resolution
    prescan.

    It's probably better to keep max_n
    """
    # `geoms` and `markers` will be optimized in-place, so we'd make a backup here
    # for later reference.
    geoms_before = copy.deepcopy(geoms)

    # Since the prescan is not perfect, we do a simple optimization process
    # over the prescan parameters.
    # Repeating optimizing over the "ground-truth geometry"
    assert nr_iters > 0
    for i in range(nr_iters):
        # Get Least-Square optimal points by analytically backprojection the
        # marker locations, which is a LS intersection of lines.
        markers = markers_from_leastsquares_intersection(
            geoms, data,
            optimizable=False,
            plot=plot,
        )

        # Then find the optimal geoms given the `points` and `data`, in-place.
        problem = XrayOptimizationProblem(
            markers=markers,
            geoms=geoms,
            data=data,
            use_multiprocessing=False
        )

        from cate.param import params2ndarray
        import scipy.optimize
        r = scipy.optimize.least_squares(
            fun=problem,
            x0=params2ndarray(problem.params()),
            bounds=problem.bounds(),
            verbose=2,
            # method=method,
            tr_solver='exact',
            # loss=loss,
            jac='3-point',
            max_nfev=max_nfev,
            **kwargs
        )

        np.set_printoptions(precision=4, suppress=True)
        for i, (g1, g2) in enumerate(zip(geoms_before, geoms)):
            print("")
            print(f"--- GEOM {i} old:new properties ---")
            print(f"source   : {g1.source} : {g2.source}")
            print(f"detector : {g1.detector} : {g2.detector}")
            print(f"roll     : {g1.roll} : {g2.roll}")
            print(f"pitch    : {g1.pitch} : {g2.pitch}")
            print(f"yaw      : {g1.yaw} : {g2.yaw}")
            print(f"--- GEOM {i} unshifted old:new:err properties ---")
            g1_un = g1.decorated_geometry.decorated_geometry
            g2_un = g2.decorated_geometry.decorated_geometry
            print(
                f"source   : {g1_un.source} : {g2_un.source} : {g1_un.source - g2_un.source}")
            print(
                f"detector : {g1_un.detector} : {g2_un.detector} : {g1_un.detector - g2_un.detector}")
            print(
                f"roll     : {g1_un.roll} : {g2_un.roll} : {g1_un.roll - g2_un.roll}")
            print(
                f"pitch    : {g1_un.pitch} : {g2_un.pitch} : {g1_un.pitch - g2_un.pitch}")
            print(
                f"yaw      : {g1_un.yaw} : {g2_un.yaw} : {g1_un.yaw - g2_un.yaw}")
            print(f"--- GEOM {i} old:new parameters ---")
            for i, (p1, p2) in enumerate(zip(g1.parameters(),
                                             g2.parameters())):
                print(f"{id(p1)} : {p1.value} : {p2.value}")

    # noinspection PyUnboundLocalVariable
    return markers
