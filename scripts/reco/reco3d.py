import argparse
import os
from pathlib import Path
from typing import Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import warnings

import scripts.settings as sett
from fbrct import loader, reco, Scan, DynamicScan, StaticScan, FluidizedBedScan
from fbrct.reco import Reconstruction
from fbrct.util import plot_projs


def _reco(projs_dir: str, detector):
    """Generate a reconstruction object."""
    return reco.RayveReconstruction(
        projs_dir,
        detector=detector,
        expected_voxel_size_x=sett.APPROX_VOXEL_WIDTH,
        expected_voxel_size_z=sett.APPROX_VOXEL_HEIGHT,
    )


def _plot(x):
    import pyqtgraph as pq
    import matplotlib.pyplot as plt

    pq.image(x)
    plt.figure()
    plt.imshow(x[..., x.shape[2] // 2])
    plt.show()


def reconstruct(
    scan: Scan,
    recodir: str,
    ref: Scan = None,
    voxels_x: int = 300,
    plot: bool = False,
    iters: int = 200,
    locking=False,
    overwrite=False,
    save=False,
    save_tiff=False,
    save_mat=False,
    algo="sirt",
    timeframes=None,
    detector_rows: range = None,
    ref_reduction: str = None
):
    print(f"Next up is {scan}...")

    def _callbackf(i, x, y_tmp):
        import matplotlib.pyplot as plt

        # median_filter(x, size=3, output=x)
        # footprint = cp.ones((1, 1, 3), dtype=cp.bool)
        # median_filter(x, footprint=footprint, output=x)
        if plot and i % 100 == 0:
            # if i % 3 == 0:
            # x[x < 0.001] = 0.
            plt.figure("quick-3d")
            for z in range(0, x.shape[2], 10):
                plt.cla()
                plt.imshow(x[..., z].get())
                plt.pause(0.01)

            # plt.figure("MIP axis=1")
            # plt.imshow(cp.max(x, axis=1).get().T)
            # plt.pause(0.0001)

    reconstructor = _reco(scan.projs_dir, scan.detector)

    if isinstance(scan, DynamicScan):
        if scan.projs is None:
            scan_projs = loader.projection_numbers(scan.projs_dir)  # all frames
        else:
            scan_projs = scan.projs

        if timeframes is not None:
            if not np.all([s in scan_projs for s in timeframes]):
                raise ValueError(
                    "One or more timeframes are not in the set of"
                    " the scan's defined projections."
                )
        else:
            timeframes = scan_projs

        projs = _sino_dynamic(
            scan, reconstructor, ref,
            timeframes=timeframes, detector_rows=detector_rows,
            ref_reduction=ref_reduction
        )
        # per-frame reconstruction:
        for t, sino_t in zip(timeframes, projs):
            if plot:
                plot_projs(projs[0])

            _inner_reco(
                scan,
                reconstructor,
                recodir,
                algo,
                sino_t,
                scan.geometry(),
                voxels_x,
                iters,
                callbackf=_callbackf,
                t=t,
                save=save,
                save_tiff=save_tiff,
                save_mat=save_mat,
                locking=locking,
                overwrite=overwrite,
                plot=plot,
            )

        reconstructor.clear()
    elif isinstance(scan, StaticScan):
        projs, geoms = _sino_static(scan, reconstructor,
                                    ref, plot=plot, **kwargs)
        plot_projs(projs)
        # vol_id, vol_geom = reco.backward(
        #     proj_id,
        #     proj_geom,
        #     algo=algo,
        #     voxels_x=voxels_x,
        #     iters=iters,
        #     min_constraint=0.,
        #     max_constraint=1.,
        #     callback=callbackf)
        # x = reco.volume(vol_id)
        _inner_reco(
            scan,
            reconstructor,
            recodir,
            algo,
            projs,
            geoms,
            voxels_x,
            iters,
            callbackf=_callbackf,
            locking=locking,
            overwrite=overwrite,
            save=save,
        )
    else:
        raise ValueError()

    reconstructor.clear()


def _sino_dynamic(
    scan: Scan,
    reco: Reconstruction,
    ref: Scan,
    ref_reduction: None,
    timeframes=None,
    **kwargs
):
    """

    Parameters
    ----------
    normalization : object
    """

    if isinstance(ref, StaticScan):
        if issubclass(type(scan), FluidizedBedScan):
            assert (
                not ref.is_rotational
            ), "Scan is fluidized bed, but reference is rotational?"
        else:
            assert ref.is_rotational, "Scan is dynamic, but reference is static?"

        ref_path = ref.projs_dir
        ref_lower_density = not ref.is_full
        ref_projs = [i for i in range(ref.proj_start, ref.proj_end)]
        ref_rotational = ref.is_rotational
        if not ref_reduction in ('mean', 'median'):
            warnings.warn("For static scans, 'mean' or 'medium' are good"
                          " reduction choices.")
    elif isinstance(ref, FluidizedBedScan):
        if issubclass(type(scan), FluidizedBedScan):
            if ref.liter_per_min != scan.liter_per_min:
                warnings.warn("Using a reference with a different l/min then "
                              "that is used in the experiment. This is not "
                              "optimal.")
        ref_path = ref.projs_dir
        ref_lower_density = not ref.is_full
        ref_projs = ref.projs
        if not ref_reduction in ('mode',):
            warnings.warn("For fluidized bed scans, 'mode' is a good"
                          " reduction choice. Another option that might not"
                          " introdce bubble bias is 'min', but this will cause"
                          " a density mismatch.")
        ref_rotational = ref.is_rotational
    else:
        ref_projs = []
        ref_path = None
        ref_lower_density = None
        ref_rotational = False

    empty_path = None
    empty_projs = None
    empty_rotational = False
    if scan.density_factor is None:
        assert ref_path is not None
        if scan.empty is not None:
            empty_rotational = scan.empty.is_rotational
            assert not scan.empty.is_rotational
            assert not scan.empty.is_full
            empty_path = scan.empty.projs_dir
            empty_projs = [p for p in range(
                scan.empty.proj_start, scan.empty.proj_end
            )]

    sino = reco.load_sinogram(
        t_range=timeframes,
        # t_range=range(t, t + 1),
        # t_range=range(1, t),  # hacky way to average projs
        ref_rotational=ref_rotational,
        ref_reduction=ref_reduction,
        ref_path=ref_path,
        ref_projs=ref_projs,
        empty_path=empty_path,
        empty_rotational=empty_rotational,
        empty_projs=empty_projs,
        # darks_ran=range(10),
        # darks_path=scan.darks_dir,
        ref_lower_density=ref_lower_density,
        density_factor=scan.density_factor,
        col_inner_diameter=scan.col_inner_diameter,
        **kwargs,
    )
    return sino


def _sino_static(
    scan: StaticScan,
    reco: Reconstruction,
    ref: Scan = None,
    plot: bool = False,
    cameras: Tuple = None,
    angles=None,
    normalization=None,
):
    if cameras is None:
        cameras = scan.cameras
    elif not np.all([c in scan.cameras for c in cameras]):
        raise ValueError("One or more unknown cameras.")

    scan_angles = range(scan.proj_start, scan.proj_end)
    if angles is None:
        angles = scan_angles
    elif not np.all([a in scan_angles for a in angles]):
        raise ValueError(
            f"One or more unknown projection angles. {scan}"
            f" has angles defined for"
            f" {scan.proj_start}-{scan.proj_end}."
        )

    ref_projs = []
    ref_path = None
    ref_lower_density = None
    ref_mode = "reco"
    if ref is not None:
        ref_path = ref.projs_dir
        ref_lower_density = not ref.is_full
        ref_mode = "reco" if ref.is_rotational else "static"
        for a in angles:
            ref_projs.append(a - scan.proj_start + ref.proj_start)

    ref_normalization_projs = []
    ref_normalization_path = None
    if normalization is not None:
        assert ref_path is not None
        assert normalization.is_rotational
        ref_normalization_path = normalization.projs_dir
        for a in angles:
            ref_normalization_projs.append(
                a - scan.proj_start + normalization.proj_start
            )
    darks_ran = None
    darks_path = None
    if scan.darks is not None:
        darks_path = scan.darks.projs_dir
        darks_ran = range(scan.darks.proj_start, scan.darks.proj_end)

    sinos = []
    for cam in cameras:
        sino = reco.load_sinogram(
            t_range=angles,
            ref_rotational=ref_mode,
            cameras=(cam,),
            ref_path=ref_path,
            ref_projs=ref_projs,
            darks_path=darks_path,
            darks_ran=darks_ran,
            ref_lower_density=ref_lower_density,
            empty_path=ref_normalization_path,
            empty_projs=ref_normalization_projs,
        )
        sino = np.squeeze(sino, axis=0)
        # sino = np.transpose(sino, [1, 0, 2])
        sinos.append(sino)

    # concat 3 cams into 1
    sino_flat = np.concatenate(sinos, axis=0)

    if plot:
        plot_projs(sino_flat)

    scan._geometry_rotation_offset = np.pi / 6
    geom_flat = np.array(
        [
            g
            for c, gs in scan.geometry().items()
            if c in cameras
            for i, g in enumerate(gs, scan.proj_start)
            if i in angles
        ]
    )
    return sino_flat, geom_flat


def _inner_reco(
    scan: Scan,
    reco: Reconstruction,
    recodir: str,
    algo: str,
    sino,
    geoms,
    voxels_x,
    iters: int,
    t: int = None,
    callbackf: Callable = None,
    locking: bool = False,
    overwrite: bool = False,
    save: bool = False,
    save_tiff: bool = False,
    save_mat: bool = False,
    min_constraint: float = 0.0,
    max_constraint: float = 1.0,
    plot=False,
):
    def _fname(ext, t=None):
        """Filename for reconstruction"""

        if t is not None:
            save_name = f"recon_t{str(t).zfill(6)}.{ext}"
            return os.path.join(
                recodir,
                scan.name,
                f"size_{voxels_x}_algo_{algo}_iters_{iters}",
                save_name,
            )
        else:
            return os.path.join(
                recodir, scan.name, f"size_{voxels_x}_algo_{algo}_iters_{iters}.{ext}"
            )

    # sino[0, ...] = np.mean(sino, axis=0)  # hacky averaging projs

    # for i, s in enumerate(sino):
    #    sino[i, ...] = savgol_filter(s, 7, 5, axis=-2, mode='mirror')
    #    sino[i, ...] = savgol_filter(s, 7, 5, axis=-1, mode='mirror')

    # sino[sino < 0.025] = 0.

    # for i, s in enumerate(sino):
    #     sino[i, ...] = savgol_filter(s, 13, 7, axis=-2, mode='mirror')

    #    sino[i, ...] = savgol_filter(s, 13, 7, axis=-1, mode='mirror')
    # sino[sino < 0.5] = 0.

    # for i, s in enumerate(sino[0]):
    #    sino[0, i] = denoise_wavelet(s, 1e-1, rescale_sigma=True)

    filename = _fname("npy", t)
    if os.path.exists(filename) and not overwrite:
        print(f"File {filename} exists, continuing.")
        return

    if locking:
        lockfile = _fname("lock", t)
        if os.path.exists(lockfile):
            print(f"Lockfile {lockfile} exists, continuing.")
            return
        else:
            os.makedirs(os.path.dirname(lockfile), exist_ok=True)
            Path(lockfile).touch()

    x = None
    try:
        print(f"Starting {filename}...")

        proj_id, proj_geom = reco.sino_gpu_and_proj_geom(sino, geoms)

        vol_id, vol_geom = reco.backward(
            proj_id,
            proj_geom,
            algo=algo,
            voxels_x=voxels_x,
            iters=iters,
            min_constraint=min_constraint,
            max_constraint=max_constraint,
            col_mask=True,
            callback=callbackf,
        )
        x = reco.volume(vol_id)

        infodict = {
            "timeframe": t,
            "name": scan.name,
            "volume": x,
            "geometry": proj_geom,
            "algorithm": algo,
            "nr_iters": iters,
            "vol_params": reco.vol_params(voxels_x),
        }

        print(f"Saving {filename}...")
        if save or save_mat or save_tiff:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        if save:
            np.save(filename, infodict, allow_pickle=True)

        if save_tiff:
            tifffile.imsave(_fname("tiff", t), x, x.shape)

        if save_mat:
            from scipy.io import savemat

            filename = _fname("mat", t)
            savemat(filename, infodict)
    finally:
        if locking:
            try:
                print(f"Lockfile {_fname('lock', t)} could not be unlinked.")
                Path(_fname("lock", t)).unlink()
            except:
                pass

    if plot and x is not None:
        import pyqtgraph as pq

        pq.image(x.T)
        plt.figure()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction")

    parser.add_argument(
        "--scan",
        type=str,
        help="Name of the Scan to run."
        " Must be available in settings.py. Runs"
        " all SCANS from settings.py if not provided.",
        default="all",
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="Name of the reference scan."
        " Must be available in settings.py. If not "
        " given, will use `scan.references[0]`.",
        default=None,
    )
    parser.add_argument(
        "--recodir", type=str, help="directory to store reconstructions", default="./"
    )
    parser.add_argument(
        "--algo", type=str, help="Algorithm to use (sirt, fdk, nesterov)", default=None
    )
    parser.add_argument(
        "--iters",
        type=int,
        help="Number of iterations for iterative algorithms.",
        default=None,
    )
    parser.add_argument(
        "--backend", type=str, help="Backend to use (astra, rayve)", default="rayve"
    )
    parser.add_argument(
        "--cam", type=int, help="Reconstructs with a single camera.", default=None
    )
    parser.add_argument(
        "--time",
        type=int,
        help="Starting timeframe from a dynamic scan.",
        default=None,
    )
    parser.add_argument(
        "--time-start",
        type=int,
        help="Starting timeframe from a dynamic scan.",
        default=None,
    )
    parser.add_argument(
        "--time-end",
        type=int,
        help="Timeframe to end, in a dynamic scan.",
        default=None,
    )
    parser.add_argument(
        "--angle",
        type=int,
        help="Reconstruct a single angle from a rotational scan.",
        default=None,
    )
    parser.add_argument(
        "--detector-rows",
        type=int,
        nargs=2,
        help="Subselect a detector row range, i.e. --detector-rows 100 500",
        default=None,
    )
    parser.add_argument(
        "--ref-max",
        type=int,
        help="Limit the number of averaging reference projections, to save loading time.",
        default=None,
    )
    parser.add_argument(
        "--ref-reduction",
        type=str,
        help="For nonrotational references, how to average the projections."
             " When the ref is a fluidized bed, choose `mode` or `min`. For "
             " e.g. a full bed with many shots, choose `mean` or `median`.",
        default=None,
    )

    parser.add_argument("--plot", "--p", action="store_true", default=False)
    parser.add_argument("--save", "--s", action="store_true", default=False)
    parser.add_argument("--save-tiff", action="store_true", default=False)
    parser.add_argument("--save-mat", action="store_true", default=False)
    parser.add_argument("--locking", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()
    name = args.scan
    refname = args.ref
    algo = args.algo
    iters = args.iters
    recodir = args.recodir
    time = args.time
    time_start = args.time_start
    time_end = args.time_end
    angle = args.angle
    cam = args.cam
    plot = args.plot
    save = args.save
    save_tiff = args.save_tiff
    save_mat = args.save_mat
    locking = args.locking
    overwrite = args.overwrite
    detector_rows = args.detector_rows
    ref_max = args.ref_max
    ref_reduction = args.ref_reduction

    if plot:
        plt.rcParams.update({"figure.raise_window": False})

    if name != "all":
        scans = sett.get_scans(name)
        if len(scans) == 0:
            raise ValueError(f"Scan {name} not found.")
    else:
        scans = sett.SCANS

    kwargs = {"plot": plot, "locking": locking, "save": save,
              "overwrite": overwrite, 'save_tiff': save_tiff,
              'save_mat': save_mat, 'ref_reduction': ref_reduction}
    if algo is not None:
        kwargs["algo"] = algo
    if iters is not None:
        kwargs["iters"] = iters

    times = None
    if time is not None:
        assert time_start is None and time_end is None
        times = [t for t in range(time, time + 1)]
    if time_start is not None:
        assert time is None and time_end > time_start
        times = [t for t in range(time_start, time_end)]

    kwargs["timeframes"] = times

    if cam is not None:
        kwargs["cameras"] = [cam]
    if angle is not None:
        kwargs["angles"] = [angle]
    if detector_rows is not None:
        assert 0 <= detector_rows[0] < detector_rows[1]
        kwargs["detector_rows"] = range(detector_rows[0], detector_rows[1])

    for scan in scans:
        if refname is None:
            if len(scan.references) == 1:
                ref = scan.references[0]
            else:
                raise ValueError(
                    f"Zero, or multiple references defined for scan {scan.name}. Either"
                    f" define a reference in settings.py or provide with which reference"
                    f" to reconstruct."
                )
        else:
            found = False
            for ref in scan.references:
                if ref.name == refname:
                    found = True
                    break

            if not found:
                refs = sett.get_scans(refname)
                assert len(refs) == 1
                raise ValueError(f"Add {refs[0].name} to {scan}'s references.")

        if ref_max is not None:
            if isinstance(ref, StaticScan):
                if ref.is_rotational:
                    raise ValueError(
                        "Cannot set maximum to-use projections for averaging"
                        " reference. When the scan is rotational, there is only"
                        " one projection per angle, nothing to average."
                    )
                assert ref_max >= 1
                ref.proj_end = min((ref.proj_start + ref_max, ref.proj_end))
            elif isinstance(ref, FluidizedBedScan):
                ref.projs = range(ref.projs[0], ref.projs[0] + ref_max)
            else:
                raise NotImplementedError(f"Don't know how to apply `ref_max` "
                                          f"to {ref}.")

            if isinstance(scan.empty, StaticScan):
                if scan.empty.is_rotational:
                    raise ValueError(
                        "Cannot set maximum to-use projections for averaging"
                        " empty. When the scan is rotational, there is only"
                        " one projection per angle, nothing to average."
                    )
                assert ref_max >= 1
                scan.empty.proj_end = min(
                    (scan.empty.proj_start + ref_max, scan.empty.proj_end))
            else:
                raise NotImplementedError(f"Don't know how to apply `ref_max` "
                                          f"to {scan.empty}.")

        reconstruct(scan, recodir, ref, **kwargs)
