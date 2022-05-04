import argparse
import itertools
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os

from fbrct import loader, reco, Scan, DynamicScan, StaticScan, FluidizedBedScan
import scripts.settings as sett
from fbrct.reco import Reconstruction


def _reco(projs_dir):
    """Generate a reconstruction object."""

    detector = {
        "rows": sett.DETECTOR_ROWS,
        "cols": sett.DETECTOR_COLS,
        "pixel_width": sett.DETECTOR_PIXEL_WIDTH,
        "pixel_height": sett.DETECTOR_PIXEL_HEIGHT,
    }

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


def _plot_sinos(y, pixel_start, pixel_end):
    from fbrct.util import plot_projs

    plot_projs(
        y, sett.DETECTOR_PIXEL_WIDTH, sett.DETECTOR_PIXEL_HEIGHT, pixel_start, pixel_end
    )


def reconstruct(
    scan: Scan,
    recodir: str,
    ref: Scan = None,
    voxels_x: int = 300,
    plot: bool = False,
    iters: int = 200,
    locking=False,
    save=False,
    algo="sirt",
    timeframes=None,
    **kwargs,
):
    print(f"Next up is {scan}...")

    def _callbackf(i, x, y_tmp):
        # median_filter(x, size=3, output=x)
        # footprint = cp.ones((1, 1, 3), dtype=cp.bool)
        # median_filter(x, footprint=footprint, output=x)
        if plot:
            # if i % 3 == 0:
            # x[x < 0.001] = 0.
            import matplotlib.pyplot as plt

            plt.figure("callbackf")
            plt.cla()
            # plt.imshow(x[..., 567].get())
            # plt.imshow(x[..., 278].get())
            # plt.imshow(x[..., 400].get())
            plt.imshow(x[..., x.shape[2] // 2].get())
            plt.pause(0.0001)

            plt.figure("callbackf2")
            plt.cla()
            plt.imshow(x[..., x.shape[2] // 2 - 100].get())
            plt.pause(0.0001)

            plt.figure("callbackf3")
            plt.cla()
            plt.imshow(x[..., x.shape[2] // 2 + 100].get())
            plt.pause(0.0001)

    reconstructor = _reco(scan.projs_dir)

    if isinstance(scan, DynamicScan):
        if scan.timeframes is None:
            scan_timeframes = loader.projection_numbers(scan.projs_dir)  # all frames
        else:
            scan_timeframes = scan.timeframes

        if timeframes is not None:
            if not np.all([s in scan_timeframes for s in timeframes]):
                raise ValueError(
                    "One or more timeframes are not in the set of"
                    " the scan's defined timeframes."
                )
        else:
            timeframes = scan_timeframes

        sino = _reconstruct_dynamic(
            reconstructor, scan, timeframes=timeframes, **kwargs
        )
        # per-frame reconstruction:
        for t, sino_t in zip(timeframes, sino):
            _inner_reco(
                scan,
                reconstructor,
                sino_t,
                scan.geometry,
                voxels_x,
                iters,
                callbackf=_callbackf,
                t=t,
                save=save,
                locking=locking,
            )
        reconstructor.clear()
    elif isinstance(scan, StaticScan):
        projs, geoms = _reconstruct_static(
            reconstructor, scan, ref, plot=plot, **kwargs
        )
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
            projs,
            geoms,
            voxels_x,
            iters,
            callbackf=_callbackf,
            locking=locking,
            save=save,
        )
    else:
        raise ValueError()

    reconstructor.clear()


def _reconstruct_dynamic(reco, scan: Scan, timeframes=None, normalization=None):
    """

    Parameters
    ----------
    normalization : object
    """

    ref_projs = []
    ref_path = None
    ref_lower_density = None
    ref_mode = "static"
    if ref is not None:
        if issubclass(type(scan), FluidizedBedScan):
            assert (
                not ref.is_rotational
            ), "Scan is fluidized bed, but reference is rotational?"
        else:
            assert ref.is_rotational, "Scan is dynamic, but reference is static?"

        ref_path = ref.projs_dir
        ref_lower_density = not ref.is_full
        ref_projs = [i for i in range(ref.proj_start, ref.proj_end)]

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

    sino = reco.load_sinogram(
        t_range=timeframes,
        # t_range=range(t, t + 1),
        # t_range=range(1, t),  # hacky way to average projs
        ref_mode=ref_mode,
        ref_path=ref_path,
        ref_projs=ref_projs,
        ref_normalization_path=ref_normalization_path,
        ref_normalization_projs=ref_normalization_projs,
        # darks_ran=range(10),
        # darks_path=scan.darks_dir,
        ref_lower_density=ref_lower_density,
    )
    return sino


def _reconstruct_static(
    reco,
    scan: StaticScan,
    ref: Scan = None,
    plot: bool = False,
    cameras=None,
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
            ref_mode=ref_mode,
            cameras=(cam,),
            ref_path=ref_path,
            ref_projs=ref_projs,
            darks_path=darks_path,
            darks_ran=darks_ran,
            ref_lower_density=ref_lower_density,
            ref_normalization_path=ref_normalization_path,
            ref_normalization_projs=ref_normalization_projs,
        )
        sino = np.squeeze(sino, axis=0)
        # sino = np.transpose(sino, [1, 0, 2])
        sinos.append(sino)

    # concat 3 cams into 1
    sino_flat = np.concatenate(sinos, axis=0)

    if plot:
        _plot_sinos(sino_flat, 550, 750)

    scan._geometry_rotation_offset = np.pi / 6
    geom_flat = np.array(
        [
            g
            for c, gs in scan.geometry.items()
            if c in cameras
            for i, g in enumerate(gs, scan.proj_start)
            if i in angles
        ]
    )
    return sino_flat, geom_flat


def _inner_reco(
    scan: Scan,
    reco: Reconstruction,
    sino,
    geoms,
    voxels_x,
    iters,
    t=None,
    callbackf=None,
    locking=False,
    save=False,
    save_tiff=False,
    save_mat=False,
    min_constraint=0.0,
    max_constraint=1.0,
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
    if os.path.exists(filename):
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
        if save:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
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
        "--ref-name",
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
        help="Starting timeframe from a dynamic " " scan.",
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
        help="Reconstruct a single angle from a rotational " " scan.",
        default=None,
    )

    parser.add_argument("--plot", "--p", action="store_true", default=False)
    parser.add_argument("--save", "--s", action="store_true", default=False)
    parser.add_argument("--save-tiff", action="store_true", default=False)
    parser.add_argument("--save-mat", action="store_true", default=False)
    parser.add_argument("--locking", action="store_true", default=False)

    args = parser.parse_args()
    name = args.scan
    refname = args.ref_name
    algo = args.algo
    iters = args.iters
    recodir = args.recodir
    time = args.time
    time_end = args.time_end
    angle = args.angle
    cam = args.cam
    plot = args.plot
    save = args.save
    locking = args.locking

    if plot:
        plt.rcParams.update({"figure.raise_window": False})

    if name != "all":
        scans = sett.get_scans(name)
        if len(scans) == 0:
            raise ValueError(f"Scan {name} not found.")
    else:
        scans = sett.SCANS

    kwargs = {"plot": plot, "locking": locking, "save": save}
    if algo is not None:
        kwargs["algo"] = algo
    if iters is not None:
        kwargs["iters"] = iters

    times = []
    if time is not None:
        if time_end is None:
            time_end = time + 1

        for t in range(time, time_end):
            times.append(t)
    kwargs["timeframes"] = times

    if cam is not None:
        kwargs["cameras"] = [cam]
    if angle is not None:
        kwargs["angles"] = [angle]

    for scan in scans:
        try:
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
                refs = sett.get_scans(refname)
                assert len(refs) == 1
                ref = refs[0]

            if scan.normalization is not None:
                kwargs["normalization"] = scan.normalization

            reconstruct(scan, recodir, ref, **kwargs)
        except ValueError:
            pass
