import argparse
import itertools
import warnings
import numpy as np

import scipy.signal

from fbrct import loader, reco
from settings import *


def find_center_ball(vol, phantom_radius, vox_sz):
    phantom_radius_vxls_x = phantom_radius / vox_sz[0]
    phantom_radius_vxls_y = phantom_radius / vox_sz[1]
    phantom_radius_vxls_z = phantom_radius / vox_sz[2]

    print(f"Calculating optimal location for phantom with radius "
          f"{phantom_radius * 10}mm, and a volume scaling size of "
          f" that means that approximately "
          f"voxel radius is ~({phantom_radius_vxls_x},{phantom_radius_vxls_y},{phantom_radius_vxls_z}).")

    r = int(np.round((
                         phantom_radius_vxls_x + phantom_radius_vxls_y + phantom_radius_vxls_z) / 3))
    x, y, z = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    filter = (x * x + y * y + z * z <= r * r).astype(np.float32)
    vol_convolved = scipy.signal.fftconvolve(vol, filter)
    coord = np.unravel_index(np.argmax(vol_convolved), vol_convolved.shape)

    # because returned volume has r-padding on the boundaries, due to the convolution
    return coord[0] - r, coord[1] - r, coord[2] - r


def parse(scan, recodir):
    coords_savename = f'{recodir}/{scan.name}_coords.npy'
    if os.path.exists(coords_savename):
        warnings.warn(f"Path {coords_savename} exists.")
        return

    if scan.timeframes is None:
        first_proj_nr = loader.load_first_projection_number(scan.projs_dir)
        time_range = itertools.count(first_proj_nr)
    else:
        time_range = scan.timeframes

    coords = {}
    for t in time_range:
        print('.', end='', flush=True)
        vol_filename = f'{recodir}/recon_t{str(t).zfill(6)}.npy'

        if not os.path.exists(vol_filename):
            print(vol_filename)
            print(f"Volume with t={t} not found, breaking.")
            break

        reco = np.load(vol_filename, allow_pickle=True).item()
        vol = reco['volume']
        vol = np.flip(vol, axis=2)  # reverse z-axis, 0=ground

        vox_sz = reco['vol_params'][3]
        coord = find_center_ball(vol, scan.phantoms[0].radius, vox_sz)
        print(coord)
        coords[t] = coord * np.array(vox_sz) * 10
        np.save(coords_savename, coords)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coordinate parsing")

    parser.add_argument("--name", type=str,
                        help="Name of the experiment to run."
                             " Must be available in settings.py. Runs"
                             " all SCANS from settings.py if not provided.")
    parser.add_argument("--reco", type=str,
                        help="Filename of reconstruction (without .npy).")
    parser.add_argument("--recodir", type=str,
                        help="directory to store reconstructions",
                        default='./')

    args = parser.parse_args()
    recodir = args.recodir

    assert len(get_scans(args.name)) == 1
    scan = get_scans(args.name)[0]
    if not isinstance(scan, TraverseScan):
        warnings.warn("Expecting a `TraverseScan`.")
        exit(0)

    subdir = os.path.join(recodir, scan.name, args.reco)
    parse(scan, subdir)
