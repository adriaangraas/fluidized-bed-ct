import itertools

import scipy.signal

from settings import *


def compute_connected_components_3d(labels_in: np.ndarray,
                                    connectivity: int = 6):
    import cc3d

    if labels_in.dtype != np.int32:
        raise ValueError("Should be int32.")

    if labels_in.ndim != 3:
        raise ValueError("Should be 3-dimensional.")

    if connectivity not in [6, 18, 26]:
        raise ValueError("Should be 6, 18, 26.")

    labels_out = cc3d.connected_components(labels_in,
                                           connectivity=connectivity)

    # You can adjust the bit width of the output to accomodate
    # different expected image statistics with memory usage tradeoffs.
    # uint16, uint32 (default), and uint64 are supported.
    # labels_out = cc3d.connected_components(labels_in, out_dtype=np.uint16)

    # You can extract individual components like so:
    # N = np.max(labels_out)
    # for segid in range(1, N + 1):
    #     extracted_image = labels_out * (labels_out == segid)
    #     process(extracted_image)

    # We also include a region adjacency graph function
    # that returns a set of undirected edges. It is not optimized
    # (70-80x slower than connected_components) but it could be improved.
    # graph = cc3d.region_graph(labels_out, connectivity=connectivity)

    return labels_out


def find_center_ball_cc():
    # this code is not finished.

    # CONNECTED COMPS 3D
    vol = (vol / np.max(vol) * 255).astype(np.int32)
    cc = compute_connected_components_3d(vol, 26)
    bins = np.bincount(cc.flatten())  # counts of each possible value
    largest_cc = sorted([[i, count] for i, count in enumerate(bins)],
                        key=lambda x: x[1],
                        reverse=True)  # most frequent counts
    nr = 1
    ccc = np.where(cc == largest_cc[nr][0], 1, 0)
    voll = np.where(cc == largest_cc[nr][0], vol, 0)


def find_center_ball(vol, phantom_radius, vox_sz=None):
    if vox_sz is None:
        volume_scaling_size_x = vol.shape[1] / DETECTOR_COLS
        volume_scaling_size_z = vol.shape[0] / DETECTOR_ROWS
        print(vol.shape)

        phantom_radius_vxls_x = phantom_radius / APPROX_VOXEL_WIDTH * volume_scaling_size_x
        phantom_radius_vxls_y = phantom_radius / APPROX_VOXEL_WIDTH * volume_scaling_size_x
        phantom_radius_vxls_z = phantom_radius / APPROX_VOXEL_HEIGHT * volume_scaling_size_z
    else:
        phantom_radius_vxls_x = phantom_radius / vox_sz[0]
        phantom_radius_vxls_y = phantom_radius / vox_sz[1]
        phantom_radius_vxls_z = phantom_radius / vox_sz[2]

    print(f"Calculating optimal location for phantom with radius "
          f"{phantom_radius * 10}mm, and a volume scaling size of "
          f" that means that approximately "
          f"voxel radius is ~({phantom_radius_vxls_x},{phantom_radius_vxls_y},{phantom_radius_vxls_z}).")

    r = int(np.round((phantom_radius_vxls_x + phantom_radius_vxls_y + phantom_radius_vxls_z) / 3))
    x, y, z = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    filter = (x * x + y * y + z * z <= r * r).astype(np.float32)
    vol_convolved = scipy.signal.fftconvolve(vol, filter)
    coord = np.unravel_index(np.argmax(vol_convolved), vol_convolved.shape)

    # because returned volume has r-padding on the boundaries, due to the convolution
    return coord[0] - r, coord[1] - r, coord[2] - r


# RECO_DIR = "/bigstore/adriaan/recons/evert/calibrated_24march2021/"
# RECO_DIR = "/home/adriaan/data/evert/recons/"
RECO_DIR = "/export/scratch2/adriaan/evert/reco/"
# DATA_DIR = "/bigstore/adriaan/data/evert/CWI/"
ACTUAL_VOX_SZ = [0.03390784] * 3

for scan in SCANS.items():  # type: Scan
    reco_dir = os.path.join(RECO_DIR, scan.name, 'size_250_algo_sirt_iters_200')
    coords_savename = f'{RECO_DIR}/{scan.name}_coords.npy'

    if isinstance(scan, DynamicScan) and len(scan.phantoms) > 1:
        print("Don't know how to deal with multi-phantoms scans yet. Skipin'")
        continue

    print("Loading volumes into memory...")
    coords = {}

    if not os.path.exists(coords_savename):
        if isinstance(scan, DynamicScan):
            if scan.timeframes is None:
                first_proj_nr = load_first_projection_number(scan.projs_dir)
                time_range = itertools.count(first_proj_nr)
            else:
                time_range = scan.timeframes

            for t in time_range:
                print('.', end='', flush=True)

                vol_filename = f'{reco_dir}/recon_t{str(t).zfill(6)}.npy'

                if not os.path.exists(vol_filename):
                    print(vol_filename)
                    print(f"Volume with t={t} not found, breaking.")
                    break

                vol = np.load(vol_filename)[...]
                vol = np.flip(vol, axis=2)  # reverse z-axis, 0=ground
                coord = find_center_ball(vol, scan.phantoms[0].radius, ACTUAL_VOX_SZ)
                print(coord)

                coords[t] = coord * np.array(ACTUAL_VOX_SZ) * 10

                np.save(coords_savename, coords)
    else:
        print(f"Path {coords_savename} exists, skipping.")

# if __name__ == '__main__':
#     # obsolete connected components code:
#
#     # find all connected components, 26 = faces + edges + corners
#     cc = compute_connected_components_3d(vol, 26)
#
#     # find biggest connected component
#     bins = np.bincount(cc.flatten())  # counts of each possible value
#     largest_cc = sorted([[i, count] for i, count in enumerate(bins)],
#                         key=lambda x: x[1],
#                         reverse=True)  # most frequent counts
#
#     # largest_cc[0] are the zeroes
#     # largest_cc[i] = i-th largest component = [index, number of voxels]
#
#     # plot_3d(cc)
#     nr = 1
#     ccc = np.where(cc == largest_cc[nr][0], 1, 0)
#     # extracted_image = cc * (cc == largest_cc[0])
#     plot_qtgraph(ccc)
#
#     N = np.max(cc)
#     for segid in range(1, N + 1):
#         extracted_image = cc * (cc == segid)
#         # process(extracted_image)
#         plot_3d(extracted_image)