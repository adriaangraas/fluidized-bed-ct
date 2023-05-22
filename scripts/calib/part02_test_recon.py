import matplotlib.pyplot as plt
import pyqtgraph as pq

import cate.astra as cate_astra
from cate.util import geoms_from_interpolation
from scripts.calib.util import *
from scripts.settings import *

detector = cate_astra.Detector(DETECTOR_ROWS, DETECTOR_COLS,
                               DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT)

# directory of the calibration scan
DATA_DIR_CALIB = "/home/adriaan/ownCloud2/"
MAIN_DIR_CALIB = "pre_proc_Calibration_needle_phantom_30degsec_table474mm"

# directory of a scan to reconstruct (can be different or same to calib)
DATA_DIR = "/home/adriaan/ownCloud2/"
MAIN_DIR = "pre_proc_Calibration_needle_phantom_30degsec_table474mm"
PROJS_PATH = f'{DATA_DIR}/{MAIN_DIR}'

# configure which projection range to take
if MAIN_DIR == "pre_proc_3x10mm_foamballs_vertical_01":
    proj_start = 37
    proj_end = 1621
    ref_path = '/home/adriaan/ownCloud3/pre_proc_Full_30degsec_03'
elif MAIN_DIR == "pre_proc_Calibration_needle_phantom_30degsec_table474mm":
    proj_start = 39
    proj_end = 813  # or 814, depending on who you ask
    ref_path = '/home/adriaan/ownCloud3/pre_proc_Brightfield'
else:
    raise Exception()

# postfix of stored claibration
POSTFIX = f'{MAIN_DIR_CALIB}_calibrated_on_25aug2021'
nr_projs = proj_end - proj_start

t_range = range(proj_start, proj_start + nr_projs, 6)  # save GPU memory
x = 50  # offset
t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3)]

# restore calibration
multicam_geom = np.load(f'multicam_geom_{POSTFIX}.npy', allow_pickle=True)

# If the calibration was good, the cameras (without rotating) are here:
hopefully_real_geoms = [c[0] for c in multicam_geom]  # unrotated geoms

# for d1, d2 in zip(multicam_data[3],
#                   xray.xray_multigeom_project(multicam_geom, markers)):
#     xray.plot_projected_markers(d1, d2, det=detector, det_padding=1.2)


if False:
    for test_cam_id in range(1, 4):
        # first make a reco from only one camera
        detector_cropped = crop_detector(detector)
        reco = Reconstruction(PROJS_PATH, detector_cropped.todict())

        geoms_interp = xray.geoms_from_interpolation(
            interpolation_geoms=multicam_geom[test_cam_id - 1],
            interpolation_nrs=t_range,
            interpolation_calibration_nrs=t_annotated,
            plot=False)
        projs = reco.load_sinogram(t_range=t_range, cameras=[test_cam_id],
                                   ref_path=ref_path,
                                   ref_lower_density=True,
                                   ref_mode='static')
        projs = prep_projs(projs, ignore_cols)

        vol_id, vol_geom = astra_reco_rotation_singlecamera(
            reco, projs, geoms_interp, algo='FDK', iters=150, voxels_x=200)
        x = reco.volume(vol_id)
        pq.image(x)
        plt.figure()
        plt.show()
else:
    detector_cropped = cate_astra.crop_detector(detector, 0)
    reco = Reconstruction(PROJS_PATH, detector_cropped.todict())

    all_geoms = []
    all_projs = []
    for cam_id in range(1, 4):
        geoms_interp = geoms_from_interpolation(
            interpolation_geoms=multicam_geom[cam_id - 1],
            interpolation_nrs=t_range,
            interpolation_calibration_nrs=t_annotated,
            plot=False)
        all_geoms.extend(geoms_interp)
        projs = reco.load_sinogram(t_range=t_range, cameras=[cam_id],
                                   ref_path=ref_path,
                                   ref_lower_density=True)
        projs = prep_projs(projs)
        all_projs.append(projs)
    all_projs = np.concatenate(all_projs, axis=1)

    vol_id, vol_geom = astra_reco_rotation_singlecamera(
        reco, all_projs, all_geoms, 'FDK', iters=150)
    x = reco.volume(vol_id)
    print(x.shape)
    pq.image(x)
    plt.figure()
    plt.imshow(x[100, :, :])
    plt.show()

for res_cam_id in range(1, 4):
    projs_annotated = reco.load_sinogram(
        t_range=t_annotated,
        cameras=[res_cam_id])
    projs_annotated = prep_projs(projs_annotated)
    res = astra_residual(reco,
                         projs_annotated, vol_id, vol_geom,
                         multicam_geom[res_cam_id - 1])
    plot_projections(res, title='res')
    plot_projections(projs_annotated, title='projs')
    plot_projections(astra_project(
        reco, vol_id, vol_geom,
        multicam_geom[res_cam_id - 1]), title='reprojs')
    plt.show()

reco.clear()