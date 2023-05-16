"""
Make a test reconstruction when a calibration has been obtained.
"""
import matplotlib.pyplot as plt
import pyqtgraph as pq

from cate.xray import crop_detector
from scripts.calib.util import *

SOURCE_RADIUS = 94.5
DETECTOR_RADIUS = 27.0
DETECTOR_ROWS = 1548  # including ROI
DETECTOR_COLS = 550  # including ROI
DETECTOR_COLS_SPEC = 1524  # also those outside ROI

DETECTOR_WIDTH_SPEC = 30.2  # cm, also outside ROI
DETECTOR_HEIGHT = 30.7  # cm, also outside ROI
DETECTOR_WIDTH = DETECTOR_WIDTH_SPEC / DETECTOR_COLS_SPEC * DETECTOR_COLS  # cm

DETECTOR_PIXEL_WIDTH = DETECTOR_WIDTH / DETECTOR_COLS
DETECTOR_PIXEL_HEIGHT = DETECTOR_HEIGHT / DETECTOR_ROWS
APPROX_VOXEL_WIDTH = DETECTOR_PIXEL_WIDTH / (
    SOURCE_RADIUS + DETECTOR_RADIUS) * SOURCE_RADIUS
APPROX_VOXEL_HEIGHT = DETECTOR_PIXEL_HEIGHT / (
    SOURCE_RADIUS + DETECTOR_RADIUS) * SOURCE_RADIUS

detector = xray.Detector(DETECTOR_ROWS, DETECTOR_COLS,
                         DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT)

# directory of the calibration scan
data_dir_calib = "/home/adriaan/ownCloud2/"
main_dir_calib = "pre_proc_Calibration_needle_phantom_30degsec_table474mm"

# directory of a different scan to reconstruct
# data_dir = "/home/adriaan/ownCloud3/"
# main_dir = "pre_proc_3x10mm_foamballs_vertical_01"
data_dir = "/home/adriaan/ownCloud2/"
main_dir = "pre_proc_Calibration_needle_phantom_30degsec_table474mm"

# compute projections
projs_path = f'{data_dir}/{main_dir}'
vabs = 100

# set up how to reconstruct
def reconstruction(detector):
    return Reconstruction(projs_path,
                          detector.todict(),
                          expected_voxel_size_x=APPROX_VOXEL_WIDTH,
                          expected_voxel_size_z=APPROX_VOXEL_HEIGHT)

# configure which projection range to take
if main_dir == "pre_proc_3x10mm_foamballs_vertical_01":
    proj_start = 37
    proj_end = 1621
    ref_path = '/home/adriaan/ownCloud3/pre_proc_Full_30degsec_03'
elif main_dir == "pre_proc_Calibration_needle_phantom_30degsec_table474mm":
    proj_start = 39
    proj_end = 813  # or 814, depending on who you ask
    ref_path = '/home/adriaan/ownCloud3/pre_proc_Brightfield'
else:
    raise Exception

ignore_cols = 0  # do not use the cols in this data
postfix = f'{main_dir_calib}_calibrated_on_25aug2021'

nr_projs = proj_end - proj_start
t_range = range(proj_start, proj_start + nr_projs, 6)  # subsample for GPU memory

x = 50  # offset
t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3)]

# restore calibration
multicam_geom = np.load(f'multicam_geom_{postfix}.npy', allow_pickle=True)

# If the calibration was good, the cameras (without rotating) are here:
hopefully_real_geoms = [c[0] for c in multicam_geom]  # unrotated geoms

# for d1, d2 in zip(multicam_data[3],
#                   xray.xray_multigeom_project(multicam_geom, markers)):
#     xray.plot_projected_markers(d1, d2, det=detector, det_padding=1.2)


if False:
    for test_cam_id in range(1, 4):
        # first make a reco from only one camera
        detector_cropped = crop_detector(detector, ignore_cols)
        reco = reconstruction(detector_cropped)

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
    detector_cropped = crop_detector(detector, ignore_cols)
    reco = reconstruction(detector_cropped)

    all_geoms = []
    all_projs = []
    for cam_id in range(1, 4):
        geoms_interp = xray.geoms_from_interpolation(
            interpolation_geoms=multicam_geom[cam_id - 1],
            interpolation_nrs=t_range,
            interpolation_calibration_nrs=t_annotated,
            plot=False)
        all_geoms.extend(geoms_interp)
        projs = reco.load_sinogram(t_range=t_range, cameras=[cam_id],
                                   ref_path=ref_path,
                                   ref_lower_density=True,
                                   ref_mode='static')
        projs = prep_projs(projs, ignore_cols)
        all_projs.append(projs)

    all_projs = np.concatenate(all_projs, axis=1)

    vol_id, vol_geom = astra_reco_rotation_singlecamera(
        reco, all_projs, all_geoms, algo='FDK', iters=150)
    x = reco.volume(vol_id)
    print(x.shape)
    # np.save('5degsec_180321.npy', x)
    # print('saved')
    pq.image(x)
    plt.figure()
    plt.imshow(x[100, :, :])
    plt.show()


for res_cam_id in range(1, 4):
    projs_annotated = reco.load_sinogram(
        t_range=t_annotated, cameras=[res_cam_id])
    projs_annotated = prep_projs(projs_annotated, ignore_cols)

    # res = astra_residual(
    #     reco, projs_annotated, detector, vol_id, vol_geom, geoms_annotated)
    res = astra_residual(reco, projs_annotated, vol_id,
                         vol_geom, multicam_geom[res_cam_id - 1])
    plot_projections(res, title='res')
    plot_projections(projs_annotated, title='projs')
    plot_projections(
        astra_project(reco, vol_id, vol_geom,
                      multicam_geom[res_cam_id - 1]), title='reprojs')
    plt.show()

reco.clear()