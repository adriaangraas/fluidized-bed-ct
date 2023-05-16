from cate import annotate, astra
from cate.xray import Geometry, shift, transform
from cate.astra import Detector, crop_detector
from settings import *
from scripts.calibration.util import *

run_annotation = False
run_optimalization = True


class DelftNeedleEntityLocations(annotate.EntityLocations):
    """
    A bit cryptic, but the encoding is based on the 4 vertical sequences
    of needles glued onto the column, from top to bottom:
        BBB = ball ball ball
        BEE = ball eye eye
        EB = eye ball
        B = ball
    Then with `top`, `middle` of `bottom` I encode which of the three needles
    it is, and also provide additional redundant information for convenience.
    Note however that the vertical description does not tell how far up the
    the needle is in the column, only its relative position. There does not
    seem to be good horizontal alignment, so its difficult to segment the
    column in annotated layers.
    """

    ENTITIES = (
        "BBB  top ball stake",
        "BBB  middle ball stake",
        "BBB  bottom ball stake",
        "BEE  top ball drill",
        "BEE  middle eye drill",
        "BEE  bottom eye drill",
        "EB  top eye stake",
        "EB  bottom ball drill",
        "B  ball drill",
    )

    @staticmethod
    def nr_entities():
        return len(DelftNeedleEntityLocations.ENTITIES)

    @staticmethod
    def get_iter():
        return iter(DelftNeedleEntityLocations.ENTITIES)


def triple_camera_circular_geometry(
    source_positions: Sequence,
    detector_positions: Sequence,
    angles: Sequence,
    optimize_rotation=False,
):
    """
    This function retunrns Geometry objects with free parameters, so that
    these can be optimized.

    Parameters:
     - each source and detector has a free position, and detector has RPY
      - with exception from detector 1, this is fixed.
     - one shift and global rotation w.r.t. center of rotation
     - for each angle there is a rotation of the rotation table

    :param source_position:
    :param detector_position:
    :param nr_angles:
    :param angle_start:
    :param angle_stop:
    :return:
    """
    nr_cams = len(source_positions)
    assert nr_cams == len(detector_positions)

    # The first cam needs is totally fixed, to prevent arbitrary shifts
    # in the solution.
    initial_geoms = [
        Geometry(
            source=VectorParameter(source_positions[0]),
            detector=detector_positions[0],
            roll=None,  # are computed automatically, from src and det
            pitch=None,
            yaw=None,
        )
    ]
    for i in range(1, nr_cams):
        initial_geoms.append(
            Geometry(
                source=VectorParameter(source_positions[i]),
                detector=VectorParameter(detector_positions[i]),
                roll=ScalarParameter(None),
                pitch=ScalarParameter(None),
                yaw=ScalarParameter(None),
            )
        )

    # the whole set-up can be arbitrarily shifted and rotated in space
    # but while keeping sources and detectors relatively fixed
    shift_param = VectorParameter(np.array([0.0, 0.0, 0.0]))
    initial_geoms = [shift(g, shift_param) for g in initial_geoms]
    roll_param, pitch_param, yaw_param = [ScalarParameter(0.0) for _ in range(3)]
    initial_geoms = [
        transform(g, roll_param, pitch_param, yaw_param) for g in initial_geoms
    ]

    if optimize_rotation:
        # per initial geom, make a list of angles
        # rotations per angle, the first rotation is not optimizable
        geoms = [[transform(g, yaw=angles[0])] for g in initial_geoms]
        # a different but shared rotation param for each angle
        for i in range(1, len(angles)):
            angle = ScalarParameter(angles[i] - angles[0])
            for cam_angles in geoms:
                cam_angles.append(transform(cam_angles[0], yaw=angle))
    else:
        geoms = []
        for g in initial_geoms:
            ang = [transform(g, yaw=a) for a in angles]
            geoms.append(ang)

    return geoms


detector = Detector(
    DETECTOR_ROWS, DETECTOR_COLS, DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT
)

# data_dir = "/home/adriaan/ownCloud2/"
# data_dir = "/bigstore/adriaan/data/evert/2021-03-03"
# data_dir = "/bigstore/adriaan/data/evert/CWI/2021-03-17 CWI traverse rotate"
data_dir = "/export/scratch2/adriaan/evert/data/2021-08-20"

main_dir = "pre_proc_Calibration_needle_phantom_30degsec_table474mm"
# main_dir = "pre_proc_5deg_sec_360deg_01"
# main_dir = "pre_proc_360deg_5degsec_Cal_phantom_table505mm_offcenter"
# main_dir = "pre_proc_360deg_5degsec_Cal_phantom_table525mm_offcenter_inclined"
projs_path = f"{data_dir}/{main_dir}"
ref_path = "/home/adriaan/ownCloud3/pre_proc_Brightfield"
vabs = 100


def reconstruction(detector):
    return Reconstruction(
        projs_path,
        detector.todict(),
        expected_voxel_size_x=APPROX_VOXEL_WIDTH,
        expected_voxel_size_z=APPROX_VOXEL_HEIGHT,
    )


if main_dir == "pre_proc_Calibration_needle_phantom_30degsec_table474mm":
    # first frame before motion
    # 31-32 shows a very tiny bit of motion, but seems insignificant
    proj_start = 33
    # final state frame, img 806 equals 32, so the range should be without 806
    proj_end = 806
    nr_projs = proj_end - proj_start  # 773
    x = 50
    t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3)]
    ignore_cols = 0  # det width used is 550
elif main_dir == "pre_proc_Calibration_needle_phantom_30degsec_table534mm":
    # first frame before motion
    # 31-32 shows a very tiny bit of motion, but seems insignificant
    proj_start = 39
    proj_end = 813  # or 814, depending on who you ask
    nr_projs = proj_end - proj_start  # 774
    x = 50
    t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3)]
    ignore_cols = 0  # det width used is 550
else:
    raise Exception

postfix = f"{main_dir}_calibrated_on_26aug2021"

t_range = range(proj_start, proj_start + nr_projs, 6)
for t in t_annotated:
    assert proj_start <= t < proj_end, f"{t} is not within proj start-end."

# launch the annotation tool
multicam_data = annotated_data(
    projs_path,
    t_annotated,  # any frame will probably do
    DelftNeedleEntityLocations,
    fname=main_dir,
    cameras=[1, 2, 3],
    # fulls_path=fulls_path,
    open_annotator=run_annotation,
    vmin=6.0,
    vmax=10.0,
)
astra.pixels2coords(multicam_data, detector)

# geoms, params = xray.circular_geometry(
#     source_pos, det_pos, nr_angles=len(t_annotated),
#     parametrization='rotation_from_init')
pre_geoms = triangle_geom(SOURCE_RADIUS, DETECTOR_RADIUS, rotation=False, shift=False)
srcs = [g.source for g in pre_geoms]
dets = [g._detector for g in pre_geoms]
angles = (np.array(t_annotated) - proj_start) / nr_projs * 2 * np.pi
multicam_geom = triple_camera_circular_geometry(srcs, dets, angles=angles)

# geoms_interp = xray.geoms_from_interpolation(
#     interpolation_geoms=multicam_geom,
#     interpolation_nrs=t_range,
#     interpolation_calibration_nrs=t_annotated,
#     plot=False)
#
# projs = reco.load_sinogram(t_range=t_range, cameras=[3])
# projs = np.squeeze(projs)
# projs = np.swapaxes(projs, 0, 1)
# projs = np.ascontiguousarray(projs)
#
# vol_id, vol_geom = astra_reco_rotation_singlecamera(
#     reco, projs, detector, geoms_interp, algo='FDK')
# x = reco.volume(vol_id)
# pq.image(x)
#
# projs_annotated = reco.load_sinogram(
#     t_range=range(t_annotated[0], t_annotated[0] + 1), cameras=[3])
# projs_annotated = np.squeeze(projs_annotated, axis=0)
# projs_annotated = np.swapaxes(projs_annotated, 0, 1)
# projs_annotated = np.ascontiguousarray(projs_annotated)
# geoms_annotated = multicam_geom[0:1]
# res = astra_residual(
#     reco, projs_annotated, detector, vol_id, vol_geom, geoms_annotated)
# plot_residual(res, title='before')
# astra.clear()
# plt.figure()
# plt.show()

if run_optimalization:  # optimize, or use existing optimization?
    multicam_geom_flat = [g for c in multicam_geom for g in c]
    multicam_data_flat = [d for c in multicam_data.values() for d in c]
    markers = run_initial_marker_optimization(
        multicam_geom_flat, multicam_data_flat, nr_iters=2, plot=True, max_nfev=10
    )
    np.save(f"markers_{postfix}.npy", markers)

    # calib (export format)
    rotation_0_geoms = {}
    for key, val in zip(multicam_data.keys(), multicam_geom):
        rotation_0_geoms[key] = val[0]._g.asstatic()
    np.save(f"geom_{postfix}.npy", [rotation_0_geoms])
    print("Optimalization saved.")
else:
    # calibration = np.load(f'calibration_{postfix}.npy', allow_pickle=True)
    markers = np.load(f"markers_{postfix}.npy", allow_pickle=True)
    multicam_geom = np.load(f"geom_{postfix}.npy", allow_pickle=True)

# for d1, d2 in zip(multicam_data[3],
#                   xray.xray_multigeom_project(multicam_geom, markers)):
#     xray.plot_projected_markers(d1, d2, det=detector, det_padding=1.2)

detector_cropped = crop_detector(detector, ignore_cols)
reco = reconstruction(detector_cropped)

all_geoms = []
all_projs = []
for cam_id in range(1, 4):
    geoms_interp = xray.geoms_from_interpolation(
        interpolation_geoms=multicam_geom[cam_id - 1],
        interpolation_nrs=t_range,
        interpolation_calibration_nrs=t_annotated,
        plot=False,
    )
    all_geoms.extend(geoms_interp)

    projs = reco.load_sinogram(t_range=t_range, cameras=[cam_id])
    projs = prep_projs(projs, ignore_cols)
    all_projs.append(projs)

all_projs = np.concatenate(all_projs, axis=1)
vol_id, vol_geom = astra_reco_rotation_singlecamera(
    reco, all_projs, all_geoms, algo="FDK", iters=150
)
x = reco.volume(vol_id)
import pyqtgraph as pq

pq.image(x)
import matplotlib.pyplot as plt

plt.figure()
plt.show()

# for res_cam_id in range(1, 4):
#     projs_annotated = reco.load_sinogram(
#         t_range=t_annotated, cameras=[res_cam_id])
#     projs_annotated = prep_projs(projs_annotated, ignore_cols)
#
#     # res = astra_residual(
#     #     reco, projs_annotated, detector, vol_id, vol_geom, geoms_annotated)
#     res = astra_residual(reco, projs_annotated, vol_id,
#                          vol_geom, multicam_geom[res_cam_id - 1])
#     plot_projections(res, title='res')
#     plot_projections(projs_annotated, title='projs')
#     plot_projections(
#         astra_project(reco, vol_id, vol_geom,
#                       multicam_geom[res_cam_id - 1]), title='reprojs')
#     plt.show()
#
# reco.clear()
