import copy

from cate.xray import StaticGeometry, XrayOptimizationProblem, \
    crop_detector, shift, transform
from cate import annotate, astra
from scripts.calib.util import *
from settings import *
from util import run_initial_marker_optimization

run_annotation = True
run_optimalization = True


class MetalPieces(annotate.EntityLocations):
    ENTITIES = (
        'High',
        'Mid',
        'Low',
    )

    @staticmethod
    def nr_entities():
        return len(MetalPieces.ENTITIES)

    @staticmethod
    def get_iter():
        return iter(MetalPieces.ENTITIES)


def load_previous_geometry(fname):
    multicam_geom = np.load(fname, allow_pickle=True)[0]
    geoms_all_cams = []
    detector = xray.Detector(DETECTOR_ROWS, DETECTOR_COLS,
                             DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT)

    for cam in self.cameras:
        geoms = []
        ang_increment = 2 * np.pi / self.nr_projs
        for i in range(self.nr_projs):
            g = xray.transform(multicam_geom[cam],
                               yaw=i * ang_increment)
            v = xray.geom2astravec(g, detector.todict())
            geoms.append(v)

        geoms_all_cams.append(geoms)

    return geoms_all_cams


def amended_geometry(
    previous_geometry: dict,
    angles: Sequence
):
    """
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
    # the whole set-up can be arbitrarily shifted and rotated in space
    # but while keeping sources and detectors relatively fixed
    shift_param = VectorParameter(np.array([0., 0., 0.]))
    roll_param, pitch_param, yaw_param = [ScalarParameter(0.) for _ in
                                          range(3)]

    amended = {}
    for cam, geom in previous_geometry.items():
        geom = shift(geom, shift_param)
        geom = transform(geom, roll_param, pitch_param, yaw_param)
        geoms_angles = [transform(geom, yaw=a) for a in angles]
        amended[cam] = geoms_angles

    return amended


detector = xray.Detector(DETECTOR_ROWS, DETECTOR_COLS,
                         DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT)

data_dir = "/home/adriaan/data/evert/2021-08-24"
main_dir = "pre_proc_3x10mm_foamballs_vertical_wall"
projs_path = f'{data_dir}/{main_dir}'

if main_dir == "pre_proc_3x10mm_foamballs_vertical_wall":
    proj_start = 1025
    proj_end = 1799
    nr_projs = proj_end - proj_start
    x = 1025
    t_annotated = [x, int(x + nr_projs / 3), int(x + 2 * nr_projs / 3 + 20)]
    ignore_cols = 0  # det width used is 550
    cameras = (1, 2)
else:
    raise Exception

postfix = f"table474mm_26aug2021_amend_{main_dir}_31aug2021"

for t in t_annotated:
    assert proj_start <= t < proj_end, f"{t} is not within proj start-end."

data = annotated_data(projs_path,
                      t_annotated,  # any frame will probably do
                      MetalPieces,
                      fname=main_dir,
                      cameras=cameras,
                      open_annotator=run_annotation,
                      vmin=8., vmax=9.)
astra.pixels2coords(data, detector)

# first load the previous geom that we want to amend
prev_geom = np.load('geom_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
                    allow_pickle=True)[0]  # type: dict

# then "amend' the geometry, that is:
# - for each cam add a new shared shift and shared rotation parameter
# - add new rotation angle list, for calibration (not for export)
angles = (np.array(t_annotated) - proj_start) / nr_projs * 2 * np.pi
amended_geom = amended_geometry(prev_geom, angles=angles)

# remove cams from the geom that are not in the annotated data
amended_geom_stripped = {}
for cam, geom in amended_geom.items():
    if cam in data:
        amended_geom_stripped[cam] = geom

if run_optimalization:  # optimize, or use existing optimization?
    amended_geom_flat = []
    for _, geoms_per_cam in sorted(amended_geom_stripped.items()):
        amended_geom_flat.extend(geoms_per_cam)

    data_flat = [d for c in data.values() for d in c]
    markers = run_initial_marker_optimization(
        amended_geom_flat,
        data_flat,
        nr_iters=2, plot=False, max_nfev=10)

    rotation_0_geoms = {}
    for cam, geom in amended_geom.items():
        rotation_0_geoms[cam] = geom[0]._g.asstatic()

    np.save(f'geom_{postfix}.npy', [rotation_0_geoms])
    print('Optimalization saved.')
else:
    # calibration = np.load(f'calibration_{postfix}.npy', allow_pickle=True)
    amended_geom = np.load(f'geom_{postfix}.npy', allow_pickle=True)

# for d1, d2 in zip(multicam_data[3],
#                   xray.xray_multigeom_project(multicam_geom, markers)):
#     xray.plot_projected_markers(d1, d2, det=detector, det_padding=1.2)

def reconstruction(detector):
    return Reconstruction(projs_path,
                          detector.todict(),
                          expected_voxel_size_x=APPROX_VOXEL_WIDTH,
                          expected_voxel_size_z=APPROX_VOXEL_HEIGHT)


detector_cropped = crop_detector(detector, ignore_cols)
reco = reconstruction(detector_cropped)
t_range = range(proj_start, proj_start + nr_projs, 6)

all_geoms = []
all_projs = []
for cam_id in range(1, 4):
    geoms_interp = xray.geoms_from_interpolation(
        interpolation_geoms=amended_geom[cam_id - 1],
        interpolation_nrs=t_range,
        interpolation_calibration_nrs=t_annotated,
        plot=False)
    all_geoms.extend(geoms_interp)

    projs = reco.load_sinogram(t_range=t_range, cameras=[cam_id])
    projs = prep_projs(projs, ignore_cols)
    all_projs.append(projs)

all_projs = np.concatenate(all_projs, axis=1)
vol_id, vol_geom = astra_reco_rotation_singlecamera(
    reco, all_projs, all_geoms, algo='FDK', iters=150)
x = reco.volume(vol_id)
import pyqtgraph as pq
pq.image(x)
import matplotlib.pyplot as plt
plt.figure()
plt.show()