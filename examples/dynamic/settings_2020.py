import numpy as np

# in the 2020 dataset the amount of COLS changed to 400
# TODO(Adriaan) add dimension checking to the loader?
SCANS = {
    '2020-02-12': [{'projs_dir': 'pre_proc_20mm_ball_62mmsec_03',
                    'fulls_dir': 'pre_proc_Full_11_6lmin'}],
    '2020-02-17': [{'projs_dir': p,
                    'fulls_dir': 'pre_proc_Full'}
                   for p in [
                       'pre_proc_10_20mm_phantoms_perpendicular',
                       'pre_proc_10mm_phantom_center',
                       'pre_proc_20mm_phantom_center',
                       'pre_proc_20mm_phantom_wall',
                       'pre_proc_10_20mm_phantoms_parallel',
                       'pre_proc_10_20mm_phantoms_perpendicular_62mmsec',
                       'pre_proc_10mm_phantom_wall',
                       'pre_proc_20mm_phantom_side']],
    '2020-02-18': [{'projs_dir': p,
                    'fulls_dir': 'pre_proc_Full'}
                   for p in [
                       'pre_proc_10mm_ball_250mmsec_side',
                       'pre_proc_10mm_ball_62mmsec_wall',
                       'pre_proc_20mm_ball_250mmsec_side',
                       'pre_proc_24lmin',
                       'pre_proc_10mm_ball_125mmsec_center',
                       'pre_proc_10mm_ball_62mmsec_center',
                       'pre_proc_20mm_ball_125mmsec_ side',
                       'pre_proc_22lmin']]
}

SOURCE_RADIUS = 93.7
DETECTOR_RADIUS = 53.4

DETECTOR_ROWS = 1548
DETECTOR_COLS = 400

DETECTOR_WIDTH = 30.0 / 1524 * DETECTOR_COLS  # cm
DETECTOR_HEIGHT = 30.0  # cm

# auxiliaries
DETECTOR_PIXEL_WIDTH = DETECTOR_WIDTH / DETECTOR_COLS
DETECTOR_PIXEL_HEIGHT = DETECTOR_HEIGHT / DETECTOR_ROWS
DETECTOR_SIZE = np.array([DETECTOR_WIDTH, DETECTOR_HEIGHT])
DETECTOR_SHAPE = np.array([DETECTOR_ROWS, DETECTOR_COLS])
