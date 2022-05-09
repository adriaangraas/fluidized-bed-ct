import os
from pathlib import Path

from fbrct.scan import (
    StaticScan,
    Phantom,
    FluidizedBedScan,
    TraverseScan,
    MovingPhantom,
)

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
APPROX_VOXEL_WIDTH = (
    DETECTOR_PIXEL_WIDTH / (SOURCE_RADIUS + DETECTOR_RADIUS) * SOURCE_RADIUS
)
APPROX_VOXEL_HEIGHT = (
    DETECTOR_PIXEL_HEIGHT / (SOURCE_RADIUS + DETECTOR_RADIUS) * SOURCE_RADIUS
)

detector = {
    "rows": DETECTOR_ROWS,
    "cols": DETECTOR_COLS,
    "pixel_width": DETECTOR_PIXEL_WIDTH,
    "pixel_height": DETECTOR_PIXEL_HEIGHT,
}

data_dir = os.getenv("DATA_DIR", "/export/scratch3/adriaan/evert/data")
data_dir_19 = f"{data_dir}/2021-08-19"
data_dir_20 = f"{data_dir}/2021-08-20"
data_dir_23 = f"{data_dir}/2021-08-23"
data_dir_24 = f"{data_dir}/2021-08-24"

calib_dir = str(Path(__file__).parent / "calibration")
calib = f"{calib_dir}/geom_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy"

SCANS = []
ball_20mm = Phantom(2, None)
ball_10mm = Phantom(1, None)


def get_scans(name: str) -> list:
    selected = []
    for s in SCANS:
        if s.name == name:
            selected.append(s)

    return selected


###############################################################################
# Modifications for testing on Scan3
###############################################################################
# scan = FluidizedBedScan(
#     "2021-08-24_10mm_23mm_horizontal",
#     f'{data_dir_24}/pre_proc_10mm_23mm_foamballs_horizontal',
#     ref_dir=f'{data_dir_23}/pre_proc_Full_30degsec',
#     ref_ran=range(1, t-9), # for this scan always one more than timeframes
#     # ref_ran=range(t-10, t-9), # for this scan always one more than timeframes
#     # ref_ran=range(1, 2), # for this scan always one more than timeframes
#     ref_normalization_dir=f'{data_dir_23}/pre_proc_Empty_30degsec',
#     ref_normalization_ran=range(1, t-9),
#     # darks_dir=f'{data_dir}/pre_proc_Dark',
#     geometry=f'{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#     geometry_scaling_factor=1. / 1.0106333,
#     cameras=(1, 2, 3),
#     ref_lower_density=False,
#     liter_per_min=False,
#     col_inner_diameter=5.0,
#     timeframes=range(t, t+1),
# )
# SCANS.append(scan)
#
#
# # to test ASTRA vs Rayve
# scan = FluidizedBedScan(
#     "2021-08-24_10mm_23mm_horizontal_simulation_firstangle",
#     f'{data_dir_24}/pre_proc_10mm_23mm_foamballs_horizontal',
#     ref_dir=f'{data_dir_24}/pre_proc_Full_30degsec',
#     ref_ran=range(1, 3),
#     # ref_normalization_dir=f'{data_dir_24}/pre_proc_Empty_30degsec',
#     # ref_normalization_ran=range(1, 100),
#     # darks_dir=f'{data_dir}/pre_proc_Dark',
#     geometry='resources/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#     geometry_scaling_factor=1. / 1.0106333,
#     cameras=(1, 2, 3),
#     ref_lower_density=False,
#     liter_per_min=False,
#     col_inner_diameter=5.0,
#     timeframes=range(1, 1000, 200),
# )
# # SCANS.append(scan)
#
# # reconstruct bed to see influence of noise
# scan = StaticScan(
#     "2021-08-24_10mm_14mm_23mm_horizontal_refempty",
#     f'{data_dir_24}/pre_proc_10mm_14mm_23mm_foamballs_horizontal',
#     proj_start=1025,
#     proj_end=1799,
#     ref_dir=f'{data_dir_24}/pre_proc_Empty_30degsec',
#     ref_start=1026,
#     # ref_ran=range(1, 1000),
#     # darks_dir=f'{data_dir}/pre_proc_Dark',
#     geometry=f'{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#     geometry_scaling_factor=1. / 1.0106333,
#     cameras=(1, 2, 3),
#     ref_lower_density=True,
#     # ref_mode='static',
# )
# SCANS.append(scan)
#
# # show artifacts on real data, in this rotational scan we have only one
# # empty and full image per projection, so noise level is high
# for t in range(1025, 1799, 100):
#     scan = FluidizedBedScan(
#         "2021-08-23_10mm_14mm_23mm_horizontal_simulation",
#         f'{data_dir_24}/pre_proc_10mm_14mm_23mm_foamballs_horizontal',
#         ref_dir=f'{data_dir_23}/pre_proc_Full_30degsec',
#         ref_ran=range(t+1, t+2), # for this scan always one more than timeframes
#         ref_normalization_dir=f'{data_dir_23}/pre_proc_Empty_30degsec',
#         ref_normalization_ran=range(t, t+1),
#         # darks_dir=f'{data_dir}/pre_proc_Dark',
#         geometry='resources/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#         geometry_scaling_factor=1. / 1.0106333,
#         cameras=(1, 2, 3),
#         ref_lower_density=False,
#         liter_per_min=False,
#         col_inner_diameter=5.0,
#         timeframes=range(t, t+1),
#     )
#     # SCANS.append(scan)
#
# # show artifacts on real data, in this rotational scan we have only one
# # empty and full image per projection, so noise level is high
# for t in range(1025, 1799, 100):
#     scan = FluidizedBedScan(
#         "2021-08-24_10mm_23mm_horizontal_simulation",
#         f'{data_dir_24}/pre_proc_10mm_23mm_foamballs_horizontal',
#         ref_dir=f'{data_dir_24}/pre_proc_Full_30degsec',
#         ref_ran=range(t+1, t+2), # for this scan always one more than timeframes
#         ref_normalization_dir=f'{data_dir_24}/pre_proc_Empty_30degsec',
#         ref_normalization_ran=range(t, t+1),
#         # darks_dir=f'{data_dir}/pre_proc_Dark',
#         geometry='resources/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#         geometry_scaling_factor=1. / 1.0106333,
#         cameras=(1, 2, 3),
#         ref_lower_density=False,
#         liter_per_min=False,
#         col_inner_diameter=5.0,
#         timeframes=range(t, t+1),
#     )
#     # SCANS.append(scan)
#
#
# # in this simulation of a fluidized bed we remain in the first angle,
# # so we have many full/empties for referencing and the noise level should
# # be relatively low/comparable to real fluidized bed scans
# scan = FluidizedBedScan(
#     "2021-08-24_10mm_23mm_horizontal_simulation_firstangle",
#     f'{data_dir_24}/pre_proc_10mm_23mm_foamballs_horizontal',
#     ref_dir=f'{data_dir_24}/pre_proc_Full_30degsec',
#     ref_ran=range(1, 1000),
#     ref_normalization_dir=f'{data_dir_24}/pre_proc_Empty_30degsec',
#     ref_normalization_ran=range(1, 100),
#     # darks_dir=f'{data_dir}/pre_proc_Dark',
#     geometry='resources/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#     geometry_scaling_factor=1. / 1.0106333,
#     cameras=(1, 2, 3),
#     ref_lower_density=False,
#     liter_per_min=False,
#     col_inner_diameter=5.0,
#     timeframes=range(1, 1000, 200),
# )
# # SCANS.append(scan)
#
#
# # looks okay with negligable artifacts, from the 474mm calibration
# scan = StaticScan(
#     "Full col",
#     f'{data_dir_23}/pre_proc_Full_30degsec',
#     proj_start=1026,
#     proj_end=1802,
#     # ref_dir=f'{data_dir_19}/pre_proc_Brightfield',
#     # ref_start=f'{data_dir_19}/pre_proc_Brightfield',
#     geometry=f'{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#     # geometry=calib,
#     cameras=(1, 2, 3),
#     ref_dir=f'{data_dir_23}/pre_proc_Empty_30degsec',
#     ref_start=1026,
#     ref_lower_density=True
# )
# # SCANS.append(scan)
#
# # looks okay with negligable artifacts, from the 474mm calibration
# scan = StaticScan(
#     "Calibration_needle_phantom_30degsec_table534mm",
#     f'{data_dir_20}/pre_proc_Calibration_needle_phantom_30degsec_table534mm',
#     proj_start=39,
#     proj_end=813,
#     ref_dir=f'{data_dir_19}/pre_proc_Brightfield',
#     # darks_dir=f'{data_dir}/pre_proc_Dark_frames',
#     geometry=calib,
#     # geometry_manual=True,
#     # cameras=(1, 2, 3),
#     cameras=(1,),
#     ref_lower_density=True,
#     geometry_scaling_factor=1. / 1.012447804,
# )
# # SCANS.append(scan)
#
# scan = StaticScan(
#     "Calibration_needle_phantom_30degsec_table474mm",
#     f'{data_dir_20}/pre_proc_Calibration_needle_phantom_30degsec_table474mm',
#     proj_start=33,
#     proj_end=806,
#     ref_dir=f'{data_dir_19}/pre_proc_Brightfield',
#     darks_dir=f'{data_dir_19}/pre_proc_Dark_frames',
#     geometry=calib,
#     cameras=(1, 2, 3),
#     ref_lower_density=True
# )
# # SCANS.append(scan)
#
# for size in ['10mm']:
#     scan = StaticScan(
#         f"2021-08-23_3x{size}_foamballs_vertical_refempty",
#         f'{data_dir_23}/pre_proc_3x{size}_foamballs_vertical',
#         proj_start=1025,
#         proj_end=1801,
#         ref_dir=f'{data_dir_23}/pre_proc_Empty_30degsec',
#         ref_start=1026,
#         # darks_dir=f'{data_dir}/pre_proc_Darkfield',
#         geometry=f'{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#         cameras=(1, 2, 3),
#         ref_lower_density=True,
#         geometry_scaling_factor=1. / 1.0106333
#     )
#     SCANS.append(scan)
#

#
# def mask_scan():
#     mask = Mask(
#         "mask_pre_proc_Full_30degsec",
#         f"{data_dir_23}/pre_proc_Full_30degsec",
#         mask_min=.05,  # this value is nr.projs. dependent
#         proj_start=1026,
#         proj_end=1802,  # 1802
#         ref_dir=f"{data_dir_23}/pre_proc_Empty_30degsec",
#         ref_start=1026,
#         cameras=(1,),
#         geometry=calib,
#         ref_lower_density=True,
#     )
#     return mask
#
#
# for size in ['23mm', '14mm', '10mm']:
#     scan = FluidizedBedScan(
#         f"2021-08-23_3x{size}_foamballs",
#         f'{data_dir_23}/pre_proc_3x{size}_foamballs_vertical',
#         ref_dir=f'{data_dir_23}/pre_proc_Full_30degsec',
#         ref_ran=range(1, 100),
#         ref_normalization_dir=f'{data_dir_23}/pre_proc_Empty_30degsec',
#         ref_normalization_ran=range(1, 100),
#         # geometry=calib,
#         # geometry_manual=True,
#         geometry=f'{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#         geometry_scaling_factor=1. / 1.0106333,
#         cameras=(1, 2, 3),
#         ref_lower_density=False,
#         liter_per_min=False,
#         timeframes=range(100, 101),
#         col_inner_diameter=5.0,
#     )
#     # SCANS.append(scan)
#
# for lmin in [25]:
#     scan = FluidizedBedScan(
#         f"2021-08-23_{lmin}Lmin_reffull",
#         f'{data_dir_23}/pre_proc_{lmin}Lmin',
#         liter_per_min=lmin,
#         ref_dir=f'{data_dir_23}/pre_proc_Full_30degsec',
#         ref_ran=range(1, 100),
#         ref_normalization_dir=f'{data_dir_23}/pre_proc_Empty_30degsec',
#         ref_normalization_ran=range(1, 100),
#         # darks_dir=f'{data_dir}/pre_proc_Darkfield',
#         # geometry=calib,
#         geometry=f'{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#         cameras=(1, 2, 3),
#         ref_lower_density=False,
#         # mask_scan=mask_scan(),
#         timeframes=range(411, 412),  # nice "bubble" in the top
#         col_inner_diameter=5.0
#     )
#     # SCANS.append(scan)
#
# for lmin in [25]:
#     scan = FluidizedBedScan(
#         f"2021-08-23_{lmin}Lmin",
#         f'{data_dir_23}/pre_proc_{lmin}Lmin',
#         liter_per_min=lmin,
#         # ref_dir=f'{data_dir}/pre_proc_Empty_30degsec',
#         # ref_ran=range(1, 100),
#         # ref_dir=f'{data_dir}/pre_proc_{lmin}Lmin',
#         # ref_ran=range(1000, 1299),
#         # darks_dir=f'{data_dir}/pre_proc_Darkfield',
#         ref_dir=f'{data_dir_23}/pre_proc_Full_30degsec',
#         ref_ran=range(1, 10),
#         geometry=calib,
#         geometry_scaling_factor=1. / 1.012447804,
#         # geometry=f'{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy',
#         cameras=(1, 2, 3),
#         ref_lower_density=False,
#         # mask_scan=mask_scan()
#         timeframes=range(1100, 1101),  # nice "bubble" in the top
#         col_inner_diameter=5.0,
#         ref_normalization_dir=f'{data_dir_23}/pre_proc_Empty_30degsec',
#         ref_normalization_ran=range(1, 100),
#     )
#     SCANS.append(scan)

###############################################################################
# Scans on 19 aug 2021
###############################################################################
ref_dir = (f"/export/scratch2/adriaan/evert/data/2021-08-19/pre_proc_Empty_30degsec",)
ref_ran = (range(1, 100),)

_19_empty_rotating = StaticScan(
    "2021-08-19_pre_proc_Empty_30degsec",
    detector,
    f"{data_dir_19}/pre_proc_Empty_30degsec",
    proj_start=1,
    proj_end=100,
    is_full=False,
    is_rotational=True,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)

###############################################################################
# Scans on 20 aug 2021
###############################################################################
# # looks okay with negligable artifacts, from the 474mm calibration
# scan = StaticScan(
#     "Calibration_needle_phantom_30degsec_table534mm",
#     f'{data_dir_23}/pre_proc_Calibration_needle_phantom_30degsec_table534mm',
#     proj_start=39,
#     proj_end=813,
#     ref_dir=f'{data_dir_23}/pre_proc_Brightfield',
#     # darks_dir=f'{data_dir}/pre_proc_Dark_frames',
#     geometry=f'{calib_dir}/resources/geom_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
#     ref_lower_density=True
# )
# # SCANS.append(scan)
#
# scan = StaticScan(
#     "Calibration_needle_phantom_30degsec_table474mm",
#     '/home/adriaan/ownCloud2/pre_proc_Calibration_needle_phantom_30degsec_table474mm',
#     proj_start=33,
#     proj_end=806,
#     ref_dir='/home/adriaan/ownCloud3/pre_proc_Brightfield',
#     darks_dir='/home/adriaan/ownCloud3/pre_proc_Dark_frames',
#     geometry='resources/multicam_rot0_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
#     ref_lower_density=True
# )
# # SCANS.append(scan)
#
# scan = StaticScan(
#     "Calibration_needle_phantom_30degsec_table534mm",
#     '/home/adriaan/ownCloud2/pre_proc_Calibration_needle_phantom_30degsec_table534mm',
#     proj_start=39,
#     proj_end=813,
#     ref_dir='/home/adriaan/ownCloud3/pre_proc_Brightfield',
#     darks_dir='/home/adriaan/ownCloud3/pre_proc_Dark_frames',
#     geometry='resources/multicam_rot0_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
#     ref_lower_density=True
# )
# # SCANS.append(scan)
#

size_of_marble = 2.5  # cm
for mmsec, ran in [(62, range(100, 200)), (125, range(40, 100))]:
    for pos in ["center", "wall_det2", "wall_source2"]:
        for scaling in ["", "_noscaling"]:
            _scan = TraverseScan(
                f"Small_marble_{pos}_{mmsec}mmsec_65Hz{scaling}",
                detector,
                f"{data_dir_20}/pre_proc_Small_marble_{pos}_{mmsec}mmsec_65Hz",
                motor_velocity=mmsec,
                # darks_dir='/home/adriaan/ownCloud3/pre_proc_Dark_frames',
                # geometry='resources/multicam_rot0_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
                # geometry=calib,
                geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
                geometry_scaling_factor=1.0 / 1.0106333 if scaling != "" else None,
                framerate=65,
                timeframes=range(ran.start - 10, ran.stop + 10),
                references=[_19_empty_rotating]
                # col_inner_diameter=5.0,
            )
            _scan.add_phantom(MovingPhantom(size_of_marble, interesting_time=ran))
            SCANS.append(_scan)

# size_of_marble = 2.5  # cm
# for mmsec, ran in [
#     (62, range(100, 200)),
#     (256, range(40, 100))]:
#     for pos in ['center', 'wall_det2', 'wall_source2']:
#         scan = TraverseScan(
#             f'Small_marble_{pos}_{mmsec}mmsec_65Hz',
#             f'{data_dir_23}/pre_proc_Small_marble_{pos}_{mmsec}mmsec_65Hz',
#             motor_velocity=mmsec,
#             ref_dir=f'{data_dir_19}/pre_proc_Empty_30degsec',
#             # darks_dir='/home/adriaan/ownCloud3/pre_proc_Dark_frames',
#             geometry='resources/multicam_rot0_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy',
#             framerate=65,
#             ref_lower_density=True)
#         scan.add_phantom(MovingPhantom(size_of_marble, interesting_time=ran))
#         # SCANS.append(scan)

###############################################################################
# Scans on 23 aug 2021
###############################################################################
_23_full_rotating = StaticScan(
    "2021-08-23_pre_proc_Full_30degsec",
    detector,
    f"{data_dir_23}/pre_proc_Full_30degsec",
    proj_start=1026,
    proj_end=1800,
    is_full=True,
    is_rotational=True,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_23_full_rotating)

_23_full_static = StaticScan(
    "2021-08-23_pre_proc_Full",
    detector,
    f"{data_dir_23}/pre_proc_Full_30degsec",
    proj_start=10,
    proj_end=1050,
    is_full=True,
    is_rotational=False,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_23_full_static)

_23_empty_static = StaticScan(
    "2021-08-23_pre_proc_Empty",
    detector,
    f"{data_dir_23}/pre_proc_Empty_30degsec",
    proj_start=10,
    proj_end=1000,
    is_full=True,
    is_rotational=False,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_23_empty_static)

for size in ["10mm", "14mm", "23mm"]:
    _scan = StaticScan(
        f"2021-08-23_3x{size}_foamballs_vertical_refempty",
        detector,
        f"{data_dir_23}/pre_proc_3x{size}_foamballs_vertical",
        proj_start=1025,
        proj_end=1801,
        # darks_dir=f'{data_dir}/pre_proc_Darkfield',
        geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
        references=[_23_full_rotating],
    )
    SCANS.append(_scan)

# # The 474mm table not-amended geometry looks better than
# #     the amended 3x10mm post-calibration, for the 19lmin scan
for lmin in [19, 20, 22, 25]:
    _scan = FluidizedBedScan(
        f"2021-08-23_{lmin}Lmin",
        detector,
        f"{data_dir_23}/pre_proc_{lmin}Lmin",
        liter_per_min=lmin,
        # darks_dir=f'{data_dir}/pre_proc_Darkfield',
        geometry=f"{calib_dir}/geom_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_26aug2021.npy",
        cameras=(1, 2, 3),
        col_inner_diameter=5.0,
        references=[_23_full_static],
        empty=_23_empty_static,
        # density_factor=0.10167654752731324,
    )
    _scan.references.append(_scan)  # other possible reference is own data
    SCANS.append(_scan)

###############################################################################
# Scans on 24 aug 2021
###############################################################################
_24_darks = StaticScan(
    "2021-08-24_pre_proc_Dark",
    detector,
    f"{data_dir_24}/pre_proc_Dark",
    proj_start=1,
    proj_end=1000,
    is_full=False,
    is_rotational=False,
)
SCANS.append(_24_darks)

_24_empty_rotating = StaticScan(
    "2021-08-24_pre_proc_Empty_30degsec_nonrotating",
    detector,
    f"{data_dir_24}/pre_proc_Empty_30degsec",
    proj_start=1026,
    proj_end=1800,
    is_full=False,
    is_rotational=True,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_24_empty_rotating)

_24_full_rotating = StaticScan(
    "2021-08-24_pre_proc_Full_30degsec_rotating",
    detector,
    f"{data_dir_24}/pre_proc_Full_30degsec",
    proj_start=1026,
    proj_end=1800,
    is_full=True,
    is_rotational=True,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_24_full_rotating)

_24_empty_nonrotating = StaticScan(
    "2021-08-24_pre_proc_Empty_30degsec_nonrotating",
    detector,
    f"{data_dir_24}/pre_proc_Empty_30degsec",
    proj_start=1,
    proj_end=1000,
    is_full=False,
    is_rotational=False,
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
)
SCANS.append(_24_empty_nonrotating)

for nr, pos in [(2, "horizontal"), (3, "vertical_wall")]:
    for size in [10, 14, 23]:
        _scan = StaticScan(
            f"2021-08-24_{nr}x{size}mm_foamballs_{pos}_refempty",
            detector,
            f"{data_dir_24}/pre_proc_{nr}x{size}mm_foamballs_{pos}",
            proj_start=1025,
            proj_end=1799,
            # darks_dir='/home/adriaan/data/evert/2021-08-23/pre_proc_Darkfield',
            geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
            cameras=(1, 2, 3),
            references=[_24_empty_rotating],
        )
        # SCANS.append(scan)

_scan = StaticScan(
    "2021-08-24_10mm_14mm_23mm_horizontal_refempty",
    detector,
    f"{data_dir_24}/pre_proc_10mm_14mm_23mm_foamballs_horizontal",
    proj_start=1025,
    proj_end=1799,
    # darks_dir=f'{data_dir}/pre_proc_Dark',
    geometry="resources/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    references=[_24_empty_rotating],
)
# SCANS.append(scan)

_scan = StaticScan(
    "2021-08-24_10mm_23mm_horizontal",
    detector,
    f"{data_dir_24}/pre_proc_10mm_23mm_foamballs_horizontal",
    proj_start=1025,
    proj_end=1799,
    # darks_dir=f'{data_dir}/pre_proc_Dark',
    geometry=f"{calib_dir}/geom_table474mm_26aug2021_amend_pre_proc_3x10mm_foamballs_vertical_wall_31aug2021.npy",
    geometry_scaling_factor=1.0 / 1.0106333,
    references=[_24_full_rotating],
    darks=_24_darks,
)
SCANS.append(_scan)

###############################################################################
# Scans on 25 aug 2021
###############################################################################


###############################################################################
# Other
###############################################################################


# # cam 1: 1027, 1798 (771)
# # cam 2: 1018, 1785 (767)
# # cam 3: 1020, 1791 (771), glitch at 1735
# scan = StaticPhantomScan(
#     1018,
#     1785, # inclusive, == 1027 visually
#     projs_dir=f'{DATA_DIR}/pre_proc_3x10mm_foamballs_vertical_01',
#     fulls_dir=f'{DATA_DIR}/pre_proc_Full_30degsec_03',
#     darks_dir=f'{DATA_DIR}/pre_proc_Dark_frames',
#     geometry='resources/vectors_geom_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_23aug2021.npy')
#
# scan.add_phantom(ball_10mm)
# scan.add_phantom(ball_10mm)
# scan.add_phantom(ball_10mm)
# SCANS.append(scan)
#
# # cam 1: 1025, 1799 (774)
# # cam 2: 1030, 1800 (770)
# # cam 3: 1027, 1800 (774)
# scan = StaticPhantomScan(
#     1018,
#     1785, # inclusive, == 1027 visually
#     projs_dir=f'{DATA_DIR}/pre_proc_3x10mm_foamballs_vertical_01',
#     fulls_dir=f'{DATA_DIR}/pre_proc_Full_30degsec_03',
#     darks_dir=f'{DATA_DIR}/pre_proc_Dark_frames',
#     geometry='resources/vectors_geom_pre_proc_Calibration_needle_phantom_30degsec_table474mm_calibrated_on_23aug2021.npy')
#
# scan.add_phantom(ball_10mm)
# scan.add_phantom(ball_10mm)
# scan.add_phantom(ball_10mm)
# SCANS.append(scan)
