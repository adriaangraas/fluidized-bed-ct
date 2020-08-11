import scipy.optimize

from examples.dynamic.settings_2020 import *
from fbrct.calibrate import *


def perturb_geoms(geoms):
    """Perturb geoms to simulate real data. """

    out = []
    for g in geoms:
        u, v = g.u, g.v
        # note: world coordinates for src, det,
        # but intrinic coordinates for rpy
        geom = StaticGeometry.fromDetectorVectors(
            source=g.source + 20 * np.array([-0.12, 0.35, 0.11]),
            detector=g.detector + [-0.45, 0.35, .25],
            u=u,
            v=v,
            roll=g.roll + 1 / 60 * np.pi,
            pitch=g.pitch + 2 / 60 * np.pi,
            yaw=g.yaw + 3 / 60 * np.pi
        )
        out.append(geom)

    return out


def perturb_points(points, sigma=0.5):
    """Perturb points to simulate real data."""

    out = []
    for i, p in enumerate(points):
        point = copy.deepcopy(p)
        err = sigma * (np.random.rand(3) * 2 - .1)
        point.value = point.value + err

        out.append(point)

    return out


"""
To make a sclable marker object, we would like to know how big the imaged
area is, given by the geometry. The area that can be seen from all 3 detectors
is a hexagon in the central plane. Given that you use cone beam sources ofc.,
and when the geometry is free of errors.

The height H and width W of the volumetric reconstruction box that is in the
center of the geom are given by the equations
   H / SOD = h / (SOD+ODD)
   W / SOD = w / (SOD+ODD)
with h, w are half the detector height and width. The box is given by [-W, W]
and [-H, H].

However, not the full box can be seen on all detectors, that is why we have
to shrink the box a bit. A safe choice seems the inscribed circle of 
the hexagon/polygon as the imaging circle. The inner radius of that circle on
the central plane in the volume is precisely W from the equation above.

We also have to remove some off the top of the volume, because the rays travel
under an angle and so not the full volume of height H may be seen by all
detectors. To be on the safe side, we compute the lowest entry point of an
xray in the volume. The steepest ray has an increment of h/(SOD+ODD) so from
point H that is the precise height in the middle of the volume, we travel the
radius to get to one of the lowest ray entry points:
    H_m = H - r * h/(SOD+ODD).
    
All in all my conclusion is that the maximally imagable area is a round column
with radius r and height 2 * H_m:
    r = W = w / (SOD+SOD) * SOD
    2 * H_m = 2 * (h / (SOD+ODD) * SOD - r * h/(SOD+ODD))
"""

# set-up geometries and marker points
geoms = triangle_geom(SOURCE_RADIUS, DETECTOR_RADIUS)
w = DETECTOR_WIDTH_SPEC / 2
h = DETECTOR_HEIGHT / 2
SDD = SOURCE_RADIUS + DETECTOR_RADIUS
column_radius = w / SDD * SOURCE_RADIUS
column_height = 2 * (h / SDD * SOURCE_RADIUS - column_radius * h / SDD)

# include a safety factor of 50% / 75%
column_radius *= 0.80
column_height *= 0.80
print(f"Suggested column size: r={column_radius}, h={column_height}, "
      f"d={2 * column_radius}")
points = triangle_column_points(rad=column_radius, height=column_height,
                                num_angles=3)
rotate_points(points, 0, 0, np.pi / 40)
plot_scatter3d([p.value for p in points])

# in cm, it looks like 1mm is ok, 2 mm isn't, 3 mm is horrible
error_magnitude_points = .200

# fake real data
geoms_true = perturb_geoms(geoms)
points_true = perturb_points(points, sigma=error_magnitude_points)
data_true = xray_multigeom_op(geoms_true, points_true)

# `geoms` and `points` will be optimized in-place, so we'd make a backup here
# for later reference.
geoms_initial = copy.deepcopy(geoms)
points_initial = copy.deepcopy(points)

# make sure to optimize over the points
for p in points:  # type: Point
    p.optimize = False
    p.bounds[0] = p.value - [1 * error_magnitude_points] * 3
    p.bounds[1] = p.value + [1 * error_magnitude_points] * 3

model = OptimizationProblem(
    points=points,
    geoms=geoms,
    data=data_true,
)

r = scipy.optimize.least_squares(
    fun=model,
    x0=params2ndarray(model.params()),
    bounds=model.bounds(),
    verbose=1,
    method='trf',
    tr_solver='exact',
    loss='huber',
    jac='3-point'
)
geoms_calibrated, points_calibrated = model.update(r.x)


for d1, d2 in zip(data_true, xray_multigeom_op(geoms_initial, points_initial)):
    plot_scatter2d(d1, d2, det=[w, h])
for d1, d2 in zip(data_true,
                  xray_multigeom_op(geoms_calibrated, points_calibrated)):
    plot_scatter2d(d1, d2, det=[w, h])

for i, (g1, g2) in enumerate(zip(geoms_true, geoms_calibrated)):
    print(f"--- GEOM {i} ---")
    print(f"source   : {g1.source} : {g2.source}")
    print(f"detector : {g1.detector} : {g2.detector}")
    print(f"roll     : {g1.roll} : {g2.roll}")
    print(f"pitch    : {g1.pitch} : {g2.pitch}")
    print(f"yaw      : {g1.yaw} : {g2.yaw}")

for i, (g1, g2) in enumerate(zip(geoms_true, geoms_calibrated)):
    try:
        decimal_accuracy = 3
        np.testing.assert_almost_equal(g1.source, g2.source, decimal_accuracy)
        np.testing.assert_almost_equal(g1.detector, g2.detector,
                                       decimal_accuracy)
        np.testing.assert_almost_equal(g1.roll, g2.roll, decimal_accuracy)
        np.testing.assert_almost_equal(g1.pitch, g2.pitch, decimal_accuracy)
        np.testing.assert_almost_equal(g1.yaw, g2.yaw, decimal_accuracy)
    except:
        pass

# put back a different object
points_cube = cube_points(w=column_radius, d=column_radius, h=column_height)
data_cube_true = xray_multigeom_op(geoms_true, points_cube)
data_cube_calibrated = xray_multigeom_op(geoms_calibrated, points_cube)
for d1, d2 in zip(data_cube_true, data_cube_calibrated):
    plot_scatter2d(d1, d2, det=[w, h])
