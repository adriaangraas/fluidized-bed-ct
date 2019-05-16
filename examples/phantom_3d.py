from fbrct import *
from fbrct.phantom import *
from fbrct.util import *

recon_height_length = 700
n = 50  # reconstruction on a n*n*m grid
L = 10  # -L cm to L cm in the physical space
# scale amount of pixels and physical height according to the selected reconstruction range
m = max(int(np.floor(n / DETECTOR_ROWS * recon_height_length)), 1)
H = L * m / n  # physical height

apart = uniform_angle_partition()
dpart = detector_partition_3d(recon_height_length)
geometry = odl.tomo.ConeFlatGeometry(apart, dpart, SOURCE_RADIUS, DETECTOR_RADIUS)
reco_space_3d = odl.uniform_discr(
    min_pt=[-L, -L, -H],
    max_pt=[L, L, H],
    shape=[n, n, m])
xray_transform = odl.tomo.RayTransform(reco_space_3d, geometry)

p = generate_3d_phantom_data(PHANTOM_3D_SIMPLE_ROTATED_ELLIPSOID, L, H, n, m, geometry)
timerange = range(p.shape[0])

for t in timerange:
    sinogram = p[t, ...]
    plot_sino(sinogram, pause=0.1)

    x = xray_transform.domain.element(np.zeros(xray_transform.domain.shape))
    reconstruct_filter(
        xray_transform,
        sinogram,
        x,
        niter=150,
        mask=column_mask([n, n, m]),
        clip=[0, None])

    plot_3d(x.data, pause=.1)
