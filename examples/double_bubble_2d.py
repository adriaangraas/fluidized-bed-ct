from fbrct import *
from fbrct.phantom import *
from fbrct.util import *
import matplotlib.pyplot as plt


recon_height = int(DETECTOR_ROWS/2)

n = 100  # amount of voxels in one dimension (i.e. nxn object)
L = 10  # centimeters  -L cm to L cm in the physical space

# We're building the linear X-Ray operator that artificially projects
# the bubble reactor onto the detector
reco_space = odl.uniform_discr(
    min_pt=[-L, -L],
    max_pt=[L, L],
    shape=[n, n])

def setup_rotated(phi):
    apart, dpart = uniform_angle_partition(offset=phi), detector_partition_2d()
    geometry = odl.tomo.FanFlatGeometry(apart, dpart, SOURCE_RADIUS, DETECTOR_RADIUS)
    xray_transform = odl.tomo.RayTransform(reco_space, geometry)

    return xray_transform


def data_rotated(phi):
    # we need a 3D geometry to simulate projection (detector) data
    dpart_3d = detector_partition_3d(DETECTOR_ROWS)
    phantom_geometry = odl.tomo.ConeFlatGeometry(uniform_angle_partition(offset=phi), dpart_3d,
                                                 SOURCE_RADIUS, DETECTOR_RADIUS)
    return generate_3d_phantom_data(PHANTOM_3D_DOUBLE_BUBBLE, L, L, n, n, phantom_geometry)

phi = np.pi / 3
p = data_rotated(phi)

for t in range(p.shape[0]):
    h = recon_height
    # take and scale the projection data
    sinogram = p[t, :, :, h]
    plot_sino(p[t, ...], pause=0.1)

    # reconstruct iteratively
    # starting vector is 0 in iteration
    xray_transform = setup_rotated(phi)
    x = xray_transform.domain.element(
        np.zeros(xray_transform.domain.shape))

    reconstruct_filter(
        xray_transform,
        sinogram,
        x,
        niter=150,
        clip=(0, None),  # clipping values
        # fn_filter=lambda u: .1 * medians_2d(u) + .9 * u,  # median filter
        mask=circle_mask_2d(x.shape)
    )

    # plot results
    plt.figure(2 * h)
    plt.clf()
    plt.imshow(x, vmax=1.5)
    plt.colorbar()
    plt.pause(1)

