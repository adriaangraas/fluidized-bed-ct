import odl
import matplotlib.pyplot as plt
import numpy as np
from fbrct import *
from examples.settings import *
from examples.util import load_dataset, uniform_angle_partition, detector_partition_2d

fname = '/export/scratch1/adriaan/MatlabProjects/DynamicTomography/astra_scripts' \
    '/fluidized_bed_1_python_2.mat'
p, pref, (T, nr_detectors, det_height, det_count) = load_dataset(fname)

apart, dpart = uniform_angle_partition(), detector_partition_2d()
geometry = odl.tomo.FanFlatGeometry(apart, dpart, SOURCE_RADIUS, DETECTOR_RADIUS)

recon_height_low = 773  # height of the slice to reconstruct, in pixels
recon_height_high = 774

recon_start_timeframe = 13  # starting frame
recon_end_timeframe = T

n = 100  # amount of voxels in one dimension (i.e. nxn object)
L = 10  # centimeters  -L cm to L cm in the physical space

# We're building the linear X-Ray operator that artificially projects
# the bubble reactor onto the detector
reco_space = odl.uniform_discr(
    min_pt=[-L, -L],
    max_pt=[L, L],
    shape=[n, n])
xray_transform = odl.tomo.RayTransform(reco_space, geometry)

for t in range(recon_start_timeframe, recon_end_timeframe):
    sinogram_low = -(p[t, :, recon_height_low, :] - pref[:, recon_height_low, :])
    sinogram_high = -(p[t, :, recon_height_high, :] - pref[:, recon_height_high, :])

    # reconstruct iteratively
    # starting vector is 0 in iteration
    x_low = xray_transform.domain.element(np.zeros(xray_transform.domain.shape))
    x_high = xray_transform.domain.element(np.zeros(xray_transform.domain.shape))

    reconstruct_filter(
        xray_transform,
        sinogram_low,
        x_low,
        niter=100,
        clip=(0, None),  # clipping values
        fn_filter=lambda u: .1 * medians_2d(u) + .9 * u,  # median filter
        mask=circle_mask_2d(x_low.shape))

    reconstruct_filter(
        xray_transform,
        sinogram_high,
        x_high,
        niter=100,
        clip=(0, None),  # clipping values
        fn_filter=lambda u: .1 * medians_2d(u) + .9 * u,  # median filter
        mask=circle_mask_2d(x_high.shape))

    # plot results
    plt.figure(2)
    plt.imshow(x_high)
    plt.pause(.1)

    plt.figure(3)
    plt.imshow(x_low)
    plt.pause(.1)

    plt.figure(4)
    plt.imshow(x_high - x_low)
    plt.pause(.1)