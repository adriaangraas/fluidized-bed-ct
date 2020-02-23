import matplotlib.pyplot as plt

from examples.dynamic.loader import Loader
from examples.dynamic.settings import DETECTOR_ROWS, SOURCE_RADIUS, DETECTOR_RADIUS, DETECTOR_COLS, DETECTOR_PIXEL_WIDTH
from fbrct import *
from fbrct.util import *

path = '/export/scratch2/adriaan/evert/2019-07-25 dataset 3D reconstructie CWI/'
dataset = 'pre_proc_6_60lmin_83mm_FOV2'
loader = Loader(path)
recon_start_timeframe = 15
recon_end_timeframe = recon_start_timeframe + 50
p = loader.projs(dataset, range(recon_start_timeframe, recon_end_timeframe))
T = p.shape[0]
det_height = p.shape[2]


# plt.figure()
# plt.imshow(p[200,0,...])
# plt.colorbar()
# plt.show()

recon_height = DETECTOR_ROWS // 2 + 100 # height of the slice to reconstruct, in pixels
cutoff = 0  # how many pixels to remove symmetrically from both sides of the detector
recon_width_range = range(cutoff, DETECTOR_COLS - cutoff)
recon_width_length = int(len(recon_width_range))
n = 200  # amount of voxels in one dimension (i.e. nxn object)
L = 1.7178442  # -L cm to L cm in the physical space

apart, dpart = uniform_angle_partition(), detector_partition_2d(recon_width_length, DETECTOR_PIXEL_WIDTH)
geometry = odl.tomo.FanFlatGeometry(apart, dpart, SOURCE_RADIUS, DETECTOR_RADIUS)

reco_space = odl.uniform_discr(
    min_pt=[-L, -L],
    max_pt=[L, L],
    shape=[n, n])
xray_transform = odl.tomo.RayTransform(reco_space, geometry)

# phantom test:
x = odl.phantom.shepp_logan(xray_transform.domain)
# phantom test confirmed that ODL gives log-normalized detector data, so I suppose we do have to ln() our data in advance

x_list = []

for t in range(recon_start_timeframe, recon_end_timeframe):
    x = xray_transform.domain.element(np.zeros(xray_transform.domain.shape))
    # take and scale the projection data
    sinogram = (p[t - recon_start_timeframe, :, recon_height, recon_width_range.start:recon_width_range.stop])
    # sinogram = xray_transform(x)

    # plt.figure()
    # plt.imshow(p[:, 2, recon_height, recon_width_range.start:recon_width_range.stop])
    # plt.show()


    # starting vector is 0 in iteration
    # x = xray_transform.domain.element(
    #     np.zeros(xray_transform.domain.shape))

    reconstruct_filter(
        xray_transform,
        sinogram,
        x,
        niter=350,
        clip=(0, None),  # clipping values
        # fn_filter=lambda u: .1 * medians_2d(u) + .9 * u,  # median filter
        mask=circle_mask_2d(x.shape),
    )


    x_list.append(x)

plt.figure(10)
while(True):
    for x in x_list:
        plt.imshow(x, vmin=0, vmax=4)
        plt.pause(.05)
