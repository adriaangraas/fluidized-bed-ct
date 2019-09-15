import matplotlib.pyplot as plt

from examples.dynamic.loader import Loader
from fbrct import *
from fbrct.util import *

path = '/export/scratch2/adriaan/evert/2019-07-25 dataset 3D reconstructie CWI/'
dataset = 'pre_proc_6_60lmin_83mm_FOV2'
loader = Loader(path)
p = loader.projs(dataset)
T = p.shape[0]
det_height = p.shape[2]

recon_height_range = range(774 - 150, 774 + 150)
recon_height_length = int(len(recon_height_range))
recon_width_length = 274
recon_start_timeframe = 22
recon_end_timeframe = T
n = 50  # reconstruction on a n*n*m grid
L = 10  # -L cm to L cm in the physical space
# scale amount of pixels and physical height according to the selected reconstruction range
m = max(n // DETECTOR_ROWS * recon_height_length, 1)
H = L * m / n  # physical height

apart = uniform_angle_partition()
dpart = detector_partition_3d(recon_height_length, recon_width_length)
geometry = odl.tomo.ConeFlatGeometry(apart, dpart, SOURCE_RADIUS, DETECTOR_RADIUS)
reco_space_3d = odl.uniform_discr(
    min_pt=[-L, -L, -H],
    max_pt=[L, L, H],
    shape=[n, n, m])
xray_transform = odl.tomo.RayTransform(reco_space_3d, geometry)

for t in range(recon_start_timeframe, recon_end_timeframe):
    # sinogram selection and scaling
    pr = -p[t, :, recon_height_range, :]
    sinogram = np.swapaxes(pr, 0, 1)
    sinogram = np.swapaxes(sinogram, 1, 2)
    plot_sino(sinogram)

    # reconstruct
    x = xray_transform.domain.element(np.zeros(xray_transform.domain.shape))
    reconstruct_filter(
        xray_transform,
        sinogram,
        x,
        niter=1,
        clip=[0, None],
        mask=column_mask([n, n, m]))

    plot_3d(x)
    # # postprocess and plot
    # from skimage.restoration import denoise_tv_chambolle
    # y = denoise_tv_chambolle(x.data, weight=0.036)
    # plot_3d(y, vmax=0.1)
