import matplotlib.pyplot as plt
from fbrct.phantom import generate_3d_phantom_data, PHANTOM_3D_SIMPLE_ROTATED_ELLIPSOID

from examples.dynamic.loader import Loader
from examples.dynamic.settings import *
from fbrct import *
from fbrct.util import *

resume = 0
path = '/export/scratch2/adriaan/evert/2019-07-25 dataset 3D reconstructie CWI/'
# dataset = 'pre_proc_6_60lmin_83mm_FOV2'
dataset = 'pre_proc_3_30lmin_83mm_FOV2'
# dataset = 'pre_proc_1_65lmin_83mm_FOV2'
loader = Loader(path)
recon_start_timeframe = 110
recon_end_timeframe = recon_start_timeframe + 30
p = loader.projs(dataset, range(recon_start_timeframe, recon_end_timeframe))
print(p.shape)
T = p.shape[0]
det_height = p.shape[2]

detector_mid = DETECTOR_ROWS // 2
# offset = 250
offset = DETECTOR_ROWS // 2 - 100
recon_height_range = range(detector_mid - offset, detector_mid + offset)
# recon_width_range = range(40, DETECTOR_COLS - 40)
# recon_height_range = range(774 - 350, 774 + 350)
# recon_height_range = range(DETECTOR_ROWS)
cutoff = 0
recon_width_range = range(cutoff, DETECTOR_COLS - cutoff)
recon_height_length = int(len(recon_height_range))
recon_width_length = int(len(recon_width_range))

n = 200  # reconstruction on a n*n*m grid
L = 1.7178442  # -L cm to L cm in the physical space
# I admit to calculating the above value. It is the solution from det_width/sdd = 2*L/SOD.

# scale amount of voxels and physical height according to the selected reconstruction range
horizontal_voxels_per_pixel = n / DETECTOR_COLS
m = int(np.floor(horizontal_voxels_per_pixel * recon_height_length))
m = max(m, 1)
H = L * m / n  # physical height

apart = uniform_angle_partition()
dpart = detector_partition_3d(recon_width_length, recon_height_length, DETECTOR_PIXEL_WIDTH, DETECTOR_PIXEL_HEIGHT)
geometry = odl.tomo.ConeFlatGeometry(apart, dpart, SOURCE_RADIUS, DETECTOR_RADIUS)

reco_space_3d = odl.uniform_discr(
    min_pt=[-L, -L, -H],
    max_pt=[L, L, H],
    shape=[n, n, m])
xray_transform = odl.tomo.RayTransform(reco_space_3d, geometry)

ph_sphere = generate_3d_phantom_data(PHANTOM_3D_SIMPLE_ROTATED_ELLIPSOID, L, H, n, m, geometry, from_volume_accuracy=50)
# ph_sl = odl.phantom.shepp_logan(xray_transform.domain, modified=True)
# ph_sl.data[:] = 1

for t in range(recon_start_timeframe, recon_end_timeframe):
    save_name = f"recon_3d_{dataset}_t{t}"
    print(f"Starting {save_name}...")

    # sinogram selection and scaling
    pr = p[t - recon_start_timeframe, :, recon_height_range, recon_width_range.start:recon_width_range.stop]
    sinogram = np.swapaxes(pr, 0, 1)
    sinogram = np.swapaxes(sinogram, 1, 2)
    # sinogram = xray_transform(ph_sl).data
    # sinogram = ph_sphere[10]

    # plot_sino(sinogram)

    # reconstruct
    start = np.zeros(xray_transform.domain.shape) if resume is 0 else np.load(f'recon_{resume}.npy')

    x = xray_transform.domain.element(start)

    reconstruct_filter(
        xray_transform,
        sinogram,
        x,
        niter=150,
        clip=(0, None),
        mask=column_mask([n, n, m]),
        iter_start=resume,
        iter_save=150,
        save_name=save_name,
        # fn_filter=lambda u: medians_3d(u)
    )

    # plot_3d(x.data,vmin=None,vmax=None)
    # # postprocess and plot
    # from skimage.restoration import denoise_tv_chambolle
    # y = denoise_tv_chambolle(x.data, weight=0.036)
    # plot_3d(y, vmax=0.1)
