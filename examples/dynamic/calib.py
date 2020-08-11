import matplotlib.pyplot as plt
import pyqtgraph as pq
from odl.tomo.backends import astra_algorithm, astra_data, \
    astra_projection_geometry, \
    astra_projector, astra_volume_geometry
from odl.tomo.backends.astra_cuda import AstraCudaImpl

from examples.dynamic.loader import load, load_referenced_projs_from_fulls
from examples.dynamic.settings_2020 import *
from fbrct.util import *

# data_dir = "/mnt/sshfs/bigstore/adriaan/data/evert/2020-02-19 3D paper dataset 1/"
# data_dir = "/home/adriaan/data/evert/2020-02-19 3D paper dataset 1/"
data_dir = "/bigstore/adriaan/data/evert/2020-02-19 3D paper dataset 1/"

scans = filter(
    lambda s: s.projs_dir == 'pre_proc_10mm_phantom_center',
    SCANS.items()
)

scan = next(scans)
print(f"Next up is {scan}...")
projs_path = os.path.join(data_dir, scan.date, scan.projs_dir)
fulls_path = os.path.join(data_dir, scan.date, scan.fulls_dir)

print("Loading...")
reg = 'camera ([1-3])/median_(1)\.tiff$'
projs = load(projs_path, range(1, 2), regex=reg)
projs_copy = np.copy(projs)
p = load_referenced_projs_from_fulls(projs, fulls_path, 'median')

assert p.shape[2] == DETECTOR_ROWS
assert p.shape[3] == DETECTOR_COLS

detector_mid = DETECTOR_ROWS // 2
offset = DETECTOR_ROWS // 2 - 0
recon_height_range = range(detector_mid - offset, detector_mid + offset)
cutoff = 0
recon_width_range = range(cutoff, DETECTOR_COLS - cutoff)
recon_height_length = int(len(recon_height_range))
recon_width_length = int(len(recon_width_range))

L = 0.5 * DETECTOR_WIDTH / (
    SOURCE_RADIUS + DETECTOR_RADIUS) * SOURCE_RADIUS
n = 200  # reconstruction on a n*n*m grid

# scale amount of voxels and physical height according to the selected reconstruction range
horizontal_voxels_per_pixel = n / DETECTOR_COLS
m = int(np.floor(horizontal_voxels_per_pixel * recon_height_length))
m = max(m, 1)
H = L * m / n  # physical height

apart = uniform_angle_partition()
dpart = detector_partition_3d(recon_width_length,
                              recon_height_length,
                              DETECTOR_PIXEL_WIDTH,
                              DETECTOR_PIXEL_HEIGHT)

geometry = odl.tomo.ConeBeamGeometry(apart, dpart, SOURCE_RADIUS,
                                     DETECTOR_RADIUS)
# vec_geometry = odl.tomo.ConeVecGeometry.fromConebeam(geometry)

reco_space_3d = odl.uniform_discr(
    min_pt=[-L, -L, -H],
    max_pt=[L, L, H],
    shape=[n, n, m], dtype=np.float32)


class MyImpl(AstraCudaImpl):
    def create_ids(self):
        """Create ASTRA objects."""
        # Create input and output arrays
        if self.geometry.motion_partition.ndim == 1:
            motion_shape = self.geometry.motion_partition.shape
        else:
            # Need to flatten 2- or 3-dimensional angles into one axis
            motion_shape = (np.prod(self.geometry.motion_partition.shape),)

        proj_shape = motion_shape + self.geometry.det_partition.shape
        proj_ndim = len(proj_shape)

        if proj_ndim == 2:
            astra_proj_shape = proj_shape
            astra_vol_shape = self.vol_space.shape
        elif proj_ndim == 3:
            # The `u` and `v` axes of the projection data are swapped,
            # see explanation in `astra_*_3d_geom_to_vec`.
            astra_proj_shape = (proj_shape[1], proj_shape[0], proj_shape[2])
            astra_vol_shape = self.vol_space.shape

        self.vol_array = np.empty(astra_vol_shape, dtype='float32', order='C')
        self.proj_array = np.empty(astra_proj_shape, dtype='float32',
                                   order='C')

        # Create ASTRA data structures
        vol_geom = astra_volume_geometry(self.vol_space)
        proj_geom = astra_projection_geometry(self.geometry)

        # each row:
        # (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ)
        print(proj_geom['Vectors'])
        CORRECTION_FACTORS = np.zeros((3, 12))
        CORRECTION_FACTORS[0, :6] = [-2.2, -0, 0, 0.4, -4.4, 0.6]
        CORRECTION_FACTORS[1, :6] = [0, 0, 0, 0, 0, 0]
        CORRECTION_FACTORS[2, :6] = [-0.0, 0, 0, 0, 0, 0]

        proj_geom['Vectors'][0, 1:3] *= 1.0 # move source bw
        proj_geom['Vectors'][2, 1:3] *= .8 # move source bw
        proj_geom['Vectors'] += CORRECTION_FACTORS

        self.vol_id = astra_data(
            vol_geom,
            datatype='volume',
            ndim=self.vol_space.ndim,
            data=self.vol_array,
            allow_copy=False,
        )

        proj_type = 'cuda' if proj_ndim == 2 else 'cuda3d'
        self.proj_id = astra_projector(
            proj_type, vol_geom, proj_geom, proj_ndim
        )

        self.sino_id = astra_data(
            proj_geom,
            datatype='projection',
            ndim=proj_ndim,
            data=self.proj_array,
            allow_copy=False,
        )

        # Create algorithm
        self.algo_forward_id = astra_algorithm(
            'forward',
            proj_ndim,
            self.vol_id,
            self.sino_id,
            self.proj_id,
            impl='cuda',
        )

        # Create algorithm
        self.algo_backward_id = astra_algorithm(
            'backward',
            proj_ndim,
            self.vol_id,
            self.sino_id,
            self.proj_id,
            impl='cuda',
        )


sinogram = p[0, :, recon_height_range,
           recon_width_range.start:recon_width_range.stop]
sinogram = np.swapaxes(sinogram, 0, 2)
sinogram = np.ascontiguousarray(sinogram.astype(np.float32))

sinogram_highlight = np.copy(np.swapaxes(projs_copy[0, ...], 1, 2))
# sinogram_highlight[...] = 0.5

l = 2
sinogram_highlight[0, 76 - l:76 + l, 969 - l:969 + l] = 0.2
sinogram_highlight[1, 114 - l:114 + l, 962 - l:962 + l] = 0.4
sinogram_highlight[2, 193 - l:193 + l, 961 - l:961 + l] = 0.8
sinogram_highlight[0, 257 - l:257 + l, 952 - l:952 + l] = 0.2
sinogram_highlight[1, 236 - l:236 + l, 979 - l:979 + l] = 0.4
sinogram_highlight[2, 252 - l:252 + l, 970 - l:970 + l] = 0.8

sinogram_highlight[0, 105 - l:105 + l, 1008 - l:1008 + l] = 0.2
sinogram_highlight[1, 330 - l:330 + l, 1014 - l:1014 + l] = 0.4
sinogram_highlight[2, 129 - l:129 + l, 1005 - l:1005 + l] = 0.8

sinogram_highlight[0, 179 - l:179 + l, 699 - l:699 + l] = 0.2
sinogram_highlight[1, 158 - l:158 + l, 706 - l:706 + l] = 0.4
sinogram_highlight[2, 225 - l:225 + l, 703 - l:703 + l] = 0.8

sinogram_highlight[0, 168 - l:168 + l, 779 - l:779 + l] = 0.2
sinogram_highlight[1, 183 - l:183 + l, 788 - l:788 + l] = 0.4
sinogram_highlight[2, 211 - l:211 + l, 784 - l:784 + l] = 0.8

sinogram_highlight[0, 177 - l:177 + l, 593 - l:593 + l] = 0.2
sinogram_highlight[1, 210 - l:210 + l, 602 - l:602 + l] = 0.4
sinogram_highlight[2, 172 - l:172 + l, 600 - l:600 + l] = 0.8

sinogram_highlight[0, 280 - l:280 + l, 499 - l:499 + l] = 0.2
sinogram_highlight[1, 234 - l:234 + l, 521 - l:521 + l] = 0.4
sinogram_highlight[2, 43 - l:43 + l, 510 - l:510 + l] = 0.8

sinogram_highlight[0, 232 - l:232 + l, 318 - l:318 + l] = 0.2
sinogram_highlight[1, 48 - l:48 + l, 312 - l:312 + l] = 0.4
sinogram_highlight[2, 277 - l:277 + l, 300 - l:300 + l] = 0.8

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(sinogram_highlight[0].T)
ax2.imshow(sinogram_highlight[1].T)
ax3.imshow(sinogram_highlight[2].T)
plt.tight_layout()
plt.show()



def replot(x0, y0, z0, x, y, z, x2, y2, z2, x3, y3, z3, x4, y4, z4, x5, y5, z5):
    xray_transform = odl.tomo.RayTransform(
        reco_space_3d,
        geometry, impl=MyImpl
    )

    calib_vol = np.zeros(xray_transform.domain.shape)
    dia = 2
    x_end = x + dia
    y_end = y + dia
    z_end = z + dia
    calib_vol[x0:x0 + dia, y0:y0 + dia, z0:z0 + dia] = 2
    calib_vol[x:x_end, y:y_end, z:z_end] = 1
    calib_vol[x2:x2 + dia, y2:y2 + dia, z2:z2 + dia] = 1
    calib_vol[x3:x3 + dia, y3:y3 + dia, z3:z3 + dia] = 1
    calib_vol[x4:x4 + dia, y4:y4 + dia, z4:z4 + dia] = 1
    calib_vol[x5:x5 + dia, y5:y5 + dia, z5:z5 + dia] = 1

    plt.close()
    projs = xray_transform(calib_vol)
    volu = xray_transform.adjoint(sinogram_highlight)
    del xray_transform

    volu.data[x0:x0 + dia, y0:y0 + dia, z0:z0 + dia] = 1
    volu.data[x:x_end, y:y_end, z:z_end] = 1
    volu.data[x2:x2 + dia, y2:y2 + dia, z2:z2 + dia] = 1
    volu.data[x3:x3 + dia, y3:y3 + dia, z3:z3 + dia] = 1
    volu.data[x4:x4 + dia, y4:y4 + dia, z4:z4 + dia] = 1
    volu.data[x5:x5 + dia, y5:y5 + dia, z5:z5 + dia] = 1
    pq.image(volu.data)
    pq.image(np.swapaxes(volu.data, 2, 0))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow((50*1000* projs.data[0] + np.swapaxes(projs_copy[0, 0, :, :],0,1)).T)
    ax2.imshow((50*1000* projs.data[1] + np.swapaxes(projs_copy[0, 1, :, :],0,1)).T)
    ax3.imshow((50*1000* projs.data[2] + np.swapaxes(projs_copy[0, 2, :, :],0,1)).T)
    plt.tight_layout()
    plt.show()


replot(144, 75, 480, 99, 90, 391, 105, 79, 350, 104, 108, 299, 155, 155, 255, 135, 30, 155)

# start = np.zeros(xray_transform.domain.shape)

# x = xray_transform.domain.element(start)
# x = astra_cuda3d_algo(
#     xray_transform,
#     x,
#     sinogram,
#     1125,
#     min_constraint=0.,
#     vol_mask=mask
# )
# im = np.array(x.data).astype(np.float64)
#
# import matplotlib.pyplot as plt
# plt.figure()
# pq.image(im)
