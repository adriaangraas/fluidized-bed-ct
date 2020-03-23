import os
from pathlib import Path

from examples.dynamic.loader import load_referenced_projs_from_fulls
from fbrct import column_mask
from fbrct.algo import astra_cuda3d_algo
from fbrct.util import *

resume = 0
# from examples.dynamic.settings import *
# path = '/export/scratch2/adriaan/evert/2019-07-25 dataset 3D reconstructie CWI/'
# dataset = 'pre_proc_6_60lmin_83mm_FOV2'
# dataset = 'pre_proc_3_30lmin_83mm_FOV2'
# dataset = 'pre_proc_1_65lmin_83mm_FOV2'
# dataset = 'pre_proc_22lmin'
from examples.dynamic.settings_2020 import *

data_dir = "/bigstore/adriaan/data/evert/2020-02-19 3D paper dataset 1/"
reco_dir = "/bigstore/adriaan/recons/evert/2020-02-19 3D paper dataset 1/"

for date, scan_lst in SCANS.items():
    for scan in scan_lst:
        projs_path = os.path.join(data_dir, date, scan['projs_dir'])
        fulls_path = os.path.join(data_dir, date, scan['fulls_dir'])

        # recon_start_timeframe = 250
        # recon_end_timeframe = recon_start_timeframe + 30
        # p = load_referenced_projs_from_fulls(projs_path, fulls_path,
        #                                      range(recon_start_timeframe, recon_end_timeframe))
        p = load_referenced_projs_from_fulls(projs_path, fulls_path)
        recon_start_timeframe = 0
        recon_end_timeframe = p.shape[0]

        assert p.shape[2] == DETECTOR_ROWS
        assert p.shape[3] == DETECTOR_COLS

        detector_mid = DETECTOR_ROWS // 2
        offset = DETECTOR_ROWS // 2 - 0
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
        dpart = detector_partition_3d(recon_width_length, recon_height_length, DETECTOR_PIXEL_WIDTH,
                                      DETECTOR_PIXEL_HEIGHT)
        geometry = odl.tomo.ConeBeamGeometry(apart, dpart, SOURCE_RADIUS, DETECTOR_RADIUS)

        reco_space_3d = odl.uniform_discr(
            min_pt=[-L, -L, -H],
            max_pt=[L, L, H],
            shape=[n, n, m], dtype=np.float32)
        xray_transform = odl.tomo.RayTransform(reco_space_3d, geometry)

        # ph_sphere = generate_3d_phantom_data(PHANTOM_3D_SIMPLE_ROTATED_ELLIPSOID, L, H, n, m, geometry, from_volume_accuracy=50)
        # ph_sl = odl.phantom.shepp_logan(xray_transform.domain, modified=True)
        # ph_sl.data[:] = 1

        mask = xray_transform.domain.element(column_mask([n, n, m]))  # TODO: slow
        # mask = None

        for t in range(recon_start_timeframe, recon_end_timeframe):
            save_name = f"recon_t{t}.npy"
            filename = os.path.join(reco_dir, date, scan['projs_dir'], save_name)

            if os.path.exists(filename):
                print(f"File {filename} exists, continuing.")
                continue

            lockfile = filename + ".lock"
            if os.path.exists(lockfile):
                print(f"Lockfile {lockfile} exists, continuing.")
                continue
            else:
                os.makedirs(os.path.dirname(lockfile), exist_ok=True)
                Path(lockfile).touch()

            print(f"Starting {save_name}...")

            # sinogram selection and scaling
            sinogram = p[t - recon_start_timeframe, :, recon_height_range,
                       recon_width_range.start:recon_width_range.stop]
            # plot_sino(np.transpose(sinogram, [1, 2, 0]))
            sinogram = np.swapaxes(sinogram, 0, 2)
            sinogram = np.ascontiguousarray(sinogram.astype(np.float32))

            # sinogram = xray_transform(ph_sl).data
            # sinogram = ph_sphere[10]

            # reconstruct
            start = np.zeros(xray_transform.domain.shape) if resume is 0 else np.load(
                f'recon_{resume}.npy')

            x = xray_transform.domain.element(start)
            x = astra_cuda3d_algo(xray_transform,
                                  x,
                                  sinogram,
                                  350,
                                  min_constraint=0.,
                                  vol_mask=mask)

            # reconstruct_filter(
            #     xray_transform,
            #     sinogram,
            #     x,
            #     niter=250,
            #     clip=(0, None),
            #     mask=column_mask([n, n, m]),
            #     iter_start=resume,
            #     iter_save=250,
            #     save_name=save_name,  # intermediate solve save
            #     # fn_filter=lambda u: medians_3d(u)
            # )

            print(f"Saving {filename}...")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.save(filename, x.data)
            try:
                Path(lockfile).unlink()
            except:
                pass

            # plot_3d(x.data, vmin=None, vmax=None)
            # # postprocess and plot
            # from skimage.restoration import denoise_tv_chambolle
            # y = denoise_tv_chambolle(x.data, weight=0.036)
