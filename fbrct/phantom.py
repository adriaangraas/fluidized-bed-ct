import numpy as np
import odl

PHANTOM_3D_SIMPLE_ELLIPSOID = 1000
PHANTOM_3D_SIMPLE_ROTATED_ELLIPSOID = 1001
PHANTOM_3D_DOUBLE_BUBBLE = 1002
PHANTOM_3D_DOUBLE_BUBBLE_NONALIGNED = 1003


def generate_3d_phantom_data(model_number, L, H, n, m, geometry, from_volume_accuracy=256,
                             path = '/ufs/adriaan/tmp/pycharm_project_639/resources/phantoms_3D.dat'):
    import tomophantom
    from tomophantom import TomoP3D, TomoP2D

    N_size = from_volume_accuracy  # number of voxels in x, y axis
    M_size = int(np.ceil(N_size / n * m))  # number of voxels in z axis
    bigger_size = np.max(N_size, M_size)

    phantom_3Dt = TomoP3D.ModelTemporal(model_number, bigger_size, path)

    timesteps = phantom_3Dt.shape[0]

    # make an upscaled xray transform (we need accurate sinograms)
    reco_space_copy = odl.uniform_discr(
        min_pt=[-L, -L, -H],
        max_pt=[L, L, H],
        shape=[N_size, N_size, M_size])
    xray_transform = odl.tomo.RayTransform(reco_space_copy, geometry)

    p_lst = list()
    for t in range(timesteps):
        # make the phantom fit in the reconstruction space
        if N_size > M_size:
            x = xray_transform.domain.element(phantom_3Dt[t, ..., :M_size])
        else:
            left = M_size//2 - N_size//2
            right = M_size//2 + N_size//2
            x = xray_transform.domain.element(phantom_3Dt[t, left:right, left:right, :])

        p = xray_transform(x).data
        p_lst.append(p)
        # plot_3d(phantom_3Dt[t, ..., :M_size])
        # plot_sino(p)

    return np.array(p_lst)
