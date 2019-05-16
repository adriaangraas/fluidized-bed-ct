import tomophantom
from tomophantom import TomoP3D, TomoP2D
import numpy as np
import odl

PHANTOM_3D_SIMPLE_ELLIPSOID = 1000
PHANTOM_3D_SIMPLE_ROTATED_ELLIPSOID = 1001


def generate_3d_phantom_data(model_number, L, H, n, m, geometry, from_volume_accuracy=256):
    N_size = from_volume_accuracy
    M_size = int(np.ceil(N_size / n * m))

    assert M_size <= N_size

    phantom_3Dt = TomoP3D.ModelTemporal(model_number, N_size, '../resources/phantoms_3D.dat')

    timesteps = phantom_3Dt.shape[0]

    # make an upscaled xray transform (we need accurate sinograms)
    reco_space_copy = odl.uniform_discr(
        min_pt=[-L, -L, -H],
        max_pt=[L, L, H],
        shape=[N_size, N_size, M_size])
    xray_transform = odl.tomo.RayTransform(reco_space_copy, geometry)

    p_lst = list()
    for t in range(timesteps):
        x = xray_transform.domain.element(phantom_3Dt[t, ..., :M_size])
        p = xray_transform(x).data
        p_lst.append(p)
        # plot_3d(phantom_3Dt[t, ..., :M_size])
        # plot_sino(p)

    return np.array(p_lst)
