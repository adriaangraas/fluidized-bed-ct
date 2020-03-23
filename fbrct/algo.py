import astra

from odl.tomo import RayTransform
from odl.tomo.backends import astra_setup
from odl.tomo.backends.astra_cuda import ASTRA_CUDA_AVAILABLE


def astra_cuda3d_algo(op,
                      x,
                      rhs,
                      niter=1,
                      variant='SIRT3D_CUDA',
                      min_constraint=None,
                      vol_mask=None,

                      extra_algorithm_conf=None):
    """
    Signature is compatible with odl.solvers, but `op` is never explicitly
    called.
    """

    if not ASTRA_CUDA_AVAILABLE:
        raise Exception("`astra_cuda_3d_algo` cannot be used when ASTRA CUDA is "
                        "not available. Please install ASTRA CUDA first.")

    variants = {'SIRT3D_CUDA', 'CGLS3D_CUDA'}  # TODO
    if not variant in variants:
        raise ValueError('Incorrect `variant` supplied.')

    if not isinstance(op, RayTransform):
        raise ValueError('Operator is not a `RayTransform`.')

    if not op.impl == 'astra_cuda':
        raise ValueError('`op.impl` must be `astra_cuda`.')

    geometry = op.geometry
    if geometry.ndim != 3:
        raise ValueError("`geometry.ndim` must be 3d.")

    # Define ASTRA geometry
    # astra_vol_geom = astra.create_vol_geom(*domain_size)
    astra_vol_geom = astra_setup.astra_volume_geometry(op.domain)

    # det_row_count = geometry.det_partition.shape[1]
    # det_col_count = geometry.det_partition.shape[0]
    # vec = astra_setup.astra_conebeam_3d_geom_to_vec(geometry)
    # astra_proj_geom = astra.create_proj_geom('cone_vec', det_row_count, det_col_count, vec)
    astra_proj_geom = astra_setup.astra_projection_geometry(geometry)

    # Create ASTRA projector
    # proj_cfg = {}
    # proj_cfg['type'] = 'cuda3d'
    # proj_cfg['VolumeGeometry'] = astra_vol_geom
    # proj_cfg['ProjectionGeometry'] = astra_proj_geom
    # proj_cfg['options'] = {}
    # proj_id = astra.projector3d.create(proj_cfg)
    proj_id = astra_setup.astra_projector('cuda3d', astra_vol_geom, astra_proj_geom, ndim=3)

    # Create sinogram
    # sinogram_id, sinogram = astra.create_sino3d_gpu(phantom, astra_proj_geom, astra_vol_geom)
    sinogram_id = astra_setup.astra_data(astra_proj_geom,
                                         datatype='projection',
                                         data=rhs,
                                         allow_copy=False)

    # Create a data object for the reconstruction
    # rec_id = astra.data3d.create('-vol', astra_vol_geom)
    rec_id = astra_setup.astra_data(astra_vol_geom,
                                    datatype='volume',
                                    ndim=op.domain.ndim,
                                    data=x,
                                    allow_copy=False)

    # TODO(Adriaan): should this be supported by astra_setup.astra_algorithm
    # Set up the parameters for a reconstruction algorithm using the CUDA backend
    cfg = astra.astra_dict(variant)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    cfg['option'] = {}

    if min_constraint is not None:
        cfg['option']['MinConstraint'] = min_constraint

    if vol_mask is not None:
        if not vol_mask in op.domain:
            raise ValueError("`domain_mask` needs to be in `op.domain`.")

        mask_id = astra_setup.astra_data(astra_vol_geom,
                                         datatype='volume',
                                         ndim=op.domain.ndim,
                                         data=vol_mask,
                                         allow_copy=False)
        cfg['option']['ReconstructionMaskId'] = mask_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, niter)

    rec = astra.data3d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinogram_id)
    astra.projector3d.delete(proj_id)

    x.data[:] = rec
    return x
