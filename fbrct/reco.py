from abc import abstractmethod
from typing import Any

import numpy as np
import warnings

from fbrct.loader import _apply_darkfields, reference_via_mode, \
    compute_bed_density, load, preprocess


def _astra_fdk_algo(volume_geom, projection_geom, volume_id, sinogram_id):
    import astra
    import astra.experimental

    proj_cfg = {
        "type": "cuda3d",
        "VolumeGeometry": volume_geom,
        "ProjectionGeometry": projection_geom,
        "options": {},
    }

    projector_id = astra.projector3d.create(proj_cfg)
    astra.experimental.accumulate_FDK(projector_id, volume_id, sinogram_id)


def _astra_sirt_algo(
    volume_id, sinogram_id, iters, mask_id,
    min_constraint=0.0, max_constraint=None,
):
    import astra

    cfg = astra.astra_dict(
        "SIRT3D_CUDA")  # 'FDK_CUDA', 'SIRT3D_CUDA', 'CGLS3D_CUDA',
    cfg["ReconstructionDataId"] = volume_id
    cfg["ProjectionDataId"] = sinogram_id
    cfg["option"] = {"MinConstraint": min_constraint,
                     "MaxConstraint": max_constraint,
                     "ReconstructionMaskId": mask_id}

    algo_id = astra.algorithm.create(cfg)
    astra.algorithm.run(algo_id, iters)  # nr iters


class Reconstruction:
    def __init__(
        self,
        path: str,
        detector,
    ):
        """
        From the data there is two types of reconstructions that can be
        performed.
        1. A reco using the 3 detectors as angles, if there was
           no rotation table used. In this case, you want SIRT/*ART, and if
           the scan was of a static object, you possibly want to take the
           average/median over all timeframes.
        2. A reco on a rotation table. In this case (assuming
           the scan is of a static object), the timeframes correspond to
           different angles of the object. Now there is a possibilty of
           overabundance of data, since after a full rotation, all three
           detectors record the same object, and hence the using needs to
           specify which information (which detector/frames) to use as angles
           in the reco.
        """

        self._path = path
        self.detector = detector

    @staticmethod
    def _reduce_ref(ref, reduction: str, detector_rows):
        """Reduce a (temporal) stack of projections into a single projection,
        with the effect of noise reduction/bubble removal."""

        if reduction == 'mean':
            reduced = np.mean(ref, axis=0)
        elif reduction == 'median':
            reduced = np.median(ref, axis=0)
        elif reduction == 'min':
            reduced = np.min(ref, axis=0)
        elif reduction == 'mode':
            if detector_rows is None:
                detector_rows = slice(None, None)

            reduced = np.zeros(ref.shape[1:], ref.dtype)
            reduced[..., detector_rows, :] = reference_via_mode(
                ref[..., detector_rows, :])
        else:
            raise ValueError

        return reduced

    def load_sinogram(
        self,
        t_range=None,
        cameras=None,
        darks_path=None,
        ref_path=None,
        ref_lower_density=False,
        ref_rotational=False,
        ref_reduction=None,
        ref_projs=None,
        darks_ran: range = None,
        empty_path=None,
        empty_rotational=False,
        empty_reduction='mean',
        empty_projs: range = None,
        detector_rows: range = None,
        density_factor: float = None,
        col_inner_diameter=None
    ):
        """Loads and preprocesses the sinogram."""

        if np.isscalar(t_range):
            t_range = range(t_range, t_range + 1)

        load_kwargs = {}
        if cameras is not None:
            load_kwargs["cameras"] = cameras
        if detector_rows is not None:
            load_kwargs["detector_rows"] = detector_rows

        dark = None
        if darks_path is not None:
            dark = load(darks_path, darks_ran, **load_kwargs)

        ref = None
        if ref_path is not None:
            if not ref_projs:
                ref_projs = None  # use all projs for referencing (static)
                if ref_rotational:
                    ref_projs = t_range  # reference one-on-one
            ref = load(ref_path, ref_projs, **load_kwargs)
            if dark is not None:
                _apply_darkfields(dark, ref)

            if not ref_rotational:
                assert ref_reduction is not None
                ref = self._reduce_ref(ref, ref_reduction, detector_rows)
            else:
                assert len(ref) == len(t_range)

            if density_factor is None:
                density_factor = 1.0
                if empty_path is not None:
                    assert col_inner_diameter is not None, (
                        "Column diameter needs to be known to compute empty"
                        " density factor.")
                    empty = load(empty_path, empty_projs, **load_kwargs)
                    if dark is not None:
                        _apply_darkfields(dark, empty)

                    if not empty_rotational:
                        assert empty_reduction is not None
                        empty = self._reduce_ref(empty, empty_reduction,
                                                 detector_rows)
                    else:
                        assert len(empty) == len(ref)
                    density_factor = compute_bed_density(
                        empty, ref, L=col_inner_diameter)
                else:
                    warnings.warn("No empty projs, or density factor provided."
                                  " Densities will be computed, rather than "
                                  " the gas fraction!")
        else:
            density_factor = 1.

        meas = load(self._path, t_range, **load_kwargs)
        if dark is not None:
            _apply_darkfields(dark, meas)
        meas = preprocess(meas, ref, ref_lower_density=ref_lower_density,
                          scaling_factor=1 / density_factor)
        return np.ascontiguousarray(meas.astype(np.float32))

    @staticmethod
    def clear():
        pass

    @abstractmethod
    def sino_gpu_and_proj_geom(self, sinogram: Any, vectors: np.ndarray):
        pass

    @abstractmethod
    def backward(
        self,
        proj_id,
        proj_geom,
        algo,
        voxels,
        voxel_size,
        iters,
        min_constraint,
        max_constraint,
        col_mask,
        callback,
    ):
        pass

    @abstractmethod
    def volume(self, vol_id):
        pass

    @abstractmethod
    def vol_params(self, voxels_x):
        pass


class AstraReconstruction(Reconstruction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def empty_volume_gpu(self, voxels: tuple, voxel_size):
        return self.volume_gpu(voxels, voxel_size)

    def volume_gpu(self, voxels, voxel_size, data=None):
        import astra

        voxels_x, voxels_z = voxels[0], voxels[2],
        w, h = voxels[0] * voxel_size / 2, voxels_z * voxel_size / 2
        print(
            f"Creating empty GPU volume {voxels_x}:{voxels_x}:{voxels_z} "
            f"with size {w*2}:{w*2}:{h*2}."
        )
        vol_geom = astra.create_vol_geom(
            voxels_x, voxels_x, voxels_z, -w, w, -w, w, -h, h
        )
        if data is None:
            vol_id = astra.data3d.create("-vol", vol_geom)
        else:
            vol_id = astra.data3d.create("-vol", vol_geom, data)

        return vol_id, vol_geom

    def sino_gpu_and_proj_geom(self, sinogram: Any, vectors: np.ndarray):
        import astra

        det = self.detector
        rows = det["rows"]
        cols = det["cols"]

        proj_geom = astra.create_proj_geom(
            "cone_vec",
            rows,
            cols,
            np.array(vectors),
        )
        sinogram = np.swapaxes(sinogram, 0, 1)
        proj_id = astra.data3d.create("-sino", proj_geom, sinogram)
        return proj_id, proj_geom

    def backward(
        self,
        proj_id,
        proj_geom,
        algo="sirt",
        voxels=None,
        voxel_size=None,
        iters=200,
        min_constraint=0.0,
        max_constraint=None,
        **kwargs,
    ):
        vol_id, vol_geom = self.empty_volume_gpu(voxels, voxel_size)

        print("Algorithm starts...")
        algo = algo.lower()
        if algo == "sirt":
            from fbrct import column_mask
            col_mask = column_mask(voxels)
            col_mask = np.transpose(col_mask, [2, 1, 0])
            mask_id, _ = self.volume_gpu(voxels, voxel_size, col_mask)
            _astra_sirt_algo(
                vol_id,
                proj_id,
                iters,
                mask_id,
                min_constraint=min_constraint,
                max_constraint=max_constraint,
            )
        elif algo == "fdk":
            _astra_fdk_algo(vol_geom, proj_geom, vol_id, proj_id)
        else:
            raise ValueError("Algorithm value incorrect.")

        return vol_id, vol_geom

    @staticmethod
    def forward(volume_id, volume_geom, projection_geom, returnData=False):
        return astra.creators.create_sino3d_gpu(
            volume_id, projection_geom, volume_geom, returnData=returnData
        )

    @staticmethod
    def volume(volume_id):
        import astra

        vol = astra.data3d.get(volume_id)
        return np.transpose(vol, [2, 1, 0])

    @staticmethod
    def sinogram(sinogram_id):
        import astra

        return astra.data3d.get(sinogram_id)

    @staticmethod
    def clear():
        import astra

        astra.clear()


class AstrapyReconstruction(Reconstruction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sino_gpu_and_proj_geom(self, sinogram: Any, vectors: np.ndarray):
        import astrapy

        geoms = []
        for vec in vectors:
            u = np.array(vec[6:9])
            v = np.array(vec[9:12])
            geom = astrapy.Geometry(
                tube_pos=vec[0:3],
                det_pos=vec[3:6],
                u_unit=u / np.linalg.norm(u),
                v_unit=v / np.linalg.norm(v),
                detector=astrapy.Detector(
                    rows=self.detector["rows"],
                    cols=self.detector["cols"],
                    pixel_width=np.linalg.norm(u),
                    pixel_height=np.linalg.norm(v),
                ),
            )
            geoms.append(geom)

        return sinogram, geoms

    def backward(
        self,
        proj,
        geom,
        algo="sirt",
        voxels: tuple=None,
        voxel_size: float=None,
        iters=200,
        min_constraint=None,
        max_constraint=None,
        vol_mask=None,
        col_mask=False,
        col_mask_percentage=None,
        callback=None,
        vol_rotation=None,
    ):
        import astrapy

        if col_mask:
            # col_mask = astrapy.bp(proj_mask, geom, vol_shp, vol_min, vol_max)
            # 7.5 is a lowerbound intensity in the volume after backprojecting
            # plt.figure()
            # plt.imshow(vol_mask[..., 50])
            # plt.show()
            # col_mask = (col_mask > 0.5).astype(np.float)

            from fbrct import column_mask

            # radius = int(voxels_x // 2 * col_mask_percentage)
            col_mask *= column_mask(voxels)

            if vol_mask is None:
                vol_mask = col_mask
            else:
                vol_mask *= col_mask

        algo = algo.lower()
        if algo == "sirt":
            vol_gpu = astrapy.sirt_experimental(
                projections=proj,
                geometry=geom,
                mask=vol_mask,
                volume_shape=voxels,
                volume_voxel_size=[voxel_size] * 3,
                iters=iters,
                min_constraint=min_constraint,
                max_constraint=max_constraint,
                chunk_size=200,
                algo="gpu",
                callback=callback,
            )
        elif algo == "fdk":
            vol_gpu = astrapy.fdk(
                projections=proj,
                geometry=geom,
                volume_shape=voxels,
                volume_voxel_size=[voxel_size] * 3,
                return_gpu=False,
            )
        else:
            raise ValueError(f"Algorithm {algo} not implemented.")

        return vol_gpu, False

    @staticmethod
    def volume(volume_gpu):
        return volume_gpu

    @staticmethod
    def sinogram(sinogram):
        return sinogram

