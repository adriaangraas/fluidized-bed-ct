from abc import abstractmethod
from typing import Any

import numpy as np

from fbrct.loader import load, preprocess


def _astra_fdk_algo(volume_geom, projection_geom, volume_id, sinogram_id):
    import astra

    proj_cfg = {
        "type": "cuda3d",
        "VolumeGeometry": volume_geom,
        "ProjectionGeometry": projection_geom,
        "options": {},
    }

    projector_id = astra.projector3d.create(proj_cfg)
    astra.experimental.accumulate_FDK(projector_id, volume_id, sinogram_id)


def _astra_sirt_algo(
    volume_id, sinogram_id, iters, min_constraint=0.0, max_constraint=None
):
    import astra

    cfg = astra.astra_dict("SIRT3D_CUDA")  # 'FDK_CUDA', 'SIRT3D_CUDA', 'CGLS3D_CUDA',
    cfg["ReconstructionDataId"] = volume_id
    cfg["ProjectionDataId"] = sinogram_id
    cfg["option"] = {"MinConstraint": min_constraint, "MaxConstraint": max_constraint}

    algo_id = astra.algorithm.create(cfg)
    astra.algorithm.run(algo_id, iters)  # nr iters


def _astra_nesterov_algo(
    volume_geom,
    projection_geom,
    volume_id,
    sinogram_id,
    iters: int = 50,
    min_constraint: float = 0.0,
):
    from fbrct.nesterov import AcceleratedGradientPlugin
    import astra

    astra.plugin.register(AcceleratedGradientPlugin)
    projector_id = astra.create_projector("cuda3d", projection_geom, volume_geom)
    cfg = astra.astra_dict("AGD-PLUGIN")
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ReconstructionDataId"] = volume_id
    cfg["ProjectorId"] = projector_id
    cfg["option"] = {}
    cfg["option"]["MinConstraint"] = min_constraint
    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id, iters)


class Reconstruction:
    def __init__(
        self,
        path: str,
        detector,
        expected_voxel_size_x: float,
        expected_voxel_size_z: float,
        det_subsampling=1,
        verbose=True,
    ):
        """
        From the data there is two types of reconstructions that can be
        performed.
        1. A reco using the 3 detectors as angles, if there was
           no rotation table used. In this case, you want SIRT/*ART, and if
           the scan was of a static process, you possibly want to take the
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
        self.verbose = verbose
        self.detector = detector
        self.det_subsampling = det_subsampling
        self._expected_voxel_size_x = expected_voxel_size_x
        self._expected_voxel_size_z = expected_voxel_size_z

    def load_sinogram(
        self,
        t_range=None,
        cameras=None,
        darks_path=None,
        ref_path=None,
        ref_lower_density=False,
        ref_mode="static",
        ref_projs=None,
        darks_ran: range = None,
        ref_normalization_path=None,
        ref_normalization_projs: range = None,
        detector_rows: range = None,
    ):
        """Loads and preprocesses the sinogram for .

        :param t_range:
            Range of the projections to use for reconstructoin
        :param cameras:
            If `None` takes all detectors, otherwise Sequence of `int`.
        :return:
            np.ndarray
        """
        if np.isscalar(t_range):
            t_range = range(t_range, t_range + 1)

        load_kwargs = {}
        if cameras is not None:
            load_kwargs["cameras"] = cameras
        if detector_rows is not None:
            load_kwargs["detector_rows"] = detector_rows

        if ref_path is not None:
            p = load(self._path, t_range, **load_kwargs)
            if not ref_projs:
                ref_projs = None  # use all projs for referencing (static)
                if ref_mode == "reco":
                    ref_projs = t_range  # reference one-on-one

            preproc_kwargs = {
                "ref": load(ref_path, ref_projs, **load_kwargs),
                "ref_mode": ref_mode,
            }
            if darks_path is not None:
                preproc_kwargs["dark"] = load(darks_path, darks_ran, **load_kwargs)
            if ref_normalization_path is not None:
                preproc_kwargs["ref_normalization"] = load(
                    ref_normalization_path, ref_normalization_projs, **load_kwargs
                )

            p = preprocess(p, ref_lower_density=ref_lower_density, **preproc_kwargs)
        else:
            p = load(self._path, t_range, **load_kwargs)
            preproc_kwargs = {}
            if darks_path is not None:
                preproc_kwargs["dark"] = load(darks_path, darks_ran, **load_kwargs)
            p = preprocess(p, **preproc_kwargs)

        return np.ascontiguousarray(p.astype(np.float32))

    @staticmethod
    def compute_volume_dimensions(
        voxel_width, voxel_height, det, nr_voxels_x=None, det_subsampling: int = 1
    ):
        # make voxel size larger if a small number of voxels is needed, and
        # vice versa
        if nr_voxels_x is None:
            nr_voxels_x = det["cols"]
            nr_voxels_z = det["rows"]
            scaling = 1
        else:
            # the amount of z-voxels should be proportional the detector dimensions
            estimated_nr_voxels_z = nr_voxels_x / det["cols"] * det["rows"]
            nr_voxels_z = int(np.ceil(estimated_nr_voxels_z))
            # since voxels_x is given, we need to scale the suggested voxel size
            scaling = nr_voxels_x / det["cols"]

        voxel_width /= scaling
        voxel_height /= scaling

        # compensate for detector subsampling
        if det_subsampling != 1:
            voxel_width /= det_subsampling
            voxel_height /= det_subsampling

        # compute total volume size based on the rounded voxel size
        width_from_center = voxel_width * nr_voxels_x / 2
        height_from_center = voxel_height * nr_voxels_z / 2

        return nr_voxels_x, nr_voxels_z, width_from_center, height_from_center

    def _compute_volume_dimensions(self, nr_voxels_x):
        return self.compute_volume_dimensions(
            self._expected_voxel_size_x,
            self._expected_voxel_size_z,
            self.detector,
            nr_voxels_x,
            self.det_subsampling,
        )

    @staticmethod
    def clear():
        pass

    @abstractmethod
    def sino_gpu_and_proj_geom(self, sino, geoms):
        pass

    @abstractmethod
    def backward(
        self,
        proj_id,
        proj_geom,
        algo,
        voxels_x,
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

    def empty_volume_gpu(self, voxels_x=None):
        import astra

        voxels_x, voxels_z, w, h = self._compute_volume_dimensions(voxels_x)
        print(
            f"Creating empty GPU volume {voxels_x}:{voxels_x}:{voxels_z} "
            f"with size {w}:{w}:{h}."
        )
        vol_geom = astra.create_vol_geom(
            voxels_x, voxels_x, voxels_z, -w, w, -w, w, -h, h
        )
        vol_id = astra.data3d.create("-vol", vol_geom)
        return vol_id, vol_geom

    def sino_gpu_and_proj_geom(self, sinogram: Any, vectors: np.ndarray):
        import astra

        det = self.detector
        rows = det["rows"]
        cols = det["cols"]

        assert rows % self.det_subsampling == 0
        assert cols % self.det_subsampling == 0

        proj_geom = astra.create_proj_geom(
            "cone_vec",
            rows // self.det_subsampling,
            cols // self.det_subsampling,
            np.array(vectors),
        )
        proj_id = astra.data3d.create("-sino", proj_geom, sinogram)
        return proj_id, proj_geom

    def backward(
        self,
        proj_id,
        proj_geom,
        algo="sirt",
        voxels_x=None,
        iters=200,
        min_constraint=0.0,
        max_constraint=None,
        **kwargs,
    ):

        vol_id, vol_geom = self.empty_volume_gpu(voxels_x)

        print("Algorithm starts...")
        algo = algo.lower()
        if algo == "sirt":
            _astra_sirt_algo(
                vol_id,
                proj_id,
                iters,
                min_constraint=min_constraint,
                max_constraint=max_constraint,
            )
        elif algo == "fdk":
            _astra_fdk_algo(vol_geom, proj_geom, vol_id, proj_id)
        elif algo == "nesterov":
            _astra_nesterov_algo(
                vol_geom,
                proj_geom,
                vol_id,
                proj_id,
                iters=iters,
                min_constraint=min_constraint,
            )
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

        return astra.data3d.get(volume_id)

    @staticmethod
    def sinogram(sinogram_id):
        import astra

        return astra.data3d.get(sinogram_id)

    @staticmethod
    def clear():
        import astra

        # del self._sinogram  # free RAM before loading volume
        astra.clear()


class RayveReconstruction(Reconstruction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sino_gpu_and_proj_geom(self, sinogram: Any, vectors: np.ndarray):
        import astrapy

        # det = self.detector
        # rayve_det = astrapy.Detector(det['rows'], det['cols'], pw, ph)

        # assert rows % self.det_subsampling == 0
        # assert cols % self.det_subsampling == 0

        # proj_geom = astra.create_proj_geom(
        #     'cone_vec',
        #     rows // self.det_subsampling,
        #     cols // self.det_subsampling,
        #     np.array(vectors))
        # proj_id = astra.data3d.create('-sino', proj_geom, sinogram)

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

    def _compute_volume_dimensions(self, nr_voxels_x):
        # can't have anisotropic voxels yet
        iso_voxel_size = min(self._expected_voxel_size_x, self._expected_voxel_size_z)
        return self.compute_volume_dimensions(
            iso_voxel_size,
            iso_voxel_size,
            self.detector,
            nr_voxels_x,
            self.det_subsampling,
        )

    def vol_params(self, voxels_x):
        import astrapy

        _, _, w, h = self._compute_volume_dimensions(voxels_x)

        w = 5.5 / 2
        vxl_sz = w / voxels_x
        h = np.ceil(h / vxl_sz) * vxl_sz
        return astrapy.vol_params([voxels_x, voxels_x, None], [-w, -w, -h], [w, w, h])

    def backward(
        self,
        proj,
        geom,
        algo="sirt",
        voxels_x=None,
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

        vol_shp, vol_min, vol_max, vxl_sz = self.vol_params(voxels_x)

        # proj = np.swapaxes(proj, 0, 1)

        # proj_mask = np.ones_like(proj)

        if col_mask:
            # col_mask = astrapy.bp(proj_mask, geom, vol_shp, vol_min, vol_max)
            # 7.5 is a lowerbound intensity in the volume after backprojecting
            # plt.figure()
            # plt.imshow(vol_mask[..., 50])
            # plt.show()
            # col_mask = (col_mask > 0.5).astype(np.float)

            from fbrct import column_mask

            # radius = int(voxels_x // 2 * col_mask_percentage)
            col_mask *= column_mask(vol_shp)

            if vol_mask is None:
                vol_mask = col_mask
            else:
                vol_mask *= col_mask

        # plt.figure()
        # plt.imshow(vol_mask[..., 50])
        # plt.show()

        algo = algo.lower()
        if algo == "sirt":
            vol_gpu = astrapy.sirt_experimental(
                projections=proj,
                geometry=geom,
                mask=vol_mask,
                volume_shape=vol_shp,
                volume_extent_min=vol_min,
                volume_extent_max=vol_max,
                iters=iters,
                min_constraint=min_constraint,
                max_constraint=max_constraint,
                use_scaling=False,
                chunk_size=200,
                algo="gpu",
                callback=callback,
            )
        elif algo == "fdk":
            vol_gpu = astrapy.fdk(
                projections=proj,
                geometry=geom,
                volume_shape=vol_shp,
                volume_extent_min=vol_min,
                volume_extent_max=vol_max,
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
