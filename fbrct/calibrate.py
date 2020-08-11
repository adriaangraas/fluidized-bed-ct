import copy
import warnings

import numpy as np
import transforms3d


def triangle_geom(src_rad, det_rad):
    gms = []
    for src_a in [0, 2 / 3 * np.pi, 4 / 3 * np.pi]:
        det_a = src_a + np.pi  # opposing
        src = src_rad * np.array([np.cos(src_a), np.sin(src_a), 0])
        det = det_rad * np.array([np.cos(det_a), np.sin(det_a), 0])
        geom = StaticGeometry.fromOrthogonal(
            source=src,
            detector=det,
            # u=np.array([np.cos(det_a +np.pi / 2), np.sin(det_a + np.pi/2), 0]),
            # v=np.array([0, 0, 1])
        )
        gms.append(geom)

    return gms


def cube_points(w=1., d=2., h=4., optimize=False):
    points = [
        [0, 0, 0], [0, d, 0],
        [w, 0, 0], [w, d, 0],
        [0, 0, h], [0, d, h],
        [w, 0, h], [w, d, h],
    ]

    # shift points to center
    for p in points:
        p[0] -= w / 2
        p[1] -= d / 2
        p[2] -= h / 2

    point_objs = []
    for point in points:
        point_objs.append(Point(point, optimize))

    return point_objs


def triangle_column_points(rad=4., height=4., start_angle=0., num_angles=3,
                           optimize=False):
    # (0,0) is in the center of the triangle at the base
    angles = np.linspace(start_angle, start_angle + 2 * np.pi,
                         num=num_angles, endpoint=False)

    points = []
    for a in range(len(angles)):
        p = Point(
            [rad * np.cos(angles[a]), rad * np.sin(angles[a]), -height / 2],
            optimize)
        points.append(p)
        p = Point(
            [rad * np.cos(angles[a]), rad * np.sin(angles[a]), height / 2],
            optimize)
        points.append(p)

    return points


class Parameter:
    """A delayable value with optimization information."""

    def __init__(self, value, optimize: bool = True, bounds=None):
        """
        :param value: Can also be `Callable` to delay computation.
        :param optimize: Set to `False` to disable fitting procedure.
        :param bounds: `None` or a `tuple` of ndarray.
        """
        self._value = value
        self.optimize = optimize

        if bounds is None:
            lower = [-np.inf] * self.__len__()
            upper = [np.inf] * self.__len__()
            bounds = [lower, upper]

        self.bounds = bounds

    @property
    def value(self):
        if callable(self._value):
            return self._value()

        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        assert len(self) > 0

    def __len__(self):
        if np.isscalar(self._value):
            return 1

        return len(self._value)


class Point(Parameter):
    """A 3-vector ndarray that allows optimization.

    Here `optimize` has to be given explicitly because I can see use cases
    where points in the reconstruction volume are at known and unknown
    locations, so I don't want to implicitly choose one.
    """

    def __init__(self, value, optimize: bool, bounds=None):
        if issubclass(type(value), list):
            value = np.array(value)

        if not isinstance(value, np.ndarray):
            raise TypeError("`value` must be a `numpy.ndarray`.")

        if not len(value) == 3:
            raise ValueError("`np.ndarray` must have length 3.")

        super(Point, self).__init__(value, optimize, bounds)


class StaticGeometry:
    """Geometry description for single angle without moving parts.

    Internally the geometry is encoded by a source and detector vector,
    and a roll-pitch-yaw convention. Note: the RPY convention is with
    respect to the _global_ "world" coordinates, and not with respect
    to the detector plane (spanned by u,v).

    TODO: I'm not sure if either choice for RPY leads to an optimization
        problem that is simpler to solve.

    TODO: I'll probably have to make some changes to differentiate between
        the detector location and the (u,v) starting points when the ROI of
        a detector is not centered around the middle of the detector. This
        would cause a misplaced origin in the detector image.
    """
    ANGLES_CONVENTION = "sxyz"
    DETECTOR_Y_VECTOR = np.array([0, 1, 0])
    DETECTOR_Z_VECTOR = np.array([0, 0, 1])

    def __init__(
        self,
        source: np.ndarray,
        detector: np.ndarray,
        roll=0.,
        pitch=0.,
        yaw=0.,
        source_bounds=None,
        detector_bounds=None,
    ):
        """Initialialize geometry from world RPY coordinates"""
        self.source_param = Parameter(source, bounds=source_bounds)
        self.detector_param = Parameter(detector, bounds=detector_bounds)

        # TODO: sensible bounds should be on intrinic
        self.roll_param = Parameter(roll)
        self.pitch_param = Parameter(pitch)
        self.yaw_param = Parameter(yaw)

    def rotation_matrix(self):
        n = np.cross(self.u, self.v)
        return np.array([n, self.u, self.v]).T

    @property
    def source(self):
        return self.source_param.value

    @property
    def detector(self):
        return self.detector_param.value

    @property
    def roll(self):
        return self.roll_param.value

    @property
    def pitch(self):
        return self.pitch_param.value

    @property
    def yaw(self):
        return self.yaw_param.value

    @property
    def u(self):
        R = self.angles2mat(self.roll, self.pitch, self.yaw)
        return R @ self.DETECTOR_Y_VECTOR

    @property
    def v(self):
        R = self.angles2mat(self.roll, self.pitch, self.yaw)
        return R @ self.DETECTOR_Z_VECTOR

    @classmethod
    def fromOrthogonal(cls, source, detector, roll=0., pitch=0., yaw=0.):
        """With intrinsic RPY"""
        vec = detector - source

        # Gram-Schmidt orthogonalize
        v = cls.DETECTOR_Z_VECTOR - cls.DETECTOR_Z_VECTOR.dot(vec) * vec
        len_v = np.linalg.norm(v)

        if len_v == 0.:
            raise NotImplementedError(
                "Geometries with perfectly vertical source-"
                "detector line are not supported, because I didn't want"
                " to write unpredictable logic. Upright your geom or"
                " check if axes correspond to [x, y, z].")

        v /= len_v
        u = np.cross(vec, v)  # right-hand-rule = right direction?
        u /= np.linalg.norm(u)

        return cls.fromDetectorVectors(source, detector, u, v, roll, pitch,
                                       yaw)

    @classmethod
    def fromDetectorVectors(cls, source, detector, u, v, roll=0., pitch=0.,
                            yaw=0.):
        """Initiate from any pair of detector vectors u, v with intrinsic angles."""

        # convert u,v to global RPY matrix
        n = np.cross(u, v)
        R = np.array([n, u, v])

        # roll this basis
        R_intrinsic = cls.angles2mat(roll, pitch, yaw)
        R_rolled = R_intrinsic.T @ R

        # get global coordinates
        r, p, y = cls.mat2angles(R_rolled)
        return cls(source, detector, r, p, y)

    @staticmethod
    def angles2mat(r, p, y) -> np.ndarray:
        return transforms3d.euler.euler2mat(
            r, p, y,
            StaticGeometry.ANGLES_CONVENTION
        )

    @staticmethod
    def mat2angles(mat) -> tuple:
        return transforms3d.euler.mat2euler(
            mat,
            StaticGeometry.ANGLES_CONVENTION
        )

    def parameters(self):
        return [self.source_param,
                self.detector_param,
                self.roll_param,
                self.pitch_param,
                self.yaw_param]


def rotate_points(points, roll=0., pitch=0., yaw=0.):
    """In-place rotation of points"""

    R = StaticGeometry.angles2mat(roll, pitch, yaw)
    for p in points:
        if isinstance(p, Point):
            p.value = R @ p.value
        elif isinstance(p, np.ndarray):
            p[:] = R @ p
        else:
            raise TypeError("Values in `points` have to be `Point` or"
                            " `ndarray`.")


def xray_project(geom: StaticGeometry, point: np.ndarray):
    """X-ray projection

    Not sure what concensus is, but I found the geometric method quite
    confusing so I was thinking this would be simple as well.

    Consider the ray going through the source `s` and point `p`:
        r(t) = s + t * (p-s)
    The ray is parametrized by the scalar `t`. `s` and `p` are known vectors.

    Now consider the detector plane equation, an affine linear space:
        0 = d + y*u + z*v
    where `d` is the detector midpoint vector, and `u` and `v` are a orthogonal
    basis for the detector plane. `y` and `z` are again scalars that parametrize
    the plane. Again, `u`, `v` and `d` are known.

    We are now looking to solve _where_ the ray hits the detector plane. This
    is we want to find {t, y, z} that solve
        s + t*(p-s) = d + y*u + z*v,
    or, equivalently,
        t*(s-p) + y*u + z*v = s - d.

    I wouldn't know how to differentiate to a solution a linear system of
    equations, but here we are lucky. We already have an orthogonal basis,
        (u, v, u cross v)
    and hence we know that if we rotate+shift to a basis so that the detector
    is the central point and u=(0,1,0) and v=(0,0,1) the system becomes:
        t*(s-p) + y*(0,1,0) + z*(0,0,1) = s - d;
        t*(s-p) + (0,y,0) + (0,0,z) = s - d.
    Of which the first solution for `t` is free:
        t*(s-p)[0] = (s-d)[0] => t = (s-p)[0]/(s-d)[0]
    Now having `t`, we substitute to get the other two equations:
        => y =  ...
        => z =  ...

    I expect AD to have little trouble differentiating all this.
    """
    R = geom.rotation_matrix()

    # get `p-s`, `s-d`, `d` transformed
    p = np.dot(R, point - geom.detector)
    s = np.dot(R, geom.source - geom.detector)

    # solve ray parameter
    t = s[0] / (s - p)[0]

    # get (0, y, z) in the detector basis
    y = s[1] + t * (p - s)[1]
    z = s[2] + t * (p - s)[2]

    return np.array((y, z))


def xray_multigeom_op(geoms, points):
    """TODO: vectorize"""
    data = []
    for g in geoms:
        projs = []
        for p in points:
            v = p.value if isinstance(p, Point) else p
            projs.append(xray_project(g, v))

        data.append(projs)

    return data


def params2ndarray(params, optimizable_only=True, key='value'):
    """Packs a list of Packable into ndarray, and returns a list of types
    to restore to.

    :param params:
    :return:
    """
    # compute array length
    length = 0
    for p in params:
        if not issubclass(type(p), Parameter):
            warnings.warn("A value in `params` is not of type `Parameter`. "
                          "The value is ignored.", UserWarning)
            continue

        if optimizable_only and p.optimize is False:
            continue

        length += len(p)

    assert length > 0

    out = np.empty(length)
    idx = 0
    for p in params:
        if not issubclass(type(p), Parameter):
            continue

        if optimizable_only and p.optimize is False:
            continue

        len_p = len(p)
        if key == 'value':
            store = p.value
        elif key == 'min_bound':
            store = p.bounds[0]
        elif key == 'max_bound':
            store = p.bounds[1]
        else:
            return ValueError

        out[idx:idx + len_p] = store
        idx += len_p

    assert idx == length

    return out


def update_params(params, x: np.ndarray, optimizable_only=True):
    """In-place updating a list of parameters.

    Expect `params` and `x` to be given in the same order as that they were
    when they `params` was turned into an array.
    """
    idx = 0
    for p in params:
        if not issubclass(type(p), Parameter):
            continue

        if optimizable_only and p.optimize is False:
            continue

        len_p = len(p)
        assert len_p != 0
        p.value = x[idx: idx + len_p]
        idx += len_p

    assert idx == len(x)


class OptimizationProblem:
    def __init__(self, points, geoms, data):
        self.points = points
        self.geoms = geoms
        self.data = np.array(data).flatten()

    def params(self):
        params = copy.copy(self.points)  # prevents repetitive adding to points
        for g in self.geoms:
            for p in g.parameters():
                params.append(p)

        return params

    def bounds(self):
        params = self.params()
        return (
            params2ndarray(params, key='min_bound'),
            params2ndarray(params, key='max_bound')
        )

    def update(self, x):
        update_params(self.params(), x)
        return self.geoms, self.points

    def __call__(self, x: np.ndarray):
        """Optimization call"""
        self.update(x)  # param restore values from `x`
        projs = np.array(xray_multigeom_op(self.geoms, self.points)).flatten()
        return projs - self.data


def plot_scatter3d(points):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.scatter(xs, ys, zs)
    plt.show()


def plot_scatter2d(*points, det=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    from matplotlib.patches import Rectangle

    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    if det is not None:
        plt.xlim(-1.2 * det[0], 1.2 * det[0])
        plt.ylim(-1.2 * det[1], 1.2 * det[1])

    for set, m in zip(points, ['o', 'x', '*']):
        ys = [p[0] for p in set]
        zs = [p[1] for p in set]
        plt.scatter(ys, zs, marker=m)

    if det is not None:
        rect = Rectangle((-det[0], -det[1]), 2 * det[0], 2 * det[1],
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
