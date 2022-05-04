import matplotlib.pyplot as plt
import pyvista as pv
# define a categorical colormap
from matplotlib.colors import ListedColormap

from settings import *

plt.style.use('dark_background')

from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
from fbrct import column_mask
from fbrct.util import plot_nicely

# reco = np.load('out/two_bubbles_200iters.npy', allow_pickle=True).item()
reco = np.load('out/two_bubbles_halfhiding_200iters.npy',
               allow_pickle=True).item()
reco = np.load('out/bubble_full_200iters.npy', allow_pickle=True).item()
reco = np.load('out/full_hiding_200iters.npy', allow_pickle=True).item()
reco = np.load('out/gradient_sphere.npy', allow_pickle=True).item()
reco = np.load('out/25gradient.npy', allow_pickle=True).item()
reco = np.load('out/halfgradient.npy', allow_pickle=True).item()
reco = np.load('out/halfintensity.npy', allow_pickle=True).item()
reco = np.load('out/projnoise.npy', allow_pickle=True).item()
reco = np.load('out/volnoise.npy', allow_pickle=True).item()
geoms = reco['geometry']
x = reco['volume']
x = reco['gt']
c = x.shape[2] // 2

# reco = np.load('out/recon_t001100.npy', allow_pickle=True).item()
# reco = np.load('out/recon_t001020.npy', allow_pickle=True).item()
# reco = np.load('out/out_shape_script.npy', allow_pickle=True).item()
# reco = np.load('out/two_spheres.npy', allow_pickle=True).item()
reco = np.load('out/size_300_algo_sirt_iters_200/recon_t000100.npy', allow_pickle=True).item()
reco = np.load(
    'out/2021-08-24_10mm_23mm_horizontal/size_300_algo_sirt_iters_200.npy',
    allow_pickle=True).item()
scan = get_scans(reco['name'])[0]
# [c[0] for i, c in scan.geometry.items()]
geoms = reco['geometry']
x = reco['volume']
# c = x.shape[2] // 2 - 100
c = x.shape[2] // 2 - 115  # central plane for 10mm_23mm_horizontal
v = 100
x = x[..., c - v: c + v]


# 5 cm inner diameter column = 2.5 cm radius
voxel_size_x = reco['vol_params'][3][0]
vol_mask = column_mask(x.shape, radius=int(np.ceil(2.5 / voxel_size_x)))
vol_mask = vol_mask == 0
mask_vol = np.ma.masked_where(~vol_mask, vol_mask)
mask_cmap = ListedColormap(['black'])
mask_cmap._init()
mask_cmap._lut[:-1, -1] = 1.0

# mask = column_mask(x.shape)
# x[mask == 0.] = np.nan

# # x = zoom(x, .5)  # watch out: introduces negative numbers
# x = np.abs(x) + np.finfo(float).eps
# x = np.round(x, decimals=1)
#
# # x = x[..., 50:x.shape[2] // 2 + 50]
# x = np.flip(x, axis=2)
# x = np.rot90(x, axes=(0,1))


# # plt.figure()
# plt.imshow(denoise_tv_chambolle(x[..., 300], weight=0.5))
# # plt.show()

y = denoise_tv_chambolle(x, weight=.15)
y = x
# y = np.rot90(x, k=1, axes=(0, 1))
y = np.swapaxes(x, 0, 1)
# y[:y.shape[0], ...] = 0.


def rec(t):
    reco = np.load(
        f'/home/adriaan/data/size_300_algo_sirt_iters_200/recon_t000{t}.npy',
        allow_pickle=True).item()
    x = reco['volume']
    c = x.shape[2] // 2
    x = x[..., c - 450:c + 350]
    y = np.flip(x, axis=-1)

    y *= 100
    # y = np.clip(y, 0, 256)

    grid = pv.UniformGrid()
    grid.dimensions = y.shape
    grid.origin = (0., 0., 0.)
    grid.spacing = (1, 1, 1)
    grid.point_data["values"] = y.flatten(order="F")
    # grid.plot(show_edges=True)


    viridis = plt.cm.get_cmap("viridis")
    newcolors = viridis(np.linspace(0, 0.50 * 100, 256))
    cmap = ListedColormap(newcolors)

    pv.set_plot_theme("dark")
    p = pv.Plotter()
    # vol = examples.download_knee_full()
    p.add_volume(grid, cmap=cmap, ambient=1.5, clim=[0, 100])
    # p.add_mesh(grid.contour(), opacity=.35, clim=[0, .15])
    p.show()
    p.clear()
    del reco
    del x
    del y
    del grid
    del p

# 155 - 161 describes a big bubble hunting down a smaller bubble and coalescence
rec(121)
for t in range(192, 200, 1):
    rec(t)

for t in range(121, 131, 1):
    rec(t)

n_contours = 8
contours = grid.contour(isosurfaces=n_contours)  # , method='marching_cubes')
slices = [grid.slice_orthogonal()[i] for i in (0, 2)]
outline_color = 'red'
cmap = plt.cm.get_cmap("viridis", n_contours)
pv.set_plot_theme("dark")
p = pv.Plotter(lighting='three lights')
p.add_mesh(grid.outline(), opacity=0.0)
p.add_mesh(contours, opacity=.95, cmap=cmap, clim=[0, 1])
# p.show_bounds(contours, grid=False)
# p.add_volume(grid)
# opacity = [0.0, .5, .51, .51, .51, 0]
# p.add_volume(grid, cmap="bone", opacity_unit_distance=.1)
p.show_grid(show_xlabels=False, show_ylabels=False, show_zlabels=False,
            xlabel="y", ylabel="x", zlabel="z")
p.remove_scalar_bar()
p.add_scalar_bar(title="", label_font_size=20, use_opacity=True,
                 # n_colors=n_contours,
                 n_labels=2,
                 fmt="%0.1f",
                 position_x=0.9,
                 position_y=0.05,
                 width=0.05,
                 height=0.75,
                 vertical=True
                 )

for s in slices:
    p.add_mesh(s.outline(), color=outline_color)
cam = p.camera
# p.save_graphic('contour.svg')
p.show(screenshot='contour.png', window_size=[500, 400])

# for i, s in enumerate(slices):
#     pv.set_plot_theme("dark")
#     p = pv.Plotter()
#     p.theme.transparent_background = True
#     p.add_light(pv.Light(light_type='headlight'))
#     p.camera = cam
#     p.add_mesh(s.outline(), color=outline_color, line_width=3)
#     p.add_mesh(s) # cmap='binary', lighting=True)
#     p.remove_scalar_bar('values')
#     if i == 1:
#         p.add_scalar_bar(title="", label_font_size=20, use_opacity=True,
#                          # n_colors=n_contours,
#                          n_labels=2,
#                          fmt="%0.1f",
#                          position_x=0.9,
#                          position_y=0.1,
#                          width=0.05,
#                          height=0.8,
#                          vertical=True)
#         ws = 500
#     else:
#         ws = 400
#     p.show(screenshot=f'contour_slice{i}.png', window_size=[ws, 400])


# grid.plot(show_edges=True)
slices = grid.slice_orthogonal()[2]
pv.set_plot_theme("document")
p = pv.Plotter()
cpos = [
    (540.9115516905358, -617.1912234499737, 180.5084853429126),
    # (128.31920055083387, 126.4977720785509, 111.77682599082095),
    # (-0.1065160140819035, 0.032750075477590124, 0.9937714884722322)
]
p.add_mesh(slices)  # , cmap='gist_ncar_r')
# p.show_grid()
p.show()
# cmap = plt.cm.get_cmap("viridis", 4)
# grid.plot(cmap=cmap)
# slices.plot(cmap=cmap)

plot_nicely(
    None,
    x[..., x.shape[2] // 2],
    0, 1.0,
    [0.01714] * 3,
    'viridis',
    with_lines=True,
    mask=mask_vol[..., x.shape[2] // 2],
    mask_cmap=mask_cmap,
    geoms=geoms)

plot_nicely(
    None,
    x[:, x.shape[1] // 2],
    0, 1.0,
    [0.01714] * 3,
    'viridis',
    with_lines=False,
    mask=mask_vol[x.shape[0] // 2],
    mask_cmap=mask_cmap,
    ylabel='z',
    geoms=geoms)

# from mayavi import mlab
# mlab.options.backend = 'envisage'
# # y = x
# n = 5
# # y[-n:n, ...] = 0.
# # y[:, -n:n, :] = 0.
# y[..., -1] = 0.
# y[..., -n:] = 0.
# y[:, -n:, :] = 0.
# y[:, :n, :] = 0.
# y[:n, :, :] = 0.
# y[-n:, :, :] = 0.
#
# # y[...] = 0.
#
# # y[..., :y.shape[2]//2] = 0.
#
# size = (x.shape[0], x.shape[2])
# # fg, bg = (0.,0.,0.), (1., 1., 1.)
# fg, bg = (1.,1.,1.), (0., 0., 0.)
# fig = mlab.figure(
#     bgcolor=bg,
#     fgcolor=fg,
#     size=size)
#
# lw = 4.0
# tr = 0.5
#
# contours = 7
# z = np.copy(y)
# # z[: :y.shape[0]//2, :] = 0.
# # z[y.shape[1]//2:, ...] = 0.
# cntrs = mlab.contour3d(z,
#                 contours=contours,
#                 opacity=0.15,
#                 # transparent=True,
#                 vmax=1.,
#                 vmin=0.
#             )
#
# mlab.colorbar(
#     cntrs,
#     orientation='horizontal',
#     nb_colors=contours,
#     nb_labels=contours+1,
#     label_fmt='%.2f')
# # # mlab.outline(line_width=4.0)
#
# R = float(x.shape[0] // 2)
# C = float(x.shape[0] // 2)
# H = x.shape[2]
# #
# # for h in (0, H):
# # # for h in (H / 2 + 1,):
# #     phi = np.arange(0.0, 2*np.pi, .01)
# #     px = R * np.sin(phi) + C
# #     py = R * np.cos(phi) + C
# #     pz = np.copy(phi)
# #     pz.fill(h)
# #     pl = mlab.plot3d(px, py, pz, color=fg,
# #                      tube_radius=tr)
# #
# # for t in (0, np.pi):
# #     phi = np.arange(0.0, H, .01)
# #     px = R * np.sin(np.ones_like(phi) * t) + C
# #     py = R * np.cos(np.ones_like(phi) * t) + C
# #     pz = phi
# #     mlab.plot3d(px, py, pz, color=fg,
# #                 representation='surface',
# #                 tube_radius=tr)
#
# # # Plotting two cut planes
# # src = mlab.pipeline.scalar_field(y, vmin=0.0, vmax=1.0, mask=mask.astype(bool))
# # src.scalar_data[mask == 0.] = np.nan
# # # cp2 = mlab.pipeline.scalar_cut_plane(src, plane_orientation='y_axes')
# # # cp2.implicit_plane.widget.enabled = False
# # cp3 = mlab.pipeline.scalar_cut_plane(src, plane_orientation='z_axes')
# # cp3.implicit_plane.widget.enabled = False
# # # cp3.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
# # # cp3.update_pipeline()
# # cp1 = mlab.pipeline.scalar_cut_plane(src, plane_orientation='x_axes')
# # cp1.implicit_plane.widget.enabled = False
#
# # mlab.orientation_axes()
# mlab.outline(line_width=lw)
# # mlab.view(azimuth=0, elevation=0)  # top
# # mlab.view(azimuth=0, elevation=90)
# mlab.view(azimuth=0, elevation=60)
# mlab.show()
