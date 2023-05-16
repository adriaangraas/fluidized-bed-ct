import argparse

import matplotlib
from math import ceil

import numpy as np
import pyvista as pv
# define a categorical colormap
from matplotlib.colors import ListedColormap

from settings import *
from skimage.restoration import denoise_tv_chambolle
from fbrct import column_mask
from fbrct.util import plot_nicely


def plot(filename,
         use_gt=False,
         plot_volume=True,
         plot_figures=False,
         plot_frame=True,
         central_slice=None,
         slices_xyz=None):
    reco = np.load(filename, allow_pickle=True).item()
    column = reco['name'] != 'Numerical script outcome'

    geoms = reco['geometry']
    if use_gt:
        x = reco['gt']
    else:
        x = reco['volume']

    # x -= reco['gt'] # to compute bias
    if central_slice is None:
        c = x.shape[2] // 2
    else:
        c = central_slice
    # c = x.shape[2] // 2 - 115  # central plane for 10mm_23mm_horizontal
    v = 80
    # x = x[..., c - v: c + v]
    import pyqtgraph as pq
    pq.image(np.transpose(x, [2, 1, 0]))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.show()


    # mask = column_mask(x.shape)
    # x[mask == 0.] = np.nan

    # # x = zoom(x, .5)  # watch out: introduces negative numbers
    # x = np.abs(x) + np.finfo(float).eps
    # x = np.round(x, decimals=1)
    #
    # # x = x[..., 50:x.shape[2] // 2 + 50]
    x = np.flip(x, axis=2)
    # x = np.rot90(x, axes=(0,1))

    # # plt.figure()
    # plt.imshow(denoise_tv_chambolle(x[..., 300], weight=0.5))
    # # plt.show()

    # y = denoise_tv_chambolle(x, weight=.15)
    y = np.copy(x)
    # y = np.rot90(x, k=1, axes=(0, 1))
    y = np.swapaxes(y, 0, 1)
    y *= 256
    # y[:y.shape[0], ...] = 0.

    grid = pv.UniformGrid()
    grid.dimensions = y.shape
    grid.origin = (0., 0., 0.)
    grid.spacing = (1, 1, 1)
    grid.point_data["values"] = y.flatten(order="F")

    if slices_xyz is None:
        slices = [grid.slice_orthogonal()[i] for i in (0, 2,)]
    else:
        # the center can be obtainedsli by inspecting
        # grid.slice_orthogonal().center
        slices = [grid.slice_orthogonal(
            x=slices_xyz[0], y=slices_xyz[1], z=slices_xyz[2]
        )[i] for i in (0, 2,)]

    slices_colors = ['red', 'blue']
    # slices = []
    outline_color = 'red'
    pv.set_plot_theme("document")
    from plotting import CM, FONT, FONT_SIZE
    dpi = 300

    # PT are defined for a DPI of 96 pixels per inch
    # 1 point = 4/3 px
    PT = 4 / 3 * dpi / 96

    pv.global_theme.font.family = FONT.lower()
    pv.global_theme.font.size = int(5 * PT)
    pv.global_theme.font.label_size = int(5 * PT)
    plotter = pv.Plotter(lighting='three lights', off_screen=True)
    if plot_frame:
        plotter.add_mesh(grid.outline(), opacity=0.0)
    # plotter.enable_parallel_projection()

    if plot_figures:
        for s, c in zip(slices, slices_colors):
            plotter.add_mesh(s.outline(), color=c, line_width=3)
            # plotter.add_mesh(s, opacity=.95)  # cmap='binary', lighting=True)

    if plot_volume:
        # n_contours = 8
        # cmap = plt.cm.get_cmap("viridis", n_contours)
        # contours = grid.contour(
        #     isosurfaces=n_contours)  # , method='marching_cubes')
        # plotter.add_mesh(contours, opacity=.95, cmap=cmap, clim=[0, 1])

        plotter.add_volume(
            grid,
            # clim=(0, 1.),
            # opacity=[0.000, 0.015, 0.08, 0.2, 0.5, 0.5],
            # opacity='linear'
            opacity=[0.0, .1, .1],
        )

        # p.show_bounds(contours, grid=False)
        # p.add_volume(grid)
        # opacity = [0.0, .5, .51, .51, .51, 0]
        # p.add_volume(grid, cmap="bone", opacity_unit_distance=.1)
        if plot_frame:
            plotter.show_grid(show_xlabels=False, show_ylabels=False,
                              show_zlabels=False,
                              xlabel="", ylabel="", zlabel="")
            plotter.show_grid(show_xlabels=False, show_ylabels=False,
                              show_zlabels=False,
                              xlabel="y", ylabel="x", zlabel="z")

        # enable below to add paper colorbar
        # plotter.remove_scalar_bar('values')
        # plotter.add_scalar_bar(title="",
        #                        label_font_size=8,
        #                        use_opacity=False,
        #                        # n_colors=n_contours,
        #                        n_labels=2,
        #                        fmt="%0.1f",
        #                        position_x=0.05,
        #                        position_y=0.90,
        #                        width=0.10,
        #                        # height=0.15,
        #                        vertical=False,
        #                        )

        # plotter.add_volume(
        #     grid,
        #     clim=(0, 1.),
        #     # opacity=[0.000, 0.015, 0.08, 0.2, 0.5, 0.5],
        #     # opacity='linear'
        #     opacity=[0.0, .1, .1],
        #     cmap='gray_r'
        # )
        # plotter.add_scalar_bar(title="",
        #                        label_font_size=8,
        #                        use_opacity=False,
        #                        # n_colors=n_contours,
        #                        n_labels=2,
        #                        fmt="%0.1f",
        #                        position_x=0.05,
        #                        position_y=0.70,
        #                        width=0.10,
        #                        # height=0.15,
        #                        vertical=False,
        #                        )

        plotter.remove_scalar_bar()
        cam = plotter.camera
        if not column:
            size = [
                ceil(8 * CM * dpi),
                ceil(7 * CM * dpi)
            ]
        else:
            size = [
                ceil(4 * CM * dpi),
                ceil(3.5 * CM * dpi)
            ]
        # # plotter.ren_win.OffScreenRenderingOn()
        fname = 'contour'
        print(f"Saving to {fname}.pdf...")
        plotter.save_graphic(fname + '.pdf')
        # plotter.screenshot(fname + '.jpg',
        #                    window_size=size,
        #                    # transparent_background=True,
        #                    )
        plotter.show()

        # for i, (s, c) in enumerate(zip(slices, slices_colors)):
        #     print(i)
        #     plotter = pv.Plotter(lighting='three lights')
        #     plotter.enable_parallel_projection()
        #     plotter.theme.transparent_background = True
        #     plotter.add_light(pv.Light(light_type='headlight'))
        #     # plotter.camera = cam
        #     plotter.add_mesh(s.outline(), color=c, line_width=3)
        #     plotter.add_mesh(s)  # cmap='binary', lighting=True)
        #     plotter.remove_scalar_bar('values')
        #     if i == 1:
        #         #     plotter.add_scalar_bar(title="", label_font_size=20, use_opacity=True,
        #         #                      # n_colors=n_contours,
        #         #                      n_labels=2,
        #         #                      fmt="%0.1f",
        #         #                      position_x=0.9,
        #         #                      position_y=0.1,
        #         #                      width=0.05,
        #         #                      height=0.8,
        #         #                      vertical=True)
        #         ws = 500
        #     else:
        #         ws = 400
        #     plotter.show(screenshot=f'contour_slice{i}.png',
        #                  window_size=[1000, 1000])

    if plot_figures:
        # # grid.plot(show_edges=True)
        # slices = grid.slice_orthogonal()[2]
        # pv.set_plot_theme("document")
        # plotter = pv.Plotter()
        # cpos = [
        #     (540.9115516905358, -617.1912234499737, 180.5084853429126),
        #     # (128.31920055083387, 126.4977720785509, 111.77682599082095),
        #     # (-0.1065160140819035, 0.032750075477590124, 0.9937714884722322)
        # ]
        # plotter.add_mesh(slices)  # , cmap='gist_ncar_r')
        # # p.show_grid()
        # plotter.show()
        # # cmap = plt.cm.get_cmap("viridis", 4)
        # # grid.plot(cmap=cmap)
        # # slices.plot(cmap=cmap)

        # 5 cm inner diameter column = 2.5 cm radius
        voxel_size_x = reco['vol_params'][3][0]
        vol_mask = column_mask(x.shape,
                               r=int(np.ceil(2.5 / voxel_size_x)))
        vol_mask = vol_mask == 0
        mask_vol = np.ma.masked_where(~vol_mask, vol_mask)
        mask_cmap = ListedColormap(['white'])
        mask_cmap._init()
        mask_cmap._lut[:-1, -1] = 1.0

        from mpl_toolkits.axes_grid1 import Divider, Size

        from plotting import plt, TEXTWIDTH, CM
        # fig, axs = plt.subplots(2, 1,
        #                         figsize=(6 * CM, 7.0 * CM),
        #                         # gridspec_kw={ # manually adjusted
        #                         #     'height_ratios': [1, 1.875]
        #                         # }
        #                         )
        fig = plt.figure(figsize=(6 * CM, 7 * CM))

        # https://matplotlib.org/stable/gallery/axes_grid1/demo_fixed_size_axes.html
        # The first & third items are for padding and the second items are for the
        # axes. Sizes are in inches.
        h = [Size.Fixed(1.0), Size.Scaled(1.), Size.Fixed(.2)]
        v = [Size.Fixed(0.7), Size.Scaled(1.), Size.Fixed(.5)]

        divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
        # The width and height of the rectangle are ignored.

        # axs = [None, None]
        # axs[0] = fig.add_axes(divider.get_position(),
        #                       axes_locator=divider.new_locator(nx=1, ny=1))
        # axs[1] = fig.add_axes(divider.get_position(),
        #                       axes_locator=divider.new_locator(nx=1, ny=1))
        # axs[0].set_adjustable('box')
        # axs[1].set_adjustable('box')
        # plt.cla()


        plot_nicely(
            plt.gca(),
            # x[:, x.shape[1] // 2],
            x[:, int(slices_xyz[1])],  # TODO: hacky, use contour
            0, 1.0,
            [0.01714] * 3,
            'viridis',
            with_lines=False,
            # mask=mask_vol[x.shape[0] // 2],
            # mask_cmap=mask_cmap,
            xlabel=None,
            ylabel='z',
            geoms=geoms,
            spine_color=slices_colors[0]
        )
        plt.gca().tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        # axs[0].xlabel(None)

        # plt.tight_layout(h_pad=1.08, w_pad=0.01)
        # plt.subplot_tool()
        print(f"Saving out/contour_slices_vert.pdf...")
        plt.savefig(f"out/contour_slices_vert.pdf")
        plt.show()

        plt.figure(figsize=(6 * CM, 7 * CM))
        plot_nicely(
            plt.gca(),
            x[..., x.shape[2] // 2],
            0, 1.0,
            [0.01714] * 3,
            'viridis',
            with_lines=True,
            xlabel='x',
            # mask=mask_vol[..., x.shape[2] // 2],
            # mask_cmap=mask_cmap,
            geoms=geoms,
            spine_color=slices_colors[1]
        )
        plt.gca().tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True)  # labels along the bottom edge are off

        plt.subplots_adjust(
            left=0.00,
            bottom=0.0,
            right=1.0,
            top=1.0,
            wspace=0.00,
            hspace=0.00)

        # plt.tight_layout(h_pad=1.08, w_pad=0.01)
        # plt.subplot_tool()
        print(f"Saving out/contour_slices_horiz.pdf...")
        plt.savefig(f"out/contour_slices_horiz.pdf")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot 3D reconstruction files")
    parser.add_argument("--filename", type=str)
    parser.add_argument("--plot-slices", action='store_true', default=False)
    parser.add_argument("--no-frame", action='store_true', default=False)
    parser.add_argument("--central-slice", type=int, default=None)
    parser.add_argument(
        "--slices-xyz",
        type=float,
        nargs=3,
        help="Slice x, y, z coordinates",
        default=None,
    )
    args = parser.parse_args()
    plot(args.filename,
         plot_volume=True,
         plot_figures=args.plot_slices,
         plot_frame=not args.no_frame,
         central_slice=args.central_slice,
         slices_xyz=args.slices_xyz)