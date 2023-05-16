import argparse
from math import ceil

import numpy as np
import pyvista as pv
import astrapy.geom
from joblib import Memory

from skimage.restoration import denoise_tv_chambolle
from tqdm import tqdm

import pathlib
# path = pathlib.Path(__file__).parent.resolve()
# cachedir = str(path / "cache")
import plotting

cachedir = "/bigstore/adriaan/cache"
memory = Memory(cachedir, verbose=0)

@memory.cache
def _get_reco(recodir, t, tv_denoising=.065, x_flip=True):
    fname = recodir + f'recon_t000{t}.npy'
    reco = np.load(fname, allow_pickle=True).item()
    x = reco['volume']
    c = x.shape[2] // 2
    x = x[..., c - 350:c + 450]
    y = np.flip(x, axis=-1)
    if x_flip:
        y = np.flip(y, axis=-3)
    if tv_denoising != 0.0:
        y = denoise_tv_chambolle(y, weight=tv_denoising)
    y *= 256
    return y


def average(recodir, times: range):
    pv.set_plot_theme("document")

    reco = None
    for i, time in tqdm(enumerate(times)):
        if reco is None:
            reco = _get_reco(recodir,
                             time,
                             tv_denoising=0.0,
                             )
        else:
            reco += _get_reco(recodir,
                             time,
                             tv_denoising=0.0,
                             )
    reco /= len(list(times))

    grid = pv.UniformGrid()
    grid.dimensions = reco.shape
    grid.origin = (0., 0., 0.)
    grid.spacing = (1, 1, 1)
    grid.point_data["values"] = reco.flatten(order="F")

    from plotting import FONT, CM, COLUMNWIDTH
    dpi = 300
    # PT are defined for a DPI of 96 pixels per inch
    # 1 point = 4/3 px
    PT = 4 / 3 * dpi / 96
    pv.global_theme.font.family = FONT.lower()
    pv.global_theme.font.size = int(5 * PT)
    pv.global_theme.font.label_size = int(5 * PT)

    size = [
        ceil(.5 * COLUMNWIDTH * CM * dpi),
        ceil(8 * CM * dpi)
    ]
    plotter = pv.Plotter(off_screen=True,
                         window_size=size,
                         )

    opacity = [0.000, 0.015, 0.04, 0.1, 0.4, 1.0]
    # opacity = np.array([0.000, 0.0005, 0.01, 0.045, 0.15, .55])

    plotter.add_volume(grid,
                       clim=(0, 1.),
                       opacity=opacity,
                       show_scalar_bar=False,
                       scalar_bar_args={
                        'vertical': True
                       }
                   )

    # Add: cube_axes_actor.SetScreenSize(font_size)  # Adriaan
    # to pyvista's renderer.py
    plotter.show_grid(show_xlabels=False, show_ylabels=False,
                      show_zlabels=False,
                      xlabel="y", ylabel="x", zlabel="z",
                      # xlabel="", ylabel="", zlabel="",
                      )

    plotter.view_vector((-1, -1, 1))
    # plotter.view_isometric()
    plotter.camera.Zoom(1.5)

    fname = f'average_{times.start}_{times.stop}'
    print("Saving to " + fname)
    plotter.save_graphic(fname + '.pdf')
    plotter.screenshot(fname + '.png')
    # pdfcrop --margins '-15 -50 -110 -80' image_485_iso.pdf image_485_iso-crop.pdf
    plotter.close()


def movie(recodir, framerate, nr_frames_per_reco, ran, out='out.mp4'):
    from plotting import FONT
    pv.global_theme.font.family = FONT.lower()

    pv.set_plot_theme("dark")

    from plotting import FONT
    pv.global_theme.font.family = FONT.lower()

    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.open_movie(out, framerate=framerate)


    # image = pv.read('resources/1b_CWI_LogoCMYK.png')
    # plotter.add_background_

    # plotter.show(auto_close=False)  # only necessary for an off-screen movie
    # plotter.write_frame()  # write initial data

    for t in tqdm(ran):
        reco = _get_reco(recodir, t)

        grid = pv.UniformGrid()
        grid.dimensions = reco.shape
        grid.origin = (0., 0., 0.)
        grid.spacing = (1, 1, 1)
        grid.point_data["values"] = reco.flatten(order="F")

        # p.add_volume(grid, cmap=viridis, ambient=1.5, clim=[0, 100])

        plotter.add_volume(grid,
                           clim=(0, 1.),
                           opacity=[0.000, 0.015, 0.08, 0.2, 0.5, 1.0],
                           scalar_bar_args={'title': '',
                                            'n_labels': 2,
                                            'label_font_size': 12},
                           )
        center = (reco.shape[0] / 2,
                  reco.shape[1] / 2,
                  reco.shape[2] / 2)
        cylinder = pv.Cylinder(center=center, direction=[0, 0, 1],
                               radius=reco.shape[0] / 2, height=reco.shape[2],
                               resolution=150)
        # cylinder.plot(show_edges=True, line_width=5, cpos='xy')
        plotter.add_mesh(cylinder, show_edges=True, line_width=1,
                         style='points',
                         edge_color='white')

        # p.add_mesh(grid.contour(), opacity=.35, clim=[0, .15])
        plotter.add_text(f"Time: {t}", name='time-label',
                         font=plotting.FONT.lower())
        # plotter.update()
        # plotter.render()

        fname = f'{out}_still_{t}'
        print("Saving a still to " + fname + " in pdf and png.")
        plotter.remove_scalar_bar()
        plotter.save_graphic(fname + '.pdf')
        plotter.screenshot(fname + '.png')

        for _ in range(nr_frames_per_reco):
            plotter.write_frame()  # Write this frame

        # plotter.show(screenshot='image.png')
        plotter.clear()

    # Be sure to close the plotter when finished
    plotter.close()


def picture(recodir, time, column=True, tv_denoising=0.0):
    pv.set_plot_theme("document")
    reco = _get_reco(recodir, time, tv_denoising=tv_denoising)

    # # helpful to print to choose opacity values
    # hist, bins = np.histogram(reco, bins=np.arange(0, 110, 10))
    # print(hist)

    grid = pv.UniformGrid()
    grid.dimensions = reco.shape
    grid.origin = (0., 0., 0.)
    grid.spacing = (1, 1, 1)
    grid.point_data["values"] = reco.flatten(order="F")

    from plotting import FONT, CM, COLUMNWIDTH
    dpi = 300
    # PT are defined for a DPI of 96 pixels per inch
    # 1 point = 4/3 px
    PT = 4 / 3 * dpi / 96
    pv.global_theme.font.family = FONT.lower()
    pv.global_theme.font.size = int(5 * PT)
    pv.global_theme.font.label_size = int(5 * PT)

    if not column:
        size = [
            ceil(8 * CM * dpi),
            ceil(7 * CM * dpi)
        ]
    else:
        # I'm too lazy to script this. Change these values
        # and comment/uncomment the right plot in `map`
        size = [
            ceil(.5 * COLUMNWIDTH * CM * dpi),
            ceil(8 * CM * dpi)
        ]
        # # plotter.ren_win.OffScreenRenderingOn()

    # plotter = pv.Plotter(off_screen=True,
    #                      shape="1|2",
    #                      col_weights=[1, 2],
    #                      window_size=size,
    #                      border=False,
    #                      border_color='white'
    #                      )
    #
    # plotter.subplot(0, 0)
    # opacity = np.array([0.000, 0.0005, 0.01, 0.045, 0.15, .55])
    # plotter.add_volume(grid,
    #                    clim=(0, 1.),
    #                    opacity=opacity,
    #                    show_scalar_bar=False,
    #                    scalar_bar_args={
    #                     'vertical': True
    #                    }
    #                )
    # # Add: cube_axes_actor.SetScreenSize(font_size)  # Adriaan
    # # to pyvista's renderer.py
    # plotter.show_grid(show_xlabels=False, show_ylabels=False,
    #                   show_zlabels=False,
    #                   xlabel="y", ylabel="x", zlabel="z",
    #                   # xlabel="", ylabel="", zlabel="",
    #                   )
    # plotter.view_vector((-1, -1, 0))
    # plotter.camera.Zoom(1.5)
    #
    # plotter.subplot(1, 0)
    # plotter.add_volume(grid,
    #                    clim=(0, 1.),
    #                    opacity=opacity,
    #                    show_scalar_bar=False,
    #                    scalar_bar_args={
    #                     'vertical': True
    #                    }
    #                )
    # # Add: cube_axes_actor.SetScreenSize(font_size)  # Adriaan
    # # to pyvista's renderer.py
    # plotter.show_grid(show_xlabels=False, show_ylabels=False,
    #                   show_zlabels=False,
    #                   xlabel="y", ylabel="x", zlabel="z",
    #                   # xlabel="", ylabel="", zlabel="",
    #                   )
    # plotter.view_yz(True)
    # plotter.camera.Zoom(2.0)
    #
    # plotter.subplot(2, 0)
    # plotter.add_volume(grid,
    #                    clim=(0, 1.),
    #                    opacity=opacity,
    #                    show_scalar_bar=False,
    #                    scalar_bar_args={
    #                     'vertical': True
    #                    }
    #                )
    # # Add: cube_axes_actor.SetScreenSize(font_size)  # Adriaan
    # # to pyvista's renderer.py
    # plotter.show_grid(show_xlabels=False, show_ylabels=False,
    #                   show_zlabels=False,
    #                   xlabel="y", ylabel="x", zlabel="z",
    #                   # xlabel="", ylabel="", zlabel="",
    #                   )
    # plotter.view_xy()
    # plotter.camera.Zoom(2.0)
    #
    # plotter.save_graphic(f'image_{time}.pdf')

    plotter = pv.Plotter(off_screen=True,
                         window_size=size,
                         )
    # opacity = np.array([0.000, 0.0005, 0.01, 0.045, 0.15, .55])
    opacity = [0.0, .1, .1],

    plotter.add_volume(grid,
                       # clim=(0, 1.),
                       opacity=opacity,
                       show_scalar_bar=False,
                       scalar_bar_args={
                        'vertical': True
                       }
                   )
    # Add: cube_axes_actor.SetScreenSize(font_size)  # Adriaan
    # to pyvista's renderer.py
    plotter.show_grid(show_xlabels=False, show_ylabels=False,
                      show_zlabels=False,
                      xlabel="y", ylabel="x", zlabel="z",
                      # xlabel="", ylabel="", zlabel="",
                      )
    map = {
        'iso': plotter.view_isometric,
        'vector': plotter.view_vector,
        'xz': plotter.view_xz,  # first detector
        'yz': plotter.view_yz,
        'xy': plotter.view_xy,  # top
           }

    for key, value in map.items():
        # plotter.camera.zoom(2.0)  # doesn't work?
        if key == 'vector':
            value((-1, -1, 1))
            plotter.camera.Zoom(1.5)
        elif key == 'yz':
            value(True)
            plotter.camera.Zoom(2.0)
        else:
            value()  # isometric plot was for
            plotter.camera.Zoom(1.5)

        fname = f'image_{time}_{key}.pdf'
        print("Saving to " + fname)
        # plotter.screenshot(fname, window_size=size)
        plotter.save_graphic(fname)
        # pdfcrop --margins '-15 -50 -110 -80' image_485_iso.pdf image_485_iso-crop.pdf
        # pdfcrop --margins '-30 -200 -100 -200' image_485_xy.pdf image_485_xy-crop.pdf

    plotter.close()


def multicolor_picture(recodir, ran):
    pv.set_plot_theme("document")

    plotter = pv.Plotter() #notebook=False)#, off_screen=True)
    # plotter.open_movie(out, framerate=framerate)

    # image = pv.read('resources/1b_CWI_LogoCMYK.png')

    # plotter.show(auto_close=False)  # only necessary for an off-screen movie
    # plotter.write_frame()  # write initial data

    for i, t in tqdm(enumerate(sorted(ran), 1)):
        reco = _get_reco(recodir, t)

        grid = pv.UniformGrid()
        grid.dimensions = reco.shape
        grid.origin = (0., 0., 0.)
        grid.spacing = (1, 1, 1)
        grid.point_data["values"] = reco.flatten(order="F")

        # p.add_volume(grid, cmap=viridis, ambient=1.5, clim=[0, 100])

        print(i)
        print(len(ran))
        opacity = np.array([0.000, 0.005, 0.08, 0.1, 0.3, 1.0])
        if i == len(ran):
            plotter.add_volume(grid,
                               clim=(0, 1.),
                               opacity=opacity,
                               cmap='Greens'
                               )
        elif i == len(ran) - 1 or i == len(ran) - 2:
            plotter.add_volume(grid,
                               clim=(0, 1.),
                               opacity=opacity / 5,
                               cmap='Blues'
                               )
        elif i == len(ran) - 2:
            plotter.add_volume(grid,
                               clim=(0, 1.),
                               opacity=opacity,
                               cmap='Oranges'
                               )
        elif i == 1:
            plotter.add_volume(grid,
                               clim=(0, 1.),
                               opacity=opacity,
                               cmap='Reds'
                               )
        else:
            plotter.add_volume(grid,
                               clim=(0, 1.),
                               opacity=opacity / 5,
                               cmap='Purples'
                               )


        # p.add_mesh(grid.contour(), opacity=.35, clim=[0, .15])
        plotter.add_text(f"Time: {t}", name='time-label')
        # plotter.update()
        plotter.render()

        # for _ in range(nr_frames_per_reco):
        # plotter.write_frame()  # Write this frame

    plotter.show(screenshot='image.png')
    # plotter.show()
    # plotter.clear()


    # Be sure to close the plotter when finished
    plotter.close()


def frames(recodir, times):
    pv.set_plot_theme("document")

    for i, time in enumerate(times):
        reco = _get_reco(recodir, time, tv_denoising=0.065, x_flip=True)

        grid = pv.UniformGrid()
        grid.dimensions = reco.shape
        grid.origin = (0., 0., 0.)
        grid.spacing = (1, 1, 1)
        grid.point_data["values"] = reco.flatten(order="F")

        from plotting import FONT, CM, COLUMNWIDTH
        dpi = 300
        # PT are defined for a DPI of 96 pixels per inch
        # 1 point = 4/3 px
        PT = 4 / 3 * dpi / 96
        pv.global_theme.font.family = FONT.lower()
        pv.global_theme.font.size = int(5 * PT)
        pv.global_theme.font.label_size = int(5 * PT)

        size = [
            ceil(.5 * COLUMNWIDTH * CM * dpi),
            ceil(8 * CM * dpi)
        ]
        plotter = pv.Plotter(off_screen=True,
                             window_size=size,
                             )

        opacity = [0.000, 0.015, 0.04, 0.1, 0.4, 1.0]
        # opacity = np.array([0.000, 0.0005, 0.01, 0.045, 0.15, .55])

        plotter.add_volume(grid,
                           clim=(0, 1.),
                           opacity=opacity,
                           show_scalar_bar=False,
                           scalar_bar_args={
                            'vertical': True
                           }
                       )

        if i == 0:
            # Add: cube_axes_actor.SetScreenSize(font_size)  # Adriaan
            # to pyvista's renderer.py
            plotter.show_grid(show_xlabels=False, show_ylabels=False,
                              show_zlabels=False,
                              xlabel="y", ylabel="x", zlabel="z",
                              # xlabel="", ylabel="", zlabel="",
                              )

        plotter.view_vector((-1, -1, 1))
        # plotter.view_isometric()
        plotter.camera.Zoom(1.5)

        fname = f'frame_{time}.pdf'
        print("Saving to " + fname)
        plotter.save_graphic(fname)
        plotter.screenshot(f'frame_{time}.png')
        # pdfcrop --margins '-15 -50 -110 -80' image_485_iso.pdf image_485_iso-crop.pdf
        plotter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot fluidized beds")

    parser.add_argument(
        "--recodir",
        type=str,
        help="directory to store reconstructions",
        default="./"
    )
    parser.add_argument("--filename", type=str)
    parser.add_argument(
        "--movie",
        default=False,
        action='store_true')
    parser.add_argument(
        "--picture",
        default=False,
        action='store_true')
    parser.add_argument(
        "--frames",
        default=False,
        action='store_true')
    parser.add_argument(
        "--average",
        default=False,
        action='store_true')
    parser.add_argument(
        "--multicolor",
        default=False,
        action='store_true')
    parser.add_argument(
        "--range",
        default=False,
        action='store_true')
    parser.add_argument(
        "--time",
        type=int,
        nargs='+',
        help="Timeframes",
        default=None,
    )
    parser.add_argument("--framerate",
                        type=int, default=24)
    parser.add_argument("--frames-per-reco", type=int, default=6)

    args = parser.parse_args()

    if args.movie:
        assert len(args.time) == 2
        movie(args.recodir,
              args.framerate,
              args.frames_per_reco,
              range(args.time[0], args.time[1]+1),
              out=args.filename)

    if args.range and len(args.time) == 2:
        args.time = range(args.time[0], args.time[1]+1)

    if args.picture:
        assert len(args.time) == 1
        picture(args.recodir, args.time[0])

    if args.frames:
        frames(args.recodir, args.time)

    if args.average:
        average(args.recodir, args.time)

    if args.multicolor:
        multicolor_picture(args.recodir, args.time)