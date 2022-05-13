import argparse
import numpy as np
import pyvista as pv
import astrapy.geom
from joblib import Memory

from skimage.restoration import denoise_tv_chambolle
from tqdm import tqdm

import pathlib
# path = pathlib.Path(__file__).parent.resolve()
# cachedir = str(path / "cache")
cachedir = "/bigstore/adriaan/cache"
memory = Memory(cachedir, verbose=0)

@memory.cache
def _get_reco(recodir, t):
    fname = recodir + f'recon_t000{t}.npy'
    reco = np.load(fname, allow_pickle=True).item()
    x = reco['volume']
    c = x.shape[2] // 2
    x = x[..., c - 450:c + 350]
    y = np.flip(x, axis=-1)
    y = denoise_tv_chambolle(y, weight=.15)
    y *= 256
    return y


def movie(recodir, framerate, nr_frames_per_reco, ran, out='out.mp4'):
    pv.set_plot_theme("dark")

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
        plotter.add_text(f"Time: {t}", name='time-label')
        # plotter.update()
        # plotter.render()

        for _ in range(nr_frames_per_reco):
            plotter.write_frame()  # Write this frame

        # plotter.show(screenshot='image.png')
        plotter.clear()

    # Be sure to close the plotter when finished
    plotter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reconstruction")

    parser.add_argument(
        "--recodir",
        type=str,
        help="directory to store reconstructions",
        default="./"
    )
    parser.add_argument("--filename", type=str)
    parser.add_argument(
        "--time",
        type=int,
        nargs=2,
        help="Timeframes",
        default=None,
    )
    parser.add_argument("--framerate", type=int, default=24)
    parser.add_argument("--frames-per-reco", type=int, default=6)

    args = parser.parse_args()
    movie(args.recodir,
          args.framerate,
          args.frames_per_reco,
          range(args.time[0], args.time[1]),
          out=args.filename)