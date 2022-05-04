import argparse
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import os

from settings import *


def veloc(scan, recodir, subdir, save=True, mode_norm=True):
    fname = f'{recodir}/{scan.name}/{subdir}/{scan.name}_coords.npy'
    if not os.path.exists(fname):
        print(f"{fname} not found.")
        return

    print(f'Continuing with {fname}')
    coords = np.load(fname, allow_pickle=True).item()  # type: dict
    c0 = coords[list(coords.keys())[0]]
    coords = {i: c - c0 for i, c in coords.items()}

    interesting_coords = np.array(
        [coords[t] for t in scan.phantoms[0].interesting_time])

    if mode_norm:
        z_distance = [np.linalg.norm(interesting_coords[i])
                      for i in range(len(interesting_coords))]
    else:
        z_distance = interesting_coords[:, 2]

    x = np.array(range(len(interesting_coords)))
    A_x = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A_x, z_distance, rcond=None)[0]

    fig = plt.figure()
    ax = plt.gca()
    # plt.title(f"{scan.name} {scan.projs_dir}\n"
    #           f"Estimated w/ framerate: {np.round(m * scan.framerate, 2)}")

    # ax.set_xlim(xmin=-shp[0] // 2, xmax=shp[0] // 2)
    # ax.set_ylim(ymin=-shp[1] // 2, ymax=shp[1] // 2)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # cm
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))  # mm

    def y_fmt(x, y):
        return "{:.1f}".format(x)  # millis

    # ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

    # measurements
    if mode_norm:
        plt.plot([t / scan.framerate for t in coords.keys()],
                 [np.linalg.norm(c / 10) for c in coords.values()],
                 '.', markersize=3, label="$\|\|p_t - p_0\|\|_2$")
    else:
        plt.plot([t / scan.framerate for t in coords.keys()],
                 [c[2] / 10 for c in coords.values()],
                 '.', markersize=3, label="$|p_t \hat{e}_3|$")

    # interpolation
    plt.plot((x + scan.phantoms[0].interesting_time.start) / scan.framerate,
             (m * x + c) / 10,
             # 'r', label=f'Fit y={np.round(m, 2)}x+c',
             'r', label=f'{np.round(m * scan.framerate, 0)} mm/s',
             linewidth=1)

    # # expected speed
    plt.plot((x + scan.phantoms[0].interesting_time.start) / scan.framerate,
             (scan.expected_velocity / scan.framerate * x + c) / 10,
             'g-', label=f"{scan.expected_velocity} mm/s",
             linewidth=1)

    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Phantom position $z$ (cm)")
    plt.legend()
    plt.show()

    if save:
        fig.savefig(f"{recodir}/Velocity-{scan.name}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Velocity graph")

    parser.add_argument("--name", type=str,
                        help="Name of the experiment to run."
                             " Must be available in settings.py. Runs"
                             " all SCANS from settings.py if not provided.")
    parser.add_argument("--subdir", type=str,
                        help="Directory where the coords file lives.")
    parser.add_argument("--recodir", type=str,
                        help="directory to store reconstructions",
                        default='./')
    parser.add_argument("--vertical", action="store_true",
                        help="only compute velocity on the vertical component",
                        default=False)

    args = parser.parse_args()
    recodir = args.recodir
    subdir = args.subdir
    mode_norm = not args.vertical

    assert len(get_scans(args.name)) == 1
    scan = get_scans(args.name)[0]
    if not isinstance(scan, TraverseScan):
        warnings.warn("Expecting a `TraverseScan`.")
        exit(0)

    veloc(scan, recodir, subdir, True, mode_norm)