import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np


def plot_projlines(projs, colors, labels, pixel_width, pause=5.0):
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    plt.figure()
    plt.title("Noise profile on Detector 1")

    def x_fmt(x, y):
        return "{:.0f}".format(x * pixel_width)

    s = projs[0].shape[-1] // 2
    ss = slice(projs[0].shape[-2] // 2 - 200, projs[0].shape[-2] // 2 + 200)

    for i, (line, color, label) in enumerate(zip(projs, colors, labels)):
        plt.plot(line[..., ss, s], color=color, label=label)
        # ax.set_title(f"Camera {i+1}")
        # im = ax.imshow(p[pixel_start:pixel_end],
        #                vmin=0.,
        #                vmax=np.max(projs[:, pixel_start:pixel_end]))
        #
        # # ax.invert_yaxis()

    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0 / pixel_width))  # cm
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0 / pixel_width / 10))  # cm

    ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
    # ax.set_aspect('equal')
    ax.set(xlabel="Width (cm)", ylabel="Preprocessed measurement")
    ax.label_outer()
    plt.tight_layout()
    plt.legend()
    plt.pause(pause)


def plot_projs(
    projs: np.ndarray,
    pixel_width: float = None,
    pixel_height: float = None,
    row_start: int = None,
    row_end: int = None,
    pause=1.0,
):
    import matplotlib.pyplot as plt

    assert projs.ndim == 3

    if row_start is None:
        row_start = 0
    if row_end is None:
        row_end = projs.shape[1]

    plt.style.use("dark_background")
    subplt_shape = (
        (len(projs),) if projs.shape[2] > projs.shape[1] else (1, (len(projs)))
    )
    fig, axs = plt.subplots(*subplt_shape)

    def x_fmt(x, y):
        return "{:.0f}".format(x * pixel_width)

    def y_fmt(x, y):
        return "{:.0f}".format(x * pixel_height + row_start * pixel_height)

    for i, (ax, p) in enumerate(zip(axs, projs)):
        ax.set_title(f"Camera {i+1}")
        im = ax.imshow(
            p[row_start:row_end],
            vmin=0.0,
            vmax=np.max(projs[:, row_start:row_end]),
        )

        if pixel_width is not None:
            ax.xaxis.set_major_locator(plt.MultipleLocator(1.0 / pixel_width))  # cm
            ax.xaxis.set_minor_locator(
                plt.MultipleLocator(1.0 / pixel_width / 10)
            )  # cm
            ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
        if pixel_height is not None:
            ax.yaxis.set_major_locator(plt.MultipleLocator(1.0 / pixel_height))  # cm
            ax.yaxis.set_minor_locator(
                plt.MultipleLocator(1.0 / pixel_height / 10)
            )  # cm
            ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

        # ax.invert_yaxis()

    plt.gca().set_aspect("equal")

    for ax in axs.flat:
        ax.set(xlabel="Width (cm)", ylabel="Height (cm)")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.85, 0.12, 0.01, 0.80])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout()
    plt.pause(pause)


def plot_nicely(
    title,
    im,
    vmin,
    vmax,
    vox_sz,
    cmap,
    geoms=None,
    with_lines=True,
    mask=None,
    mask_cmap=None,
    ylabel="y",
):
    import matplotlib.ticker as tick

    extent = (
        -0.5 - (im.shape[0] // 2),
        (im.shape[0] // 2) - 0.5,
        -0.5 - (im.shape[1] // 2),
        (im.shape[1] // 2) - 0.5,
    )

    plt.figure()
    plt.cla()
    plt.title(title)
    im = np.flipud(np.swapaxes(im, 0, 1))
    print(im.shape)
    plt.imshow(
        im,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        origin="lower",
        extent=extent,
        interpolation="none",
    )
    ax = plt.gca()
    ax.set_xlim(xmin=-im.shape[1] // 2, xmax=im.shape[1] // 2)
    ax.set_ylim(ymin=-im.shape[0] // 2, ymax=im.shape[0] // 2)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0 / vox_sz[1]))  # cm
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0 / vox_sz[1] / 10))  # mm
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0 / vox_sz[0]))  # cm
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1.0 / vox_sz[0] / 10))  # mm

    ax.set_xlabel("$x$ (cm)")
    ax.set_ylabel(f"${ylabel}$ (cm)")

    def x_fmt(x, y):
        return "{:.1f}".format(x * vox_sz[1])  # millis

    def y_fmt(x, y):
        return "{:.1f}".format(x * vox_sz[0])  # millis

    ax.xaxis.set_major_formatter(tick.FuncFormatter(x_fmt))
    ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

    plt.imshow(
        mask,
        alpha=1,
        cmap=mask_cmap,
        origin="lower",
        extent=extent,
        interpolation="none",
    )

    if with_lines:
        for i, (g, ls) in enumerate(zip(geoms, ("--", "--", "--"))):
            # if i == 1:
            if True:
                S = g.tube_position[:2]
                D = g.detector_position[:2]

                ax.axline(
                    S / vox_sz[:2],
                    D / vox_sz[:2],
                    color="white",
                    linewidth=1,
                    ls=ls,
                    label=f"Source {i}",
                    # alpha=0.5
                )

            if False:
                SD = D - S
                SC = s_center - S  # source-sphere vector

                # unit vector from sphere to source-det line
                CZ = np.inner(SD, SC) / np.inner(SD, SD) * SD - SC
                nCZ = CZ / np.linalg.norm(CZ)

                # outer point on the ball with radius r, being hit by SD
                P = s_center + s_rad * np.array(vox_sz[:2]) * nCZ

                ax.axline(
                    S / vox_sz[:2],
                    P / vox_sz[:2],
                    color="white",
                    linewidth=1.0,
                    ls=ls,
                    label=f"p1",
                )

                P = s_center - s_rad * np.array(vox_sz[:2]) * nCZ
                ax.axline(
                    S / vox_sz[:2],
                    P / vox_sz[:2],
                    color="white",
                    linewidth=1.0,
                    ls=ls,
                    label=f"p2",
                )

        # plt.legend()
    plt.tight_layout()
