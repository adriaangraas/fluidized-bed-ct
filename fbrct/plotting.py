import matplotlib
import matplotlib.pyplot as plt

CM = 1/2.54
DPI = 500
FONT = 'Arial'
FONT_SIZE = 7
TEXTWIDTH = 18.35  # elsevier 5p
COLUMNWIDTH = 9.0

font = {'family': FONT,
        'sans-serif': FONT,
        'size': FONT_SIZE}
matplotlib.rc('font', **font)

plt.rcParams.update({
        'figure.raise_window': False,
        'figure.dpi': DPI,
})

plt.rcParams.update({
        'ps.fonttype': 42,
        'pdf.fonttype': 42,
        'image.interpolation': 'none'
})