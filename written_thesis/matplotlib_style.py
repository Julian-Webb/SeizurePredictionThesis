import matplotlib as mpl

from written_thesis.helpers import THESIS_FIGS_DIR

# External parameters --------------------------------------------------------------------------------------------------
TEXT_WIDTH_PT = 418.25555  # from LaTeX: ``\typeout{TEXTWIDTH=\the\textwidth}``
PT_PER_INCH = 72.27

FONT_SIZE = 11
SMALL_FONT_SIZE = FONT_SIZE - 3
# ----------------------------------------------------------------------------------------------------------------------

TEXT_WIDTH_INCHES = TEXT_WIDTH_PT / PT_PER_INCH


def latex_figsize(width_fraction=1.0, height_ratio=0.62):
    width_inch = TEXT_WIDTH_INCHES * width_fraction
    height_inch = width_inch * height_ratio
    return width_inch, height_inch


def apply_style(use_small_font: bool=False):
    mpl.rcParams['figure.figsize'] = latex_figsize()
    fz = SMALL_FONT_SIZE if use_small_font else FONT_SIZE
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
        'font.size': fz,
        "axes.labelsize": fz,
        "axes.titlesize": fz,
        "legend.fontsize": fz - 1,
        "xtick.labelsize": fz - 1,
        "ytick.labelsize": fz - 1,
        "savefig.bbox": None,
        "figure.dpi": 300,
    })


if __name__ == '__main__':
    print(f'{TEXT_WIDTH_INCHES=:.2f}')

    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = mpl.pyplot.subplots()
    ax.plot([1, 2, 3])
    ax.set_xlabel('x-label')
    ax.set_ylabel('y-label')
    ax.set_title('Axis title')
    fig.suptitle('Figure title')
    fig.savefig(
        THESIS_FIGS_DIR / 'latex_style_test.pdf',
    )
