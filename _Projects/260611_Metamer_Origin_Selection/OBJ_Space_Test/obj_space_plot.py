"""Save matplotlib figures under savepath with a consistent folder layout."""

import OS_Tools as ot

SCRIPT_TEST_RSP = 'Test_Obj_Space_Rsp'
SCRIPT_THOUGHT = 'Obj_Space_Thought_Reversed'
SCRIPT_MEDIATION = 'Obj_Space_Shuffle_Intersected'
SCRIPT_METAMER_NSD = 'Obj_Space_Metamer_NSD'

SAVE_FIGURES = True
SHOW_FIGURES = True
FIG_DPI = 150
FIG_FMT = 'png'


def configure(*, save=None, show=None, dpi=None, fmt=None):
    global SAVE_FIGURES, SHOW_FIGURES, FIG_DPI, FIG_FMT
    if save is not None:
        SAVE_FIGURES = save
    if show is not None:
        SHOW_FIGURES = show
    if dpi is not None:
        FIG_DPI = dpi
    if fmt is not None:
        FIG_FMT = fmt


def figure_dir(savepath, script, area=None):
    if area:
        return ot.Join(savepath, area, 'figures', script)
    return ot.Join(savepath, 'figures', script)


def save_figure(fig, savepath, script, name, area=None, dpi=None, fmt=None):
    import matplotlib.pyplot as plt

    dpi = FIG_DPI if dpi is None else dpi
    fmt = FIG_FMT if fmt is None else fmt
    out_dir = figure_dir(savepath, script, area=area)
    ot.Mkdir(out_dir, mute=True)
    path = ot.Join(out_dir, f'{name}.{fmt}')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', format=fmt)
    if not SHOW_FIGURES:
        plt.close(fig)
    return path


def finish_fig(fig, savepath, script, name, area=None, dpi=None, fmt=None):
    import matplotlib.pyplot as plt

    path = None
    if SAVE_FIGURES:
        path = save_figure(fig, savepath, script, name, area=area, dpi=dpi, fmt=fmt)
        print(f'saved figure: {path}')
    if SHOW_FIGURES:
        plt.show() 
    elif not SAVE_FIGURES:
        plt.close(fig)
    return path
