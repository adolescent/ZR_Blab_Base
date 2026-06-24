"""Shared path helpers for 50D object-space analysis (Bao et al. style)."""

BRAIN_AREAS = ('MSB', 'ML', 'ASB', 'AL')

# Shared caches at savepath root (stimulus / object space; area-independent)
SHARED_FILES = {
    'step1': 'nsd1k_obj_space_step1.npz',
    'step2': 'metamer1k_obj_space_step2.npz',
    'shuffle_axis': 'shuffle_axis.npz',
}

# Per-area files under savepath/{area}/
AREA_FILES = {
    'obj_axis_fit': 'obj_axis_fit.npz',
    'obj_axis_summary': 'obj_axis_summary.csv',
    'shuffle_neuron': 'shuffle_neuron.npz',
    'shuffle_neuron_summary': 'shuffle_neuron_summary.csv',
    'mediation': 'mediation.npz',
}


def area_dir(savepath, area):
    import OS_Tools as ot
    return ot.Join(savepath, area)


def shared_path(savepath, key):
    import OS_Tools as ot
    return ot.Join(savepath, SHARED_FILES[key])


def area_path(savepath, area, key):
    import OS_Tools as ot
    return ot.Join(area_dir(savepath, area), AREA_FILES[key])


def rsp_path(cell_rootpath, area):
    import OS_Tools as ot
    return ot.Join(ot.Join(cell_rootpath, area), 'avr_rsp.npy')


# Legacy flat layout (pre subfolder refactor): savepath/{area}_obj_axis_fit.npz
_LEGACY_AREA_FILES = {
    'obj_axis_fit': '{area}_obj_axis_fit.npz',
    'obj_axis_summary': '{area}_obj_axis_summary.csv',
    'shuffle_neuron': '{area}_shuffle_neuron.npz',
    'shuffle_neuron_summary': '{area}_shuffle_neuron_summary.csv',
}


def resolve_area_path(savepath, area, key):
    """Prefer savepath/{area}/; fall back to legacy flat filenames for reading."""
    import os
    import OS_Tools as ot
    new_p = area_path(savepath, area, key)
    if os.path.isfile(new_p):
        return new_p
    legacy = ot.Join(savepath, _LEGACY_AREA_FILES[key].format(area=area))
    if os.path.isfile(legacy):
        return legacy
    return new_p
