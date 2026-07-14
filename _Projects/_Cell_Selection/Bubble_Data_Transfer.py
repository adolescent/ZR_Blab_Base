"""
Randomly sample neurons from each area and export response/mask MAT files.

Output response order (per row):
1000 metamer + 1600 bubble + 1600 rest = 4200 frames.
"""
#%%
import os
import numpy as np
import pandas as pd
from scipy.io import savemat

import OS_Tools as ot
from Py_Structure.Info_Files.InfoLoader import Load_Info


bubble_path = r"E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_SingleBubble"
savepath = r"E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Bubble_Mats"

AREAS = ("ML", "MSB", "AL", "ASB")
N_SAMPLE_PER_AREA = 100
RANDOM_SEED = 20260709
STIMSET_NAME = "Metamer_Singlebubble_v251107"

# avr_rsp has 4240 frames, and 1000~1039 are skipped transition frames.
SLICE_METAMER = slice(0, 1000)     # 1000
SLICE_BUBBLE = slice(1040, 2640)   # 1600
SLICE_REST = slice(2640, 4240)     # 1600
N_OUT = 4200

# Raw mask indices in original stimset.
RAW_SLICE_BUBBLE = slice(1340, 2940)  # 1600
RAW_SLICE_REST = slice(2940, 4540)    # 1600


def _check_layout():
    if (SLICE_METAMER.stop - SLICE_METAMER.start) != 1000:
        raise ValueError("Metamer slice length must be 1000.")
    if (SLICE_BUBBLE.stop - SLICE_BUBBLE.start) != 1600:
        raise ValueError("Bubble slice length must be 1600.")
    if (SLICE_REST.stop - SLICE_REST.start) != 1600:
        raise ValueError("Rest slice length must be 1600.")
    if SLICE_METAMER.stop != 1000 or SLICE_BUBBLE.start != 1040:
        raise ValueError("Expected skipped transition frames: [1000, 1040).")
    if 1000 + 1600 + 1600 != N_OUT:
        raise ValueError("Output width must be 4200.")
    if (RAW_SLICE_BUBBLE.stop - RAW_SLICE_BUBBLE.start) != 1600:
        raise ValueError("Raw bubble mask slice length must be 1600.")
    if (RAW_SLICE_REST.stop - RAW_SLICE_REST.start) != 1600:
        raise ValueError("Raw rest mask slice length must be 1600.")


def _load_and_sample_area(area, rng):
    area_dir = ot.Join(bubble_path, area)
    rsp_path = ot.Join(area_dir, "avr_rsp.npy")
    info_path = ot.Join(area_dir, "cell_site_info.csv")

    if not os.path.exists(rsp_path):
        raise FileNotFoundError(f"Missing response file: {rsp_path}")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing metadata csv: {info_path}")

    avr_rsp = np.load(rsp_path)
    info = pd.read_csv(info_path)

    if avr_rsp.ndim != 2 or avr_rsp.shape[1] != 4240:
        raise ValueError(f"{area} avr_rsp shape must be (n_cell, 4240), got {avr_rsp.shape}")
    if len(info) != avr_rsp.shape[0]:
        raise ValueError(
            f"{area} metadata and response mismatch: len(info)={len(info)}, n_cell={avr_rsp.shape[0]}"
        )
    if len(info) < N_SAMPLE_PER_AREA:
        raise ValueError(f"{area} has only {len(info)} cells, but need {N_SAMPLE_PER_AREA}.")

    selected_idx = np.sort(rng.choice(len(info), size=N_SAMPLE_PER_AREA, replace=False))
    rsp_sel = np.concatenate(
        [
            avr_rsp[selected_idx, SLICE_METAMER],
            avr_rsp[selected_idx, SLICE_BUBBLE],
            avr_rsp[selected_idx, SLICE_REST],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if rsp_sel.shape != (N_SAMPLE_PER_AREA, N_OUT):
        raise ValueError(f"{area} exported response shape error: {rsp_sel.shape}")

    meta_sel = info.iloc[selected_idx].copy().reset_index(drop=True)
    meta_sel.insert(0, "area", area)
    meta_sel.insert(1, "selected_idx_in_area", selected_idx.astype(np.int32))
    return rsp_sel, meta_sel


def _load_bubble_rest_masks():
    _, masks, _ = Load_Info(setname=STIMSET_NAME, load_mask=True)
    if masks is None:
        raise FileNotFoundError(f"Mask file for {STIMSET_NAME} not found.")

    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"Mask array should be 3D, got shape={masks.shape}")
    if masks.shape[0] < RAW_SLICE_REST.stop:
        raise ValueError(
            f"Mask count too small ({masks.shape[0]}), need at least {RAW_SLICE_REST.stop}."
        )

    out = np.concatenate(
        [masks[RAW_SLICE_BUBBLE], masks[RAW_SLICE_REST]],
        axis=0,
    ).astype(bool, copy=False)
    if out.shape[0] != 3200:
        raise ValueError(f"Exported mask count must be 3200, got {out.shape[0]}.")
    return out


def main():
    _check_layout()
    ot.Mkdir(savepath)
    rng = np.random.default_rng(RANDOM_SEED)

    for area in AREAS:
        rsp_sel, meta_sel = _load_and_sample_area(area, rng)

        rsp_mat_path = ot.Join(savepath, f"bubble_rsp_rand100_{area}.mat")
        savemat(
            rsp_mat_path,
            {
                "rsp": rsp_sel,
                "area": np.array([area], dtype=object),
                "selected_idx_in_area": meta_sel["selected_idx_in_area"].to_numpy(dtype=np.int32),
                "n_metamer": np.int32(1000),
                "n_bubble": np.int32(1600),
                "n_rest": np.int32(1600),
                "skip_start": np.int32(1000),
                "skip_end": np.int32(1040),
            },
            do_compression=True,
        )

        meta_csv_path = ot.Join(savepath, f"bubble_rsp_rand100_{area}_meta.csv")
        meta_sel.to_csv(meta_csv_path, index=False, encoding="utf-8-sig")
        print(f"[{area}] rsp: {rsp_sel.shape} -> {rsp_mat_path}")
        print(f"[{area}] meta -> {meta_csv_path}")

    masks_3200 = _load_bubble_rest_masks()
    mask_mat_path = ot.Join(savepath, "bubble_rest_masks_3200.mat")
    savemat(
        mask_mat_path,
        {
            "masks": masks_3200,
            "n_bubble": np.int32(1600),
            "n_rest": np.int32(1600),
            "raw_bubble_start": np.int32(RAW_SLICE_BUBBLE.start),
            "raw_rest_start": np.int32(RAW_SLICE_REST.start),
        },
        do_compression=True,
    )
    print(f"masks: {masks_3200.shape} -> {mask_mat_path}")
#%%

if __name__ == "__main__":
    main()

#%% test: load mat, z-score per neuron, quick heatmap
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

test_area = "MSB"
mat_path = ot.Join(savepath, f"bubble_rsp_rand100_{test_area}.mat")
rsp = np.asarray(loadmat(mat_path)["rsp"], dtype=np.float32)
rsp_z = (rsp - rsp.mean(1, keepdims=True)) / (rsp.std(1, keepdims=True) + 1e-8)

sns.heatmap(rsp_z, center=0, cmap="bwr", vmin=-2, vmax=2)
plt.axvline(1000, color="k", lw=0.5)   # metamer | bubble
plt.axvline(2600, color="k", lw=0.5)   # bubble | rest
plt.title(f"{test_area} rand100, z-scored per neuron")
plt.show() 