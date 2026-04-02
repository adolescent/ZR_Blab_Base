
"""
Batch extract VGG19 responses and layer-wise Gram statistics.

Outputs:
1) last_conv (conv5_4) and fc1 responses for 1000 images.
2) Layer-wise Gram matrix sums in pool1/pool2/pool4 at constrain level 1/2/3/4.
"""

#%%
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import OS_Tools as ot


img_path = r"E:\#Stimsets\Metamer_P4_C4321_Object_STI150_1300"
img_name = ot.Get_File_Name(img_path, ".jpg")[:1000]
savepath = r"E:\#Preprocessed_Data\260305_Report_Data\VGG19_rsps"

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16


def build_conditions_1000() -> pd.DataFrame:
    img_indices = np.tile(np.arange(1, 41), 25)
    shuffle_levels = np.tile(np.repeat(np.arange(5), 40), 5)
    return pd.DataFrame(
        {
            "Trial_ID": np.arange(1000),
            "Img_Index": img_indices,
            "Shuffle_Level": shuffle_levels,
        }
    )


def compute_block_grams(activations: torch.Tensor, n_block: int) -> List[torch.Tensor]:
    """
    Follow FeatureSynthesizer gram implementation:
    gram = block @ block.T / block_pixel_num
    """
    _, channels, height, width = activations.shape
    block_h = height // n_block
    block_w = width // n_block
    grams = []

    for i in range(n_block):
        for j in range(n_block):
            block = activations[
                :, :,
                i * block_h:(i + 1) * block_h,
                j * block_w:(j + 1) * block_w,
            ]
            block = block.reshape(channels, -1)
            gram = torch.mm(block, block.t())
            gram = gram / block.size(1)
            grams.append(gram)
    return grams


def compute_gram_matrix_sum(grams: List[torch.Tensor]) -> float:
    """Return scalar sum of all Gram matrix elements across blocks."""
    gram_sum = 0.0
    for gram in grams:
        gram_sum += torch.sum(gram).item()
    return float(gram_sum)


def gram_loss_mse(cur_grams: List[torch.Tensor], tgt_grams: List[torch.Tensor]) -> float:
    """FeatureSynthesizer _compute_loss style: sum of per-block MSE."""
    if len(cur_grams) != len(tgt_grams):
        raise ValueError("cur_grams and tgt_grams must have the same number of blocks.")
    loss = 0.0
    for cur_gram, tgt_gram in zip(cur_grams, tgt_grams):
        loss += torch.mean((cur_gram - tgt_gram) ** 2).item()
    return float(loss)


def gram_similarity_pearson(cur_grams: List[torch.Tensor], tgt_grams: List[torch.Tensor]) -> float:
    """Pearson similarity on concatenated flattened block Grams."""
    cur_vec = torch.cat([g.reshape(-1) for g in cur_grams]).detach().cpu().numpy()
    tgt_vec = torch.cat([g.reshape(-1) for g in tgt_grams]).detach().cpu().numpy()
    if np.std(cur_vec) < 1e-12 or np.std(tgt_vec) < 1e-12:
        return np.nan
    return float(np.corrcoef(cur_vec, tgt_vec)[0, 1])


def summarize_synth_loss(df_trial: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df_trial.columns if c.startswith("loss_") or c.startswith("sim_")]
    summary_parts = []
    for c_level, sub in df_trial.groupby("Constrain_Level"):
        row = {"Constrain_Level": int(c_level), "N": int(len(sub))}
        for col in metric_cols:
            val = sub[col].dropna().to_numpy()
            if val.size == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_se"] = np.nan
            else:
                row[f"{col}_mean"] = float(np.mean(val))
                row[f"{col}_se"] = float(np.std(val, ddof=1) / np.sqrt(val.size)) if val.size > 1 else 0.0
        summary_parts.append(row)
    return pd.DataFrame(summary_parts).sort_values("Constrain_Level").reset_index(drop=True)


def add_loss_ratio_columns(df_trial: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-layer error proportion within each Img_Index across constrain levels.
    ratio = loss_at_this_constrain / sum(loss_across_c1..c4_for_same_image_and_layer)
    """
    if len(df_trial) == 0:
        return df_trial
    out = df_trial.copy()
    for layer_name in ("pool1", "pool2", "pool4"):
        ratio_col = f"loss_ratio_{layer_name}"
        out[ratio_col] = np.nan
        for img_idx, sub in out.groupby("Img_Index"):
            loss_cols = []
            for c_level in (1, 2, 3, 4):
                c_col = f"loss_{layer_name}_c{c_level}"
                if c_col in sub.columns:
                    loss_cols.append(c_col)
            if len(loss_cols) == 0:
                continue

            # Per-row pick the active loss column according to Constrain_Level
            active_loss = []
            for _, rr in sub.iterrows():
                c_level = int(rr["Constrain_Level"])
                c_col = f"loss_{layer_name}_c{c_level}"
                active_loss.append(rr[c_col] if c_col in sub.columns else np.nan)
            active_loss = np.array(active_loss, dtype=float)

            total_err = np.nansum(active_loss)
            if total_err > 0:
                out.loc[sub.index, ratio_col] = active_loss / total_err
            else:
                out.loc[sub.index, ratio_col] = np.nan
    return out


def extract_vgg_responses_and_losses(image_list: List[str], image_folder: str) -> None:
    os.makedirs(savepath, exist_ok=True)

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device).eval()
    model.requires_grad_(False)

    feature_indices = {
        "pool1": 4,
        "pool2": 9,
        "pool4": 27,
        "conv5_4": 34,
    }

    valid_names = []
    valid_trial_ids = []
    missing_images = []
    last_conv_all = []
    fc1_all = []
    gram_rows = []
    # Store per-trial grams for later raw-vs-synth comparison
    pool_grams_by_trial: Dict[str, Dict[int, Dict[int, List[torch.Tensor]]]] = {
        "pool1": {1: {}, 2: {}, 3: {}, 4: {}},
        "pool2": {1: {}, 2: {}, 3: {}, 4: {}},
        "pool4": {1: {}, 2: {}, 3: {}, 4: {}},
    }

    with torch.no_grad():
        for start_idx in tqdm(range(0, len(image_list), batch_size), desc="VGG19 forward"):
            batch_names = image_list[start_idx:start_idx + batch_size]
            batch_tensors = []
            batch_valid_names = []
            batch_trial_ids = []

            for i_local, c_name in enumerate(batch_names):
                trial_id = start_idx + i_local
                full_path = ot.Join(image_folder, c_name)
                if not os.path.exists(full_path):
                    missing_images.append(c_name)
                    continue
                img = Image.open(full_path).convert("RGB")
                batch_tensors.append(preprocess(img))
                batch_valid_names.append(c_name)
                batch_trial_ids.append(trial_id)

            if not batch_tensors:
                continue

            x = torch.stack(batch_tensors, dim=0).to(device)
            pool_feats: Dict[str, torch.Tensor] = {}
            conv5_4_feat = None

            for idx, layer in enumerate(model.features):
                x = layer(x)
                if idx in feature_indices.values():
                    for k, v in feature_indices.items():
                        if idx == v:
                            if k == "conv5_4":
                                conv5_4_feat = x
                            else:
                                pool_feats[k] = x

            if conv5_4_feat is None:
                raise RuntimeError("conv5_4 feature not captured from VGG19.")

            fc_input = torch.flatten(x, 1)
            fc1 = model.classifier[0](fc_input)
            fc1 = model.classifier[1](fc1)  # ReLU after linear (standard fc1 response)

            conv_np = conv5_4_feat.reshape(conv5_4_feat.shape[0], -1).cpu().numpy().astype(np.float32)
            fc1_np = fc1.cpu().numpy().astype(np.float32)
            last_conv_all.append(conv_np)
            fc1_all.append(fc1_np)
            valid_names.extend(batch_valid_names)
            valid_trial_ids.extend(batch_trial_ids)

            for b_idx, c_name in enumerate(batch_valid_names):
                tid = int(batch_trial_ids[b_idx])
                row = {"Trial_ID": tid, "img_name": c_name}
                for layer_name in ("pool1", "pool2", "pool4"):
                    fmap = pool_feats[layer_name][b_idx:b_idx + 1]
                    grams_l1 = compute_block_grams(fmap, 1)
                    grams_l2 = compute_block_grams(fmap, 2)
                    grams_l3 = compute_block_grams(fmap, 3)
                    grams_l4 = compute_block_grams(fmap, 4)
                    row[f"gram_{layer_name}_l1"] = compute_gram_matrix_sum(grams_l1)
                    row[f"gram_{layer_name}_l2"] = compute_gram_matrix_sum(grams_l2)
                    row[f"gram_{layer_name}_l3"] = compute_gram_matrix_sum(grams_l3)
                    row[f"gram_{layer_name}_l4"] = compute_gram_matrix_sum(grams_l4)
                    pool_grams_by_trial[layer_name][1][tid] = [g.detach().cpu() for g in grams_l1]
                    pool_grams_by_trial[layer_name][2][tid] = [g.detach().cpu() for g in grams_l2]
                    pool_grams_by_trial[layer_name][3][tid] = [g.detach().cpu() for g in grams_l3]
                    pool_grams_by_trial[layer_name][4][tid] = [g.detach().cpu() for g in grams_l4]
                gram_rows.append(row)

    if last_conv_all:
        last_conv_arr = np.concatenate(last_conv_all, axis=0)
        fc1_arr = np.concatenate(fc1_all, axis=0)
    else:
        last_conv_arr = np.empty((0, 0), dtype=np.float32)
        fc1_arr = np.empty((0, 0), dtype=np.float32)

    np.savez(
        ot.Join(savepath, "VGG19_Response.npz"),
        trial_id=np.array(valid_trial_ids, dtype=np.int32),
        img_name=np.array(valid_names),
        last_conv=last_conv_arr,
        fc1=fc1_arr,
    )

    gram_df = pd.DataFrame(gram_rows)
    gram_df.to_parquet(ot.Join(savepath, "VGG19_GramSum.parquet"), index=False)
    # raw vs synth comparison
    df_conditions = build_conditions_1000()
    cond_valid = df_conditions[df_conditions["Trial_ID"].isin(valid_trial_ids)].copy()
    # shuffle-to-constrain mapping confirmed by user:
    # raw=0; shuffle1..4 corresponds to constrain4..1
    shuffle_to_constrain = {1: 4, 2: 3, 3: 2, 4: 1}
    target_constrains = {1, 2, 3, 4}
    compare_rows = []

    for img_idx, sub in cond_valid.groupby("Img_Index"):
        raw_trials = sub.loc[sub["Shuffle_Level"] == 0, "Trial_ID"].tolist()
        if len(raw_trials) == 0:
            continue
        raw_tid = int(raw_trials[0])

        for _, rr in sub.iterrows():
            shuf = int(rr["Shuffle_Level"])
            if shuf == 0:
                continue
            c_level = shuffle_to_constrain.get(shuf, None)
            if c_level not in target_constrains:
                continue
            tid = int(rr["Trial_ID"])
            row = {
                "Trial_ID": tid,
                "Img_Index": int(img_idx),
                "Shuffle_Level": shuf,
                "Constrain_Level": int(c_level),
                "Raw_Trial_ID": raw_tid,
            }
            for layer_name in ("pool1", "pool2", "pool4"):
                if tid not in pool_grams_by_trial[layer_name][c_level]:
                    continue
                if raw_tid not in pool_grams_by_trial[layer_name][c_level]:
                    continue
                cur_grams = pool_grams_by_trial[layer_name][c_level][tid]
                raw_grams = pool_grams_by_trial[layer_name][c_level][raw_tid]
                row[f"loss_{layer_name}_c{c_level}"] = gram_loss_mse(cur_grams, raw_grams)
                row[f"sim_{layer_name}_c{c_level}"] = gram_similarity_pearson(cur_grams, raw_grams)
            compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows)
    compare_df = add_loss_ratio_columns(compare_df)
    compare_df.to_parquet(ot.Join(savepath, "VGG19_SynthLoss_vsRaw.parquet"), index=False)
    summary_df = summarize_synth_loss(compare_df) if len(compare_df) > 0 else pd.DataFrame()
    summary_df.to_parquet(ot.Join(savepath, "VGG19_SynthLoss_vsRaw_summary.parquet"), index=False)

    print(f"Device: {device}")
    print(f"Saved response npz: {ot.Join(savepath, 'VGG19_Response.npz')}")
    print(f"Saved gram sum parquet: {ot.Join(savepath, 'VGG19_GramSum.parquet')}")
    print(f"Saved trial-level compare parquet: {ot.Join(savepath, 'VGG19_SynthLoss_vsRaw.parquet')}")
    print(f"Saved summary compare parquet: {ot.Join(savepath, 'VGG19_SynthLoss_vsRaw_summary.parquet')}")
    print(f"Valid images: {len(valid_names)} / {len(image_list)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"last_conv shape: {last_conv_arr.shape}")
    print(f"fc1 shape: {fc1_arr.shape}")
    if len(gram_df) > 0:
        check_cols = ["gram_pool1_l1", "gram_pool2_l1", "gram_pool4_l1"]
        print("Level=1 mean Gram sums:")
        print(gram_df[check_cols].mean())


#%%
if __name__ == "__main__":
    extract_vgg_responses_and_losses(img_name, img_path)
    #%%
    g_sum = pd.read_parquet(ot.Join(savepath, "VGG19_GramSum.parquet"))
    g_loss = pd.read_parquet(ot.Join(savepath, "VGG19_SynthLoss_vsRaw.parquet"))
    g_summary = pd.read_parquet(ot.Join(savepath, "VGG19_SynthLoss_vsRaw_summary.parquet"))




    