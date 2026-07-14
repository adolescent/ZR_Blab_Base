'''
本节目的是通过nsd建立的encoder，并串联metamer方法，使用diffusion model生成具有类似活动pattern的图片，然后评估能引发相似神经活动模式的图片，和原始图片之间存在什么关联。具体步骤如下：


1.使用nsd刺激集建立一个脑区神经元的encoding model，预测一个脑区群体神经元，对每一张图片的响应
2.将目标图片输入这个模型，得到一个群体相应模式
3.使用和metamer类似的方法，计算这张图片激活在神经元层的gram matrix
4.使用随机图片作为输入，采用diffusion model的梯度下降方法进行优化
5.损失函数的目标是让群体神经活动和encoding model预测的原始图片的神经活动在gram matrix中尽可能接近
6.迭代之后，返回优化后的图片，和损失随迭代下降的曲线

请用尽量简单的代码，一节节地运行，为我解决这个问题。


'''




#%% 1. 参数设置
from pathlib import Path

savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Encoder_Metamer'
SAVE_DIR = Path(savepath)
ENCODER_DIR = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Encoder_fc6')

# 手动运行示例时，通常只需要修改 IMAGE_PATH / AREA。
IMAGE_PATH = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Encoder_Metamer\Demo\0003.jpg')
AREA = 'AL'
SCOPE = None  # 5-fold model 可填 'ani' / 'inani' / 'all'；NSD model 保持 None。
DATASET = None  # 自动判断；也可以手动填 'nsd'。

LR = 0.01
NUM_STEPS = 40000
SAVE_EVERY = 2000
SEED = 114514
RESPONSE_WEIGHT = 1
TV_WEIGHT = 0.005
LOSS_MODE = 'cosine'  # 可选：'gram', 'mse', 'cosine', 'pearson', 'gram+mse', 'gram+cosine', 'gram+pearson', 'cosine+mse'
RESPONSE_NORM = 'center+ceiling'  # 可选：'none', 'cell_zscore', 'vector_zscore', 'center', 'unit'；可加 '+ceiling'
BEST_IMAGE_BY = 'total_loss'  # 可选：'total_loss', 'task_loss', 'tv_loss', 'cosine_r', 'pearson_r', 'last'


#%% 2. 导入依赖与工具函数
import csv
import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    torch = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import torchvision.models as models
    import torchvision.transforms as T
except ModuleNotFoundError as exc:
    models = None
    T = None
    TORCHVISION_IMPORT_ERROR = exc
else:
    TORCHVISION_IMPORT_ERROR = None


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) if torch else None
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) if torch else None


def check_dependencies():
    if TORCH_IMPORT_ERROR is not None:
        raise ModuleNotFoundError('Please install torch before running encoder metamer.') from TORCH_IMPORT_ERROR
    if TORCHVISION_IMPORT_ERROR is not None:
        raise ModuleNotFoundError('Please install torchvision before running encoder metamer.') from TORCHVISION_IMPORT_ERROR


def choose_device(device=None):
    check_dependencies()
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_target_image(image_path, device):
    check_dependencies()
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)


def normalize_image(x):
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x - mean) / std


def tensor_to_pil(x):
    x = x.detach().clamp(0, 1).squeeze(0).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


def gram_matrix(response):
    response = response.reshape(1, -1)
    return response.t() @ response / response.numel()


def total_variation_loss(image_01):
    """Penalize nearby pixel differences to reduce high-frequency noise."""
    loss_h = torch.mean(torch.abs(image_01[:, :, 1:, :] - image_01[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(image_01[:, :, :, 1:] - image_01[:, :, :, :-1]))
    return loss_h + loss_w


def vector_zscore(x, eps=1e-6):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + eps)


def cosine_loss(response, target_response, eps=1e-6):
    return 1.0 - F.cosine_similarity(response, target_response, dim=1, eps=eps).mean()


def pearson_loss(response, target_response, eps=1e-6):
    response = response - response.mean(dim=1, keepdim=True)
    target_response = target_response - target_response.mean(dim=1, keepdim=True)
    return cosine_loss(response, target_response, eps=eps)


def compute_response_losses(response, target_response, target_gram, loss_mode):
    loss_mode = loss_mode.lower().replace(' ', '')
    gram_loss = F.mse_loss(gram_matrix(response), target_gram)
    mse_loss = F.mse_loss(response, target_response)
    cos_loss = cosine_loss(response, target_response)
    pear_loss = pearson_loss(response, target_response)

    if loss_mode == 'gram':
        task_loss = gram_loss
    elif loss_mode == 'mse':
        task_loss = mse_loss
    elif loss_mode == 'cosine':
        task_loss = cos_loss
    elif loss_mode == 'pearson':
        task_loss = pear_loss
    elif loss_mode == 'gram+mse':
        task_loss = gram_loss + mse_loss
    elif loss_mode == 'gram+cosine':
        task_loss = gram_loss + cos_loss
    elif loss_mode == 'gram+pearson':
        task_loss = gram_loss + pear_loss
    elif loss_mode in ('cosine+mse', 'mse+cosine'):
        task_loss = cos_loss + mse_loss
    else:
        raise ValueError(
            f'Unknown loss_mode={loss_mode!r}. Use gram, mse, cosine, pearson, '
            'gram+mse, gram+cosine, gram+pearson, or cosine+mse.'
        )

    return task_loss, {
        'gram': gram_loss,
        'mse': mse_loss,
        'cosine': cos_loss,
        'pearson': pear_loss,
    }


def safe_suffix(image_path):
    suffix = Path(image_path).suffix.lower()
    return suffix if suffix in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff') else '.png'


def scalar(x):
    return float(x.detach().cpu())


def is_better_metric(value, best_value, metric_name):
    if best_value is None:
        return True
    if metric_name in ('cosine_r', 'pearson_r'):
        return value > best_value
    return value < best_value


#%% 3. 从 encoder 保存目录自动找到模型和 PCA
def _read_summary_rows(summary_csv):
    with open(summary_csv, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _path_from_summary(value, encoder_dir):
    path = Path(value)
    if path.is_file():
        return path

    fallback = encoder_dir / path.name
    if fallback.is_file():
        return fallback

    matches = list(encoder_dir.rglob(path.name))
    if matches:
        return matches[0]

    return path


def _find_pca_cache(encoder_dir, dataset=None):
    preferred = []
    if dataset == 'nsd':
        preferred.append('alexnet_nsd1k_global_fc6_pca.npz')
    elif dataset in ('5fold', 'metamer'):
        preferred.append('alexnet_metamer1k_global_fc6_pca.npz')

    preferred.extend([
        'alexnet_nsd1k_global_fc6_pca.npz',
        'alexnet_metamer1k_global_fc6_pca.npz',
    ])

    for name in preferred:
        path = encoder_dir / name
        if path.is_file():
            return path

    matches = sorted(encoder_dir.rglob('alexnet*_global_fc6_pca.npz'))
    if not matches:
        raise FileNotFoundError(
            f'No PCA cache found under {encoder_dir}. '
            'Expected alexnet_nsd1k_global_fc6_pca.npz or alexnet_metamer1k_global_fc6_pca.npz.'
        )
    return matches[0]


def resolve_encoder_files(encoder_dir, area='AL', scope=None, dataset=None):
    """
    只读 encoder 保存目录，自动返回 model_npz_path 和 pca_cache_path。

    支持两种 summary：
    1. encoding_model_nsd_summary.csv
    2. encoding_model_5fold_fc6_summary.csv
    """
    encoder_dir = Path(encoder_dir)
    if not encoder_dir.is_dir():
        raise NotADirectoryError(f'Encoder dir not found: {encoder_dir}')

    summary_candidates = []
    if dataset == 'nsd':
        summary_candidates.append(('nsd', encoder_dir / 'encoding_model_nsd_summary.csv'))
    elif dataset in ('5fold', 'metamer'):
        summary_candidates.append(('5fold', encoder_dir / 'encoding_model_5fold_fc6_summary.csv'))
    else:
        summary_candidates.extend([
            ('nsd', encoder_dir / 'encoding_model_nsd_summary.csv'),
            ('5fold', encoder_dir / 'encoding_model_5fold_fc6_summary.csv'),
        ])

    checked = []
    for summary_dataset, summary_csv in summary_candidates:
        checked.append(str(summary_csv))
        if not summary_csv.is_file():
            continue

        rows = _read_summary_rows(summary_csv)
        for row in rows:
            if area is not None and row.get('area') != area:
                continue
            if scope is not None and row.get('scope') != scope:
                continue

            model_npz_path = _path_from_summary(row['npz_path'], encoder_dir)
            if not model_npz_path.is_file():
                raise FileNotFoundError(f'Model listed in summary was not found: {model_npz_path}')

            pca_cache_path = _find_pca_cache(encoder_dir, dataset=summary_dataset)
            return {
                'model_npz_path': model_npz_path,
                'pca_cache_path': pca_cache_path,
                'summary_csv': summary_csv,
                'area': row.get('area', area),
                'scope': row.get('scope', scope),
                'dataset': row.get('dataset', summary_dataset),
                'model_n_pc': int(row['model_n_pc']) if row.get('model_n_pc') else None,
            }

    raise FileNotFoundError(
        f'No matching encoder model found. encoder_dir={encoder_dir}, area={area}, '
        f'scope={scope}, dataset={dataset}. Checked: {checked}'
    )


#%% 4. 加载 AlexNet + PCA + Encoder
class EncoderMetamerSynthesizer:
    def __init__(self, model_npz_path, pca_cache_path, device=None):
        self.device = choose_device(device)
        self.model_npz_path = Path(model_npz_path)
        self.pca_cache_path = Path(pca_cache_path)

        self.alexnet = self._load_alexnet()
        self._load_encoder()

    def _load_alexnet(self):
        try:
            weights = models.AlexNet_Weights.IMAGENET1K_V1
            model = models.alexnet(weights=weights)
        except AttributeError:
            model = models.alexnet(pretrained=True)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _load_encoder(self):
        model_npz = np.load(self.model_npz_path, allow_pickle=True)
        pca_npz = np.load(self.pca_cache_path, allow_pickle=True)

        for key in ('weights', 'bias', 'x_mean', 'x_std', 'model_n_pc'):
            if key not in model_npz.files:
                raise KeyError(f'{self.model_npz_path} lacks {key}')
        for key in ('pca_mean', 'pca_components'):
            if key not in pca_npz.files:
                raise KeyError(f'{self.pca_cache_path} lacks {key}')

        self.model_n_pc = int(model_npz['model_n_pc'])
        self.pca_mean = torch.tensor(pca_npz['pca_mean'], dtype=torch.float32, device=self.device)
        self.pca_components = torch.tensor(
            pca_npz['pca_components'][:self.model_n_pc],
            dtype=torch.float32,
            device=self.device,
        )
        self.x_mean = torch.tensor(model_npz['x_mean'], dtype=torch.float32, device=self.device)
        self.x_std = torch.tensor(model_npz['x_std'], dtype=torch.float32, device=self.device)
        self.weights = torch.tensor(model_npz['weights'], dtype=torch.float32, device=self.device)
        self.bias = torch.tensor(model_npz['bias'], dtype=torch.float32, device=self.device)

        if 'y_fit' in model_npz.files:
            y_fit = np.asarray(model_npz['y_fit'], dtype=np.float32)
            self.response_mean = torch.tensor(y_fit.mean(axis=1, keepdims=True).T, device=self.device)
            response_std = y_fit.std(axis=1, keepdims=True).T
            response_std[response_std < 1e-6] = 1.0
            self.response_std = torch.tensor(response_std, dtype=torch.float32, device=self.device)
        else:
            self.response_mean = self.bias.view(1, -1)
            self.response_std = torch.ones_like(self.response_mean)

        if 'ceiling_index' in model_npz.files:
            ceiling = np.asarray(model_npz['ceiling_index'], dtype=np.float32).reshape(1, -1)
            valid = np.isfinite(ceiling) & (ceiling > 0)
            if np.any(valid):
                ceiling = np.where(valid, ceiling, 0.0)
                ceiling = ceiling / np.mean(ceiling[valid])
            else:
                ceiling = np.ones_like(ceiling, dtype=np.float32)
        else:
            ceiling = np.ones((1, self.bias.numel()), dtype=np.float32)
        self.response_ceiling_weight = torch.tensor(ceiling, dtype=torch.float32, device=self.device)
        self.response_ceiling_sqrt_weight = torch.sqrt(self.response_ceiling_weight.clamp_min(0.0))

    def extract_fc6(self, image_01):
        fc6_buf = []

        def hook(_module, _inp, out):
            # classifier[2] is an inplace ReLU, so clone the linear fc6 output here.
            fc6_buf.append(out.clone())

        handle = self.alexnet.classifier[1].register_forward_hook(hook)
        self.alexnet(normalize_image(image_01))
        handle.remove()
        return fc6_buf[0]

    def predict_response(self, image_01):
        fc6 = self.extract_fc6(image_01)
        pc = (fc6 - self.pca_mean) @ self.pca_components.t()
        pc_z = (pc - self.x_mean) / self.x_std
        return pc_z @ self.weights + self.bias

    def normalize_response(self, response, response_norm='none'):
        response_norm = 'none' if response_norm is None else response_norm.lower().replace(' ', '')
        parts = response_norm.split('+')
        use_ceiling = ('ceiling' in parts) or ('noise_ceiling' in parts)
        base_parts = [p for p in parts if p not in ('ceiling', 'noise_ceiling')]
        base_norm = '+'.join(base_parts) if base_parts else 'none'

        if base_norm == 'none':
            out = response
        elif base_norm == 'cell_zscore':
            out = (response - self.response_mean) / self.response_std
        elif base_norm == 'vector_zscore':
            out = vector_zscore(response)
        elif base_norm == 'center':
            out = response - response.mean(dim=1, keepdim=True)
        elif base_norm == 'unit':
            out = F.normalize(response, p=2, dim=1)
        else:
            raise ValueError(
                f'Unknown response_norm={response_norm!r}. Use none, cell_zscore, '
                'vector_zscore, center, unit, and optionally add +ceiling.'
            )

        if use_ceiling:
            out = out * self.response_ceiling_sqrt_weight
        return out


#%% 5. 计算目标图片的神经 Gram matrix
def target_response_and_gram(synthesizer, image_path, response_norm='none'):
    target = load_target_image(image_path, synthesizer.device)
    with torch.no_grad():
        target_response_raw = synthesizer.predict_response(target)
        target_response = synthesizer.normalize_response(target_response_raw, response_norm)
        target_gram = gram_matrix(target_response)
    return target, target_response.detach(), target_gram.detach()


#%% 6. 从随机 RGB 图像优化 metamer
def optimize_encoder_metamer(
    synthesizer,
    target_response,
    target_gram,
    image_path=None,
    save_dir=None,
    lr=0.03,
    num_steps=5000,
    save_every=1000,
    seed=0,
    response_weight=1.0,
    tv_weight=1e-5,
    loss_mode='gram',
    response_norm='none',
    best_image_by='total_loss',
):
    torch.manual_seed(seed)
    if synthesizer.device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    raw_image = torch.randn(1, 3, 224, 224, device=synthesizer.device, requires_grad=True)
    optimizer = torch.optim.Adam([raw_image], lr=lr)

    loss_history = []
    metrics_history = []
    image_history = []
    live_save_dir = Path(save_dir) if save_dir is not None else None
    if live_save_dir is not None:
        live_save_dir.mkdir(parents=True, exist_ok=True)
    live_stem = Path(image_path).stem if image_path is not None else 'encoder_metamer'
    live_suffix = safe_suffix(image_path) if image_path is not None else '.png'
    best_image = None
    best_image_path = None
    best_step = None
    best_value = None
    best_image_by = best_image_by.lower()
    start_time = time.time()

    for step in tqdm(range(num_steps + 1), desc='Encoder metamer'):
        image_01 = torch.sigmoid(raw_image)
        response_raw = synthesizer.predict_response(image_01)
        response = synthesizer.normalize_response(response_raw, response_norm)
        task_loss, loss_parts = compute_response_losses(response, target_response, target_gram, loss_mode)
        tv_loss = total_variation_loss(image_01)
        loss = response_weight * task_loss + tv_weight * tv_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        row = {
            'step': step,
            'total_loss': scalar(loss),
            'task_loss': scalar(task_loss),
            'gram_loss': scalar(loss_parts['gram']),
            'mse_loss': scalar(loss_parts['mse']),
            'cosine_loss': scalar(loss_parts['cosine']),
            'pearson_loss': scalar(loss_parts['pearson']),
            'cosine_r': 1.0 - scalar(loss_parts['cosine']),
            'pearson_r': 1.0 - scalar(loss_parts['pearson']),
            'tv_loss': scalar(tv_loss),
            'lr': optimizer.param_groups[0]['lr'],
        }
        loss_history.append(row['total_loss'])
        metrics_history.append(row)

        if step % save_every == 0 or step == num_steps:
            current_img = tensor_to_pil(torch.sigmoid(raw_image))
            if live_save_dir is None:
                image_history.append((step, current_img))
            else:
                out = live_save_dir / f'{live_stem}_encoder_metamer_iter{step:05d}{live_suffix}'
                current_img.save(out)
                image_history.append((step, str(out)))

            if best_image_by == 'last':
                current_metric = row['step']
            elif best_image_by in row:
                current_metric = row[best_image_by]
            else:
                raise ValueError(
                    f'Unknown best_image_by={best_image_by!r}. Use last or one of {list(row.keys())}.'
                )

            if best_image_by == 'last' or is_better_metric(current_metric, best_value, best_image_by):
                best_value = current_metric
                best_step = step
                if live_save_dir is None:
                    best_image = current_img.copy()
                    best_image_path = None
                else:
                    best_image = None
                    best_image_path = str(out)
            print(
                f'Step {step:05d}/{num_steps}, '
                f'loss={loss.item():.6g}, task={task_loss.item():.6g}, '
                f'gram={loss_parts["gram"].item():.6g}, mse={loss_parts["mse"].item():.6g}, '
                f'cos={1 - loss_parts["cosine"].item():.4f}, '
                f'pearson={1 - loss_parts["pearson"].item():.4f}, tv={tv_loss.item():.6g}'
            )

    print(f'Finished in {time.time() - start_time:.1f} s')
    if best_image_by == 'last':
        final_image = tensor_to_pil(torch.sigmoid(raw_image))
    elif best_image is not None:
        final_image = best_image
    elif best_image_path is not None:
        final_image = Image.open(best_image_path).convert('RGB')
    else:
        final_image = tensor_to_pil(torch.sigmoid(raw_image))

    selection = {
        'best_image_by': best_image_by,
        'best_step': best_step if best_step is not None else num_steps,
        'best_value': best_value,
        'best_image_path': best_image_path,
    }
    return final_image, loss_history, metrics_history, image_history, selection


#%% 7. 保存 loss 曲线、中间图片和最终图片
def save_encoder_metamer_outputs(
    image_path,
    save_dir,
    final_image,
    loss_history,
    metrics_history,
    image_history,
    selection,
    config,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    suffix = safe_suffix(image_path)

    history_paths = []
    for step, img in image_history:
        if isinstance(img, (str, Path)):
            history_paths.append(str(img))
        else:
            out = save_dir / f'{stem}_encoder_metamer_iter{step:05d}{suffix}'
            img.save(out)
            history_paths.append(str(out))

    final_image_path = save_dir / f'{stem}_encoder_metamer_final{suffix}'
    selected_path = selection.get('best_image_path') if selection else None
    if selected_path is not None and Path(selected_path).is_file():
        shutil.copy2(selected_path, final_image_path)
    else:
        final_image.save(final_image_path)

    loss_history = np.asarray(loss_history, dtype=np.float32)
    loss_history_path = save_dir / f'{stem}_encoder_metamer_loss_history.npy'
    np.save(loss_history_path, loss_history)

    metrics_csv_path = save_dir / f'{stem}_encoder_metamer_metrics.csv'
    if metrics_history:
        with open(metrics_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_history[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_history)

    metrics_npz_path = save_dir / f'{stem}_encoder_metamer_metrics.npz'
    metrics_npz = {
        key: np.asarray([row[key] for row in metrics_history], dtype=np.float32)
        for key in metrics_history[0]
    } if metrics_history else {}
    np.savez(metrics_npz_path, **metrics_npz)

    loss_plot_path = save_dir / f'{stem}_encoder_metamer_loss_curve.png'
    steps = np.asarray(metrics_npz.get('step', np.arange(len(loss_history))), dtype=np.float32)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=150)

    axes[0, 0].plot(steps, metrics_npz.get('total_loss', loss_history), label='total')
    axes[0, 0].plot(steps, metrics_npz.get('task_loss', loss_history), label='task')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(steps, metrics_npz.get('cosine_r', np.zeros_like(steps)), label='cosine r')
    axes[0, 1].plot(steps, metrics_npz.get('pearson_r', np.zeros_like(steps)), label='pearson r')
    axes[0, 1].set_title('Response Similarity')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_ylim(0.999, 1.0)
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(steps, metrics_npz.get('gram_loss', np.zeros_like(steps)), label='gram', alpha=0.75, lw=1.1)
    axes[1, 0].plot(steps, metrics_npz.get('mse_loss', np.zeros_like(steps)), label='mse', alpha=0.55, lw=1.4)
    axes[1, 0].plot(steps, metrics_npz.get('cosine_loss', np.zeros_like(steps)), label='cosine loss', alpha=0.75, lw=1.1)
    axes[1, 0].plot(steps, metrics_npz.get('pearson_loss', np.zeros_like(steps)), label='pearson loss', alpha=0.75, lw=1.1)
    axes[1, 0].set_title('Response Loss Parts')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend(frameon=False, fontsize=7)

    axes[1, 1].plot(steps, metrics_npz.get('tv_loss', np.zeros_like(steps)), label='tv')
    axes[1, 1].set_title('Image Regularization')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('TV loss')
    axes[1, 1].legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(loss_plot_path, bbox_inches='tight')
    plt.close()

    config_path = save_dir / f'{stem}_encoder_metamer_config.json'
    if selection:
        config.update(selection)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    final_loss = float(loss_history[-1])
    with open(save_dir / f'{stem}_encoder_metamer_final_loss.txt', 'w', encoding='utf-8') as f:
        f.write(f'{final_loss:.10g}\n')

    return {
        'final_image_path': str(final_image_path),
        'loss_plot_path': str(loss_plot_path),
        'history_image_paths': history_paths,
        'loss_history_path': str(loss_history_path),
        'metrics_csv_path': str(metrics_csv_path),
        'metrics_npz_path': str(metrics_npz_path),
        'config_path': str(config_path),
        'final_loss': final_loss,
        'best_image_by': selection.get('best_image_by') if selection else None,
        'best_step': selection.get('best_step') if selection else None,
        'best_value': selection.get('best_value') if selection else None,
        'best_image_path': selection.get('best_image_path') if selection else None,
    }


def run_encoder_metamer(
    image_path,
    save_dir,
    model_npz_path,
    pca_cache_path,
    lr=0.03,
    num_steps=5000,
    save_every=1000,
    seed=0,
    response_weight=1.0,
    tv_weight=1e-5,
    loss_mode='gram',
    response_norm='none',
    best_image_by='total_loss',
    device=None,
):
    image_path = Path(image_path)
    save_dir = Path(save_dir)
    model_npz_path = Path(model_npz_path)
    pca_cache_path = Path(pca_cache_path)

    synthesizer = EncoderMetamerSynthesizer(model_npz_path, pca_cache_path, device=device)
    _, target_response, target_gram = target_response_and_gram(
        synthesizer,
        image_path,
        response_norm=response_norm,
    )

    final_image, loss_history, metrics_history, image_history, selection = optimize_encoder_metamer(
        synthesizer=synthesizer,
        target_response=target_response,
        target_gram=target_gram,
        image_path=image_path,
        save_dir=save_dir,
        lr=lr,
        num_steps=num_steps,
        save_every=save_every,
        seed=seed,
        response_weight=response_weight,
        tv_weight=tv_weight,
        loss_mode=loss_mode,
        response_norm=response_norm,
        best_image_by=best_image_by,
    )

    config = {
        'image_path': str(image_path),
        'save_dir': str(save_dir),
        'model_npz_path': str(model_npz_path),
        'pca_cache_path': str(pca_cache_path),
        'lr': lr,
        'num_steps': num_steps,
        'save_every': save_every,
        'seed': seed,
        'response_weight': response_weight,
        'tv_weight': tv_weight,
        'loss_mode': loss_mode,
        'response_norm': response_norm,
        'best_image_by': best_image_by,
        'device': str(synthesizer.device),
        'model_n_pc': synthesizer.model_n_pc,
    }
    return save_encoder_metamer_outputs(
        image_path=image_path,
        save_dir=save_dir,
        final_image=final_image,
        loss_history=loss_history,
        metrics_history=metrics_history,
        image_history=image_history,
        selection=selection,
        config=config,
    )


#%% 8. 只给 encoder 保存目录的简单 API
def run_encoder_metamer_from_encoder_dir(
    image_path,
    save_dir,
    encoder_dir,
    area='AL',
    scope=None,
    dataset=None,
    lr=0.03,
    num_steps=5000,
    save_every=1000,
    seed=0,
    response_weight=1.0,
    tv_weight=1e-5,
    loss_mode='gram',
    response_norm='none',
    best_image_by='total_loss',
    device=None,
):
    files = resolve_encoder_files(
        encoder_dir=encoder_dir,
        area=area,
        scope=scope,
        dataset=dataset,
    )
    result = run_encoder_metamer(
        image_path=image_path,
        save_dir=save_dir,
        model_npz_path=files['model_npz_path'],
        pca_cache_path=files['pca_cache_path'],
        lr=lr,
        num_steps=num_steps,
        save_every=save_every,
        seed=seed,
        response_weight=response_weight,
        tv_weight=tv_weight,
        loss_mode=loss_mode,
        response_norm=response_norm,
        best_image_by=best_image_by,
        device=device,
    )
    result.update({
        'encoder_dir': str(Path(encoder_dir)),
        'model_npz_path': str(files['model_npz_path']),
        'pca_cache_path': str(files['pca_cache_path']),
        'summary_csv': str(files['summary_csv']),
        'area': files['area'],
        'scope': files['scope'],
        'dataset': files['dataset'],
    })
    return result


#%% 9. 一行 API 示例
# # 修改第 1 节的 IMAGE_PATH / ENCODER_DIR / AREA 后，运行本 cell。
# result = run_encoder_metamer_from_encoder_dir(
#     image_path=IMAGE_PATH,
#     save_dir=SAVE_DIR,
#     encoder_dir=ENCODER_DIR,
#     area=AREA,
#     scope=SCOPE,
#     dataset=DATASET,
#     lr=LR,
#     num_steps=NUM_STEPS,
#     save_every=SAVE_EVERY,
#     seed=SEED,
#     response_weight=RESPONSE_WEIGHT,
#     tv_weight=TV_WEIGHT,
#     loss_mode=LOSS_MODE,
#     response_norm=RESPONSE_NORM,
#     best_image_by=BEST_IMAGE_BY,
#     device='cuda',
# )
# print(result)

#%% batch,一次性对所有图片计算最优响应。
'''
此处，对参数进行grid search，具体 参数选择范围包括：
1.随机10个随机数种
2.循环全部40张图片
3.对每个图片，每个数种，选择lr为0.001和0.01两种
4.对0.001，迭代20万次，对0.01迭代六万次
5.对每个迭代，循环选择norm方法cell_zsocore,center,center+ceiling,cell_zsocore+ceiling
6.对每个迭代，循环选择优化目标cosine，mse,cosine+mse
7.对每个迭代，选择RESPONSE_WEIGHT为[0.1,1,5,10,50]五种

'''


#%% 10. Grid search 批量运行
GRID_SAVE_DIR = SAVE_DIR / 'grid_search'
GRID_IMAGE_DIR = IMAGE_PATH.parent
GRID_N_IMAGE = 40
GRID_IMAGE_IDS = ['0003', '0005', '0009', '0010', '0012','0014','0015','0019','0022','0025','0027']  # 例如 ['0003', '0009']；None 表示使用 GRID_N_IMAGE 张排序后的图片。
GRID_AREAS = ['AL', 'ASB']
GRID_RANDOM_SEED = 42
GRID_N_SEED = 4
GRID_LR_STEPS = {
    0.001: 200000,
    0.01: 60000,
}
GRID_RESPONSE_NORMS = [
    'cell_zscore',
    'center',
    'center+ceiling',
]
GRID_LOSS_MODES = [
    'cosine',

]
GRID_RESPONSE_WEIGHTS = [0.1, 1, 10]
GRID_SUMMARY_FIELDS = [
    'status',
    'area',
    'image_path',
    'image_stem',
    'condition',
    'condition_dir',
    'fit_idx',
    'seed',
    'lr',
    'num_steps',
    'save_every',
    'response_norm',
    'loss_mode',
    'response_weight',
    'tv_weight',
    'best_image_by',
    'best_step',
    'best_value',
    'final_loss',
    'final_image_path',
    'metrics_csv_path',
    'config_path',
    'error',
]


def list_grid_images(image_dir=GRID_IMAGE_DIR, n_image=GRID_N_IMAGE, image_ids=GRID_IMAGE_IDS):
    image_dir = Path(image_dir)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    paths = []
    for ext in exts:
        paths.extend(sorted(image_dir.glob(ext)))
    paths = sorted(set(paths))
    if image_ids is not None:
        image_ids = [str(image_id).zfill(4) for image_id in image_ids]
        by_stem = {p.stem: p for p in paths}
        missing = [image_id for image_id in image_ids if image_id not in by_stem]
        if missing:
            raise FileNotFoundError(f'Image ids not found in {image_dir}: {missing}')
        return [by_stem[image_id] for image_id in image_ids]
    if len(paths) < n_image:
        raise FileNotFoundError(
            f'Only found {len(paths)} images in {image_dir}, but GRID_N_IMAGE={n_image}.'
        )
    return paths[:n_image]


def make_grid_seeds(n_seed=GRID_N_SEED, random_seed=GRID_RANDOM_SEED):
    rng = np.random.default_rng(random_seed)
    return [int(x) for x in rng.integers(0, 2**31 - 1, size=n_seed)]


def grid_condition_name(fit_idx, lr, num_steps, response_norm, loss_mode, response_weight):
    norm = response_norm.replace('+', '-')
    loss = loss_mode.replace('+', '-')
    lr_text = str(lr).replace('.', 'p')
    rw_text = str(response_weight).replace('.', 'p')
    return f'fit{fit_idx:03d}_lr{lr_text}_step{num_steps}_norm-{norm}_loss-{loss}_rw{rw_text}'


def append_grid_summary(summary_csv, row):
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.is_file()
    with open(summary_csv, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=GRID_SUMMARY_FIELDS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_encoder_metamer_grid_search(
    image_paths=None,
    image_dir=GRID_IMAGE_DIR,
    save_dir=GRID_SAVE_DIR,
    encoder_dir=ENCODER_DIR,
    areas=None,
    image_ids=GRID_IMAGE_IDS,
    scope=SCOPE,
    dataset=DATASET,
    seeds=None,
    lr_steps=None,
    response_norms=None,
    loss_modes=None,
    response_weights=None,
    tv_weight=TV_WEIGHT,
    save_every=SAVE_EVERY,
    best_image_by=BEST_IMAGE_BY,
    device='cuda',
    skip_existing=True,
):
    """
    对多个脑区、40 张图、多个 fit 编号和多组优化参数做 grid search。
    运行顺序是：fit_idx / 参数组 -> area -> image。
    这样会先用同一组参数跑完所有脑区和图片，再进入下一组参数。

    目录结构：
    save_dir / area / image_stem / condition_name / 单次 run 的所有输出
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = save_dir / 'grid_search_summary.csv'

    if image_paths is None:
        image_paths = list_grid_images(image_dir=image_dir, n_image=GRID_N_IMAGE, image_ids=image_ids)
    else:
        image_paths = [Path(p) for p in image_paths]

    if seeds is None:
        seeds = make_grid_seeds()
    if areas is None:
        areas = GRID_AREAS
    if lr_steps is None:
        lr_steps = GRID_LR_STEPS
    if response_norms is None:
        response_norms = GRID_RESPONSE_NORMS
    if loss_modes is None:
        loss_modes = GRID_LOSS_MODES
    if response_weights is None:
        response_weights = GRID_RESPONSE_WEIGHTS

    total = (
        len(areas)
        * len(image_paths)
        * len(seeds)
        * len(lr_steps)
        * len(response_norms)
        * len(loss_modes)
        * len(response_weights)
    )
    print(f'Grid search total runs: {total}')

    done = 0
    for fit_idx, seed in enumerate(seeds, start=1):
        for lr, num_steps in lr_steps.items():
            for response_norm in response_norms:
                for loss_mode in loss_modes:
                    for response_weight in response_weights:
                        condition = grid_condition_name(
                            fit_idx=fit_idx,
                            lr=lr,
                            num_steps=num_steps,
                            response_norm=response_norm,
                            loss_mode=loss_mode,
                            response_weight=response_weight,
                        )

                        for area in areas:
                            area_dir_out = save_dir / area
                            area_dir_out.mkdir(parents=True, exist_ok=True)

                            for image_path in image_paths:
                                done += 1
                                image_path = Path(image_path)
                                image_dir_out = area_dir_out / image_path.stem
                                image_dir_out.mkdir(parents=True, exist_ok=True)

                                condition_dir = image_dir_out / condition
                                final_path = condition_dir / f'{image_path.stem}_encoder_metamer_final{safe_suffix(image_path)}'
                                config_path = condition_dir / f'{image_path.stem}_encoder_metamer_config.json'

                                if skip_existing and final_path.is_file() and config_path.is_file():
                                    print(f'[{done}/{total}] skip existing: {area} / {image_path.stem} / {condition}')
                                    continue

                                print(f'[{done}/{total}] run: {area} / {image_path.stem} / {condition} (seed={seed})')
                                try:
                                    result = run_encoder_metamer_from_encoder_dir(
                                        image_path=image_path,
                                        save_dir=condition_dir,
                                        encoder_dir=encoder_dir,
                                        area=area,
                                        scope=scope,
                                        dataset=dataset,
                                        lr=lr,
                                        num_steps=num_steps,
                                        save_every=save_every,
                                        seed=seed,
                                        response_weight=response_weight,
                                        tv_weight=tv_weight,
                                        loss_mode=loss_mode,
                                        response_norm=response_norm,
                                        best_image_by=best_image_by,
                                        device=device,
                                    )
                                    row = {
                                        'status': 'done',
                                        'area': area,
                                        'image_path': str(image_path),
                                        'image_stem': image_path.stem,
                                        'condition': condition,
                                        'condition_dir': str(condition_dir),
                                        'fit_idx': fit_idx,
                                        'seed': seed,
                                        'lr': lr,
                                        'num_steps': num_steps,
                                        'save_every': save_every,
                                        'response_norm': response_norm,
                                        'loss_mode': loss_mode,
                                        'response_weight': response_weight,
                                        'tv_weight': tv_weight,
                                        'best_image_by': best_image_by,
                                        'best_step': result.get('best_step'),
                                        'best_value': result.get('best_value'),
                                        'final_loss': result.get('final_loss'),
                                        'final_image_path': result.get('final_image_path'),
                                        'metrics_csv_path': result.get('metrics_csv_path'),
                                        'config_path': result.get('config_path'),
                                        'error': '',
                                    }
                                except Exception as exc:
                                    row = {
                                        'status': 'failed',
                                        'area': area,
                                        'image_path': str(image_path),
                                        'image_stem': image_path.stem,
                                        'condition': condition,
                                        'condition_dir': str(condition_dir),
                                        'fit_idx': fit_idx,
                                        'seed': seed,
                                        'lr': lr,
                                        'num_steps': num_steps,
                                        'save_every': save_every,
                                        'response_norm': response_norm,
                                        'loss_mode': loss_mode,
                                        'response_weight': response_weight,
                                        'tv_weight': tv_weight,
                                        'best_image_by': best_image_by,
                                        'best_step': None,
                                        'best_value': None,
                                        'final_loss': None,
                                        'final_image_path': None,
                                        'metrics_csv_path': None,
                                        'config_path': None,
                                        'error': repr(exc),
                                    }
                                    print(f'FAILED: {area} / {image_path.stem} / {condition}: {exc}')

                                append_grid_summary(summary_csv, row)

    print(f'Grid search summary saved to: {summary_csv}')
    return summary_csv

#%%
# 手动确认路径后再运行：
grid_summary = run_encoder_metamer_grid_search()
print(grid_summary)



