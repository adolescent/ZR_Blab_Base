'''
Generate unpooled AlexNet conv5 activations (before the final maxpool).
Output shape: (N_img, 256, 13, 13), e.g. (1000, 256, 13, 13) for 1000 images.
'''

#%%
import OS_Tools as ot
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from torchvision.models.feature_extraction import create_feature_extractor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

all_filename = ot.Get_File_Name(r'E:\#Stimsets\NSD1000','.bmp')
savepath = r'E:\#Preprocessed_Data\Selected_Cells'
all_filename.sort()
all_metamer_filepath = all_filename

#%%
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


dataset = ImageDataset(all_metamer_filepath)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
alexnet.eval()

# features.11 = ReLU after last 256 conv; spatial 13x13. features.12 = maxpool -> 6x6.
return_nodes = {'features.11': 'conv5_unpooled'}
fetcher = create_feature_extractor(alexnet, return_nodes=return_nodes)

all_conv5 = []

print("Extracting AlexNet conv5 (unpooled)...")
with torch.no_grad():
    for batch_data in loader:
        images = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
        images = images.to(device)
        features = fetcher(images)
        all_conv5.append(features['conv5_unpooled'].cpu().numpy())

matrix_conv5 = np.concatenate(all_conv5, axis=0)
assert matrix_conv5.shape == (1000, 256, 13, 13), matrix_conv5.shape

save_filename = ot.Join(savepath, "Alex_conv5_unpooled.npz")
np.savez_compressed(save_filename, conv5_unpooled=matrix_conv5)

print(f"Saved: {save_filename}")
print(f"conv5_unpooled shape: {matrix_conv5.shape}")
#%%
np.savez_compressed(ot.Join(savepath,'Alex_Response_conv5_unpooled_nsd'), conv5_unpooled=matrix_conv5)
