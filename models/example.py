import numpy as np
from PIL import Image
import torch
import urllib
import matplotlib.pyplot as plt



# where MODEL_NAME can be one of:
# - dinov3_vits16
# - dinov3_vits16plus
# - dinov3_vitb16
# - dinov3_vitl16
# - dinov3_vith16plus
# - dinov3_vit7b16
# - dinov3_convnext_tiny
# - dinov3_convnext_small
# - dinov3_convnext_base
# - dinov3_convnext_large

# For instance
dinov3_vits16 = torch.hub.load(
    repo_or_dir='facebookresearch/dinov3',
    model='dinov3_convnext_tiny',
    weights='./dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
)

image_left_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/dense_sparse_matching/image_left.jpg"
mask_left_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/dense_sparse_matching/image_left_fg.png"
image_right_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/dense_sparse_matching/image_right.jpg"
mask_right_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/dense_sparse_matching/image_right_fg.png"

def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f)


image_left = load_image_from_url(image_left_uri)
mask_left = load_image_from_url(mask_left_uri)

image_right = load_image_from_url(image_right_uri)
mask_right = load_image_from_url(mask_right_uri)

## Plot images
plt.figure(figsize=(16, 8), dpi=300)

for j, (image, mask) in enumerate([(image_left, mask_left), (image_right, mask_right)]):
    foreground = Image.composite(image, mask, mask)
    mask_bg_np = np.copy(np.array(mask))
    mask_bg_np[:, :, 3] = 255 - mask_bg_np[:, :, 3]
    mask_bg = Image.fromarray(mask_bg_np)
    background = Image.composite(image, mask_bg, mask_bg)

    data_to_show = [image, mask, foreground, background]
    data_labels = ["Image", "Mask", "Foreground", "Background"]


    for i in range(len(data_to_show)):
        plt.subplot(2, len(data_to_show), 4 * j + i + 1)
        plt.imshow(data_to_show[i])
        plt.axis('off')
        plt.title(data_labels[i], fontsize=12)
        
plt.show()

print(dinov3_vits16)
