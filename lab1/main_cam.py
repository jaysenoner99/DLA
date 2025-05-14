import cv2
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import requests
import torch.nn.functional as F
from io import BytesIO
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Class Activation Maps for a pretrained ResNet18 model, using Imagenette data"
    )

    parser.add_argument(
        "--class_index",
        "--cls_index",
        "--class_idx",
        "--cls",
        default=0,
        type=int,
        help="""Class of the input image. Choose from tench (0),
            English springer (1), cassette player (2), chainsaw (3), church (4),
            French horn (5), garbage truck (6), gas pump (7), golf ball (8), parachute (9) """,
    )
    parser.add_argument(
        "--sample_index",
        "--sample_idx",
        "--sample",
        default=5,
        type=int,
        help="Index of the sample of class [--class_index] to be used",
    )
    parser.add_argument(
        "--url",
        default="",
        type=str,
        help="URL of the image to be passed as input. Must be an URL of an image in the Imagenette dataset",
    )
    args = parser.parse_args()
    return args


## Setup Imagenette dataset (160px version)
# First download and extract the dataset (uncomment to download)
# !wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
# !tar xzf imagenette2-160.tgz

# Imagenette classes (subset of 10 ImageNet classes)
imagenette_classes = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]
args = parse_args()
imagenet_classes = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
preprocess = transforms.Compose(
    [
        transforms.Resize(160),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


## Load a sample image from Imagenette
def load_imagenette_sample(class_index=0, sample_index=0):
    # Map class indices to actual directory names
    class_dir_map = {
        0: "n01440764",  # tench
        1: "n02102040",  # English springer
        2: "n02979186",  # cassette player
        3: "n03000684",  # chain saw
        4: "n03028079",  # church
        5: "n03394916",  # French horn
        6: "n03417042",  # garbage truck
        7: "n03425413",  # gas pump
        8: "n03445777",  # golf ball
        9: "n03888257",  # parachute
    }

    root_dir = "imagenette2-160/train"
    class_dir = class_dir_map[class_index]

    class_images = sorted(os.listdir(os.path.join(root_dir, class_dir)))

    img_path = os.path.join(root_dir, class_dir, class_images[sample_index])
    img = Image.open(img_path).convert("RGB")

    return img, class_index


def load_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img, 0


if args.url != "":
    # Example image URL (tench/fish)
    img_url = "https://huggingface.co/datasets/frgfm/imagenette/resolve/main/val/n01440764/ILSVRC2012_val_00009111.JPEG"
    img_pil, class_idx = load_from_url(img_url)
else:
    img_pil, class_idx = load_imagenette_sample(
        class_index=args.class_index, sample_index=args.sample_index
    )

# Display original image
plt.imshow(img_pil)
plt.title(f"True label: {imagenette_classes[class_idx]}")
plt.axis("off")
plt.show()

## Load pre-trained ResNet-18
net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
net.eval()

features_blobs = []
finalconv_name = "layer4"


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


net._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())


## CAM generation function
def returnCAM(feature_conv, weight_softmax, class_idx, img_size=(224, 224)):
    size_upsample = img_size
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


img_tensor = preprocess(img_pil)
img_variable = img_tensor.unsqueeze(0)

logit = net(img_variable)

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()

print("Top predictions:")
for i in range(5):
    class_description = imagenet_classes[idx[i]]

    primary_class_name = class_description.split(",")[0]

    print(f"{probs[i]:.3f} -> {primary_class_name}")

## Generate and visualize CAM
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
img_cv = cv2.resize(img_cv, (224, 224))
heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)
result = heatmap * 0.4 + img_cv * 0.5

result_display = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_pil)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result_display)
plt.title(
    f"CAM for: {imagenette_classes[idx[0]] if idx[0] < len(imagenette_classes) else idx[0]}\n(prob: {probs[0]:.2f})"
)
plt.axis("off")

plt.tight_layout()
plt.show()

# Save results
output_file = "imagenette_CAM_result.jpg"
cv2.imwrite(output_file, cv2.cvtColor(result_display, cv2.COLOR_RGB2BGR))
print(f"CAM visualization saved to {output_file}")
