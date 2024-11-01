import os
from models.builder import build_model
from hydra import compose, initialize
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch.nn.functional as F
import numpy as np
from operator import itemgetter
import torch
from segmentation.datasets.pascal_context import PascalContextDataset
import argparse
from helpers.visualization import mask2rgb
from typing import List

initialize(config_path="configs", version_base=None)
PALETTE = list(PascalContextDataset.PALETTE)


def visualize_per_image(file_path: str, support_images: List[str], model: torch.nn.Module, device: str, output_dir: str):
    """
    Visualizes output segmentation mask using support images and saves it in the output directory.

    :param file_path: [str] path to the image file
    :param support_images: [list(str)] list of paths to support images for segmentation
    :param model: [torch.nn.module] loaded model for inference
    :param device: either "cpu" or "cuda"
    :param output_dir: [str] output directory
    :return:
    """
    assert os.path.isfile(file_path), f"No such file: {file_path}"

    img = Image.open(file_path).convert('RGB')
    img_tens = T.PILToTensor()(img).unsqueeze(0).to(device) / 255.

    h, w = img_tens.shape[-2:]
    name = file_path.split('.')[0]

    # Load and process support images
    support_tensors = []
    for support_path in support_images:
        assert os.path.isfile(support_path), f"No such file: {support_path}"
        support_img = Image.open(support_path).convert('RGB')
        support_tensors.append(T.PILToTensor()(support_img).unsqueeze(0).to(device) / 255.)

    # Prepare the model's support images input
    model.set_support_images(support_tensors)  # Assumes model can take support images for segmentation
    output = model(img_tens).cpu()
    output = F.interpolate(output, scale_factor=model.vit_patch_size, mode="bilinear", align_corners=False)[..., :h, :w]
    output = output[0].argmax(dim=0)
    mask = mask2rgb(output, PALETTE)
    fig = plt.figure()
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{name}_ours.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Save labels as a separate file
    fig = plt.figure()
    classes = np.unique(output).tolist()
    plt.imshow(np.array(itemgetter(*classes)(PALETTE)).reshape(1, -1, 3))
    plt.xticks(np.arange(len(classes)), list(map(os.path.basename, support_images)), rotation=45)
    plt.yticks([])
    plt.savefig(os.path.join(output_dir, f'{name}_labels.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CLIP-DINOiser demo with support images')
    parser.add_argument('--cfg', help='config file name', default='clip_dinoiser.yaml')
    parser.add_argument('--support_images', help='List of paths to support images', required=True, type=list_of_strings)
    parser.add_argument('--checkpoint_path', help='Path to the checkpoint file', default='./checkpoints/last.pt')
    parser.add_argument('--output_dir', help='Directory to save the output', default='.', required=False)
    parser.add_argument('--file_path', help='Path to the image file', required=True)

    args = parser.parse_args()
    return args

def list_of_strings(arg):
    return arg.split(',')

if __name__ == '__main__':
    args = parse_args()
    cfg = compose(config_name=args.cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    support_images = args.support_images

    model = build_model(cfg.model, class_names=None)  # class_names not needed for image-based segmentation

    assert os.path.isfile(args.checkpoint_path), "Checkpoint file doesn't exist"
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()
    model.to(device)
    os.path.exists(args.output_dir)

    visualize_per_image(args.file_path, support_images, model, device, args.output_dir)
