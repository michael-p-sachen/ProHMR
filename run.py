import torch
import argparse
import os
import cv2
from tqdm import tqdm

from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMR
from prohmr.optimization import KeypointFitting
from prohmr.utils import recursive_to
from prohmr.datasets import OpenPoseDataset
from prohmr.utils.renderer import Renderer

parser = argparse.ArgumentParser(description='ProHMR demo code')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--img_folder', type=str, required=True, help='Folder with input images')
parser.add_argument('--keypoint_folder', type=str, required=True, help='Folder with corresponding OpenPose detections')
parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
parser.add_argument('--out_format', type=str, default='jpg', choices=['jpg', 'png'], help='Output image format')
parser.add_argument('--run_fitting', dest='run_fitting', action='store_true', default=False, help='If set, run fitting on top of regression')
parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, run fitting in the original image space and not in the crop.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')


args = parser.parse_args()

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if args.model_cfg is None:
    model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)

# Setup model
model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

if args.run_fitting:
    keypoint_fitting = KeypointFitting(model_cfg)

# Setup a dataloader with batch_size = 1 (Process images sequentially)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False)

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smpl.faces)

if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)


