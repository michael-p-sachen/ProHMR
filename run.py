"""
Example usage:
python demo.py --checkpoint=path/to/checkpoint.pt --img_folder=/path/to/images --keypoint_folder=/path/to/json --out_folder=/path/to/output --run_fitting

Please keep in mind that we do not recommend to use `--full_frame` when the image resolution is above 2K because of known issues with the data term of SMPLify.
In these cases you can resize all images such that the maximum image dimension is at most 2K.
"""
import torch
import argparse
import os
import cv2
from tqdm import tqdm

from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMR, SMPL
from prohmr.optimization import KeypointFitting, MultiviewRefinement
from prohmr.utils import recursive_to
from prohmr.datasets import OpenPoseDataset
from prohmr.utils.renderer import Renderer


def write_obj(vertices, faces, path):
    with open(path, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def scale_vertices(real_height, vertices):
    min_y = min(vertices[:1])
    max_y = max(vertices[:1])
    pred_height = max_y - min_y
    scaling_factor = pred_height / real_height

    scaled_verts = []
    for vertex in vertices:
        scaled_verts.append(np.array(vertex[0] * scaling_factor, vertex[1] * scaling_factor, vertex[2] * scaling_factor))

    return scaled_verts


def main():
    parser = argparse.ArgumentParser(description='ProHMR evaluation code')
    parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
    parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
    parser.add_argument('--img_folder', type=str, required=True, help='Folder with input images')
    parser.add_argument('--keypoint_folder', type=str, required=True, help='Folder with corresponding OpenPose detections')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--out_format', type=str, default='jpg', choices=['jpg', 'png'], help='Output image format')
    parser.add_argument('--run_fitting', dest='run_fitting', action='store_true', default=False, help='If set, run fitting on top of regression')
    parser.add_argument('--run_multiview', dest='run_multiview', action='store_true', default=False, help='If set, run multi view fitting on top of regression')
    parser.add_argument('--write_image', dest="write_image", default=True, help='Batch size for inference/fitting')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, run fitting in the original image space and not in the crop.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--height_in_meters', type=float, default=None, help="Height to Scale avatar")

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

    # Init Optimizations
    optimize = None
    if args.run_fitting:
        fittingObject = KeypointFitting(model_cfg)
    if args.run_multiview:
        multiviewObject = MultiviewRefinement(model_cfg)

    # Create a dataset on-the-fly
    dataset = OpenPoseDataset(model_cfg, img_folder=args.img_folder, keypoint_folder=args.keypoint_folder,
                              max_people_per_image=1)

    # Setup a dataloader with batch_size = 1 (Process images sequentially)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)
    write_image = args.write_image

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Go over each image in the dataset
    for i, batch in enumerate(tqdm(dataloader)):

        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

            # Posed SMPL
            simp = out['smpl_output']
            write_obj(simp.vertices.detach().cpu().numpy()[0], model.smpl.faces, args.out_folder + "/posed.obj")

            batch_size = batch['img'].shape[0]
            if write_image:
                for n in range(batch_size):
                    img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
                    regression_img = renderer(out['pred_vertices'][n, 0].detach().cpu().numpy(),
                                              out['pred_cam_t'][n, 0].detach().cpu().numpy(),
                                              batch['img'][n])
                    cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_regression.{args.out_format}'),
                                255 * regression_img[:, :, ::-1])

            # UnPosedSimp
            orient = out['pred_smpl_params']['global_orient']
            betas = out['pred_smpl_params']['betas']
            pose = out['pred_smpl_params']['body_pose']
            pose = [0 for x in pose] # Set data to zeros
            params = {orient, betas, pose}
            smpl_output = self.smpl(**{k: v.float() for k,v in params}, pose2rot=False)
            vertices = smpl_output.vertices.detach().cpu().numpy()[0]

            if args.height_in_meters is not None:
                vertices = scale_vertices(args.height_in_meters, vertices)

            write_obj(vertices, model.smpl.faces, args.out_folder + "/unposed.obj")

        if args.run_fitting:
            opt_out = model.downstream_optimization(regression_output=out,
                                                    batch=batch,
                                                    opt_task=fittingObject,
                                                    use_hips=False,
                                                    full_frame=args.full_frame)

            # Optimized Simp
            simp = opt_out['smpl_output']
            write_obj(simp.vertices.detach().cpu().numpy()[0], model.smpl.faces, args.out_folder + "/posed_fit_kp.obj")

            if write_image:
                for n in range(batch_size):
                    img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
                    fitting_img = renderer(opt_out['vertices'][n].detach().cpu().numpy(),
                                           opt_out['camera_translation'][n].detach().cpu().numpy(),
                                           batch['img'][n], imgname=batch['imgname'][n], full_frame=args.full_frame)
                    cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_fitting.{args.out_format}'),
                                255 * fitting_img[:, :, ::-1])

            # Unposed Optimized Simp
            orient = opt_out['smpl_params']['global_orient']
            betas = opt_out['smpl_params']['betas']
            pose = opt_out['smpl_params']['body_pose']
            pose = [0 for x in pose]
            params = {orient, betas, pose}
            smpl_output = self.smpl(**{k: v.float() for k, v in params}, pose2rot=False)

            vertices = smpl_output.vertices.detach().cpu().numpy()[0]

            if args.height_in_meters is not None:
                vertices = scale_vertices(args.height_in_meters, vertices)

            write_obj(vertices, model.smpl.faces, args.out_folder + "/unposed_fit_kp.obj")

        # Multiview fitting
        if args.run_multiview:
            opt_out = model.downstream_optimization(regression_output=out, opt_task=multiviewObject, batch=batch)
            simp = opt_out['smpl_output']
            write_obj(simp.vertices.detach().cpu().numpy()[0], model.smpl.faces, args.out_folder + "/posed_fit_mv.obj")

            orient = opt_out['smpl_params']['global_orient']
            betas = opt_out['smpl_params']['betas']
            pose = opt_out['smpl_params']['body_pose']
            pose = [0 for x in pose]
            params = {orient, betas, pose}
            smpl_output = self.smpl(**{k: v.float() for k, v in params}, pose2rot=False)

            vertices = smpl_output.vertices.detach().cpu().numpy()[0]

            if args.height_in_meters is not None:
                vertices = scale_vertices(args.height_in_meters, vertices)
            write_obj(vertices, model.smpl.faces,args.out_folder + "/unposed_fit_mv.obj")


if __name__ == "__main__":
    main()
