import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import wandb
import random

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_seismic_submission(model, args, output_path = None, split = 'Validation', iters=24):
    if output_path is None:
        flow_file_dir = Path(args.checkpoint)/args.name/f'{split}_data'
    else:
        flow_file_dir = Path(output_path)/f'{split}_data'
    flow_file_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    dataset = datasets.SeismicDataset(root = args.root, split=split, equalize=args.equalize)
    for ds_id in trange(len(dataset),desc=f'Saving {split} |', leave=True):
        image1, image2, flow_gt, valid_gt = dataset[ds_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        np_flow_pr = flow_pr.squeeze().numpy(force = True)

        flow_file_parent, flow_file_name = dataset.flow_list[ds_id].parts[-2:]
        flow_file = flow_file_dir/flow_file_parent/flow_file_name
        flow_file.parent.mkdir(exist_ok=True)

        frame_utils.writeSeismicFlowCSV(flow_file, np_flow_pr)


@torch.no_grad()
def validate_seismic(model, args, iters=24):
    """ Perform evaluation on the Seismic (valid) split """
    model.eval()
    val_dataset = datasets.SeismicDataset(root = args.root, split='Validation', equalize=args.equalize)
    epe_list = []
    Kepe_list = []
    Kout_list = []


    vis_sample_element = random.sample(range(len(val_dataset)),5)
    pp_img_list = []
    ps_img_list = []
    flow_gt_list = []
    valid_gt_list = []
    flow_low_list = []
    flow_pr_list = []

    flow_loss = 0.0

    for val_id in tqdm(list(range(len(val_dataset))), desc = 'Validation'):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = (valid_gt.view(-1) >= 0.5) & (mag < args.max_flow)

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        Kepe_list.append(epe[val].mean().item())
        Kout_list.append(out[val].cpu().numpy())

        # valid = (valid_gt >= 0.5) & (mag < args.max_flow)
        i_loss = (flow_pr[0].cpu() - flow_gt).abs()
        flow_loss += (val * i_loss.view(-1)).mean()

        if val_id in vis_sample_element:
            pp_img_list.append(wandb.Image(image1, caption=f"{(val_dataset.image_list[val_id][0]).stem}->PP"))
            ps_img_list.append(wandb.Image(image2, caption=f"{(val_dataset.image_list[val_id][1]).stem}->PS"))
            flow_gt_list.append(wandb.Image(flow_gt, caption=f"{(val_dataset.flow_list[val_id]).stem}->flow_gt"))
            valid_gt_list.append(wandb.Image(valid_gt, caption=f"{(val_dataset.flow_list[val_id]).stem}->valid_gt"))
            flow_low_list.append(wandb.Image(flow_low, caption=f"{(val_dataset.image_list[val_id][0]).stem}->flow_low"))
            flow_pr_list.append(wandb.Image(flow_pr, caption=f"{(val_dataset.image_list[val_id][0]).stem}->flow_pr"))

    # epe = np.mean(np.concatenate(epe_list))
    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)

    Kepe_list = np.array(Kepe_list)
    Kout_list = np.concatenate(Kout_list)

    Kepe = np.mean(Kepe_list)
    f1 = 100 * np.mean(Kout_list)

    return {'val_kitti-epe': Kepe, 'val_kitti-f1': f1, 'val_epe':epe, 'val_px1':px1, 'val_px3':px3, 'val_px5':px5, 'val_loss':flow_loss,
            'pp_img_list':pp_img_list, 'ps_img_list':ps_img_list, 'flow_gt_list':flow_gt_list, 
            'valid_gt_list':valid_gt_list, 'flow_low_list':flow_low_list, 'flow_pr_list':flow_pr_list, 
            }

@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)


