import sys
sys.path.append('core')

from argparse import Namespace
import argparse

import datasets
import evaluate
from raft import SeismicRAFT as RAFT

from pathlib import Path
from tqdm.auto import tqdm


import torch
import torch.nn as nn

def get_default_args(args = None):
    if args is None:
        args = Namespace(
            root = 'PP_PS_data',
            checkpoint_file = 'checkpoints/seismic-raft_7000.pth',
            output_path = None,
            
            small=False,
            equalize=True,

            gpus=[0],
            mixed_precision=False,
        )

    return get_args(args)

def get_args(args = None):

    if args is None:
        parser = argparse.ArgumentParser()

        parser.add_argument('--root', help="path to dataset")
        parser.add_argument('--checkpoint_file', help="path to saved checkpoint .pth file", default='./checkpoints/seismic-raft_7000.pth')
        parser.add_argument('--output_path', default=None, help="output path to save flow csv files. Optional")

        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--equalize', action='store_true', help='equalize histogram')

        parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        args = parser.parse_args()

    return args

def get_model(args):
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    checkpoint = torch.load(Path(args.checkpoint_file))
    model.load_state_dict(checkpoint['model'], strict=False)

    return model


if __name__ == '__main__':
    args = get_args()

    model = get_model(args)
    evaluate.create_flow_submission(model,  args)

