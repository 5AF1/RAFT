import sys
sys.path.append('core')

from argparse import Namespace
import argparse

import datasets
import evaluate
from raft import SeismicRAFT as RAFT

from pathlib import Path
from tqdm.autonotebook import tqdm
import numpy as np
import random
from pprint import pprint
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import wandb

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
# MAX_FLOW = 400
# SUM_FREQ = 100
# VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'tra_epe': epe.mean().item(),
        'tra_1px': (epe < 1).float().mean().item(),
        'tra_3px': (epe < 3).float().mean().item(),
        'tra_5px': (epe < 5).float().mean().item(),
        'tra_loss':flow_loss.item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

        self.args = args

    def _print_training_status(self):
        metrics_data = [(k,self.running_loss[k]/self.args.log_every) for k in sorted(self.running_loss.keys())]
        training_str = f"\n{{'Steps': {(self.total_steps+1):6d}, 'last lr': {(self.scheduler.get_last_lr()[0]):10.7f}, "
        # metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        metrics_str = ''
        for k,data in metrics_data:
            metrics_str += f"'{k}': {(data):7.5f}, "
        
        # print the training status
        print(training_str + metrics_str + '}')
        print('='*len(training_str + metrics_str))

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/self.args.log_every, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.args.log_every == self.args.log_every-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

class EarlyStopper:
    def __init__(self, patience=1, threshold=0):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.finish_flag = False

    def __call__(self, validation_loss = None):
        if validation_loss is None:
            return self.finish_flag
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.threshold):
            self.counter += 1
            if self.counter >= self.patience:
                self.finish_flag = True
                return True
        return False

def wandb_train(args):
    wandb.login()

    with wandb.init(entity = args.wandb_entity, project=args.wandb_project, id = args.wandb_run_id, resume=args.wandb_resume, config=args) as wandb_run:
        args = wandb.config

        args_dict = vars(args)
        pprint(args_dict)
        
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        torch.backends.cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        Path(args.checkpoint).mkdir(exist_ok=True)

        model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
        print(f"Parameter Count: {count_parameters(model)}")
        optimizer, scheduler = fetch_optimizer(args, model)
        scaler = GradScaler(enabled=args.mixed_precision)
        total_steps = 0
        epoch = 0

        if args.restore_ckpt is not None:
            checkpoint = torch.load(args.restore_ckpt)
            model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
            scheduler = checkpoint['scheduler']
            scaler.load_state_dict(checkpoint['scaler'], strict=False)
            total_steps = checkpoint['steps']
            epoch = checkpoint['epoch']

        model.cuda()
        model.train()

        train_loader = datasets.fetch_seismic_dataloader(args, split = 'Train')
        train_loader_len = len(train_loader)

        wandb.watch(model, log="all", log_freq=10)
        # logger = Logger(model, scheduler, args)

        should_keep_training = True
        early_stopper = EarlyStopper(patience=args.early_stop_patience, threshold=args.early_stop_threshold)

        while should_keep_training:
            for i_batch, data_blob in enumerate(tqdm(train_loader, desc = f'Epoch {epoch} Step {total_steps+1} of {args.num_steps}', total = train_loader_len)):
                optimizer.zero_grad()
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

                if args.add_noise and args.equalize:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 1023.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 1023.0)

                flow_predictions = model(image1, image2, iters=args.iters)            

                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma, max_flow=args.max_flow)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                wandb.log(metrics, step=total_steps)
                wandb.log({'last_lr': (scheduler.get_last_lr()[0])}, step = total_steps)

                if total_steps % args.validation_every == args.validation_every - 1:
                    PATH = Path(args.checkpoint)
                    PATH.mkdir(exist_ok=True)
                    PATH = PATH/f'{total_steps+1}_{args.name}.pth'
                    checkpoint = { 
                        'epoch': epoch,
                        'steps': total_steps,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler':scaler.state_dict(),
                        'scheduler': scheduler}
                    torch.save(checkpoint, PATH)

                    results = {}
                    
                    results.update(evaluate.validate_seismic(model.module,  args))

                    wandb.log(results, step=total_steps)

                    if early_stopper(results['val_loss']):
                        should_keep_training = False
                        args.name = +args.name
                        break
                    
                    model.train()
            
                total_steps += 1

                if total_steps > args.num_steps:
                    should_keep_training = False
                    break
                
            epoch += 1
    
    PATH = Path(args.checkpoint)
    PATH.mkdir(exist_ok=True)
    PATH = PATH/f'{"early_" if early_stopper() else ''}{args.name}.pth'
    torch.save(model.state_dict(), PATH)

def train(args):
    args_dict = vars(args)
    pprint(args_dict)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    Path(args.checkpoint).mkdir(exist_ok=True)

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print(f"Parameter Count: {count_parameters(model)}")

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    train_loader = datasets.fetch_seismic_dataloader(args, split = 'Train')
    train_loader_len = len(train_loader)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)
    epoch = 0

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(tqdm(train_loader, desc = f'Step {total_steps+1} of {args.num_steps}', total = train_loader_len)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)            

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma, max_flow=args.max_flow)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % args.validation_every == args.validation_every - 1:
                PATH = Path(args.checkpoint)
                PATH.mkdir(exist_ok=True)
                PATH = PATH/f'{total_steps+1}_{args.name}.pth'
                torch.save(model.state_dict(), PATH)

                results = {}
                
                results.update(evaluate.validate_seismic(model.module,  args.root, equalize=args.equalize))

                logger.write_dict(results)
                
                model.train()
            
            total_steps += 1
            epoch += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = Path(args.checkpoint)
    PATH.mkdir(exist_ok=True)
    PATH = PATH/f'{args.name}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH

def get_default_args():
    args = Namespace(
        wandb_entity = 'wandb_seismic-raft_team',
        wandb_project = 'seismic-raft',
        wandb_run_id = None, #############
        wandb_resume = False, #############

        name = None,
        root = '/Dataset',
        checkpoint = './checkpoints/',
        restore_ckpt = None, ###############
        small=False,
        equalize=True,

        lr=0.000125, num_steps=100000,
        batch_size=2,
        gpus=[0],
        mixed_precision=False,

        iters=12,
        wdecay=1e-05,epsilon=1e-08, 
        clip=1.0,
        dropout=0.0,
        gamma=0.85,
        add_noise=False,
        
        seed = 1234,
        log_every = 100,
        validation_every = 1000,
        max_flow = 400,

        num_workers = 2,
        pin_memory = False,
        shuffle = True,
        drop_last = True,

        early_stop_patience = 10,
        early_stop_threshold = 50,
    )

    return get_args(args)

def get_args(args = None):

    if args is None:
        parser = argparse.ArgumentParser()

        parser.add_argument('--wandb_entity', default='wandb_seismic-raft_team', help="wandb entity")
        parser.add_argument('--wandb_project', default='seismic-raft', help="wandb project")
        parser.add_argument('--wandb_run_id', default=None, help="wandb run id")
        parser.add_argument('--wandb_resume', action='store_true', help="resume wandb run")
        
        parser.add_argument('--name', default=None, help="name your experiment")
        # parser.add_argument('--stage', help="determines which dataset to use for training")
        parser.add_argument('--root', help="path to dataset")
        parser.add_argument('--checkpoint', help="path to save checkpoint", default='./checkpoints/')
        parser.add_argument('--restore_ckpt', help="restore checkpoint")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--equalize', action='store_true', help='equalize histogram')
        # parser.add_argument('--validation', type=str, nargs='+')

        parser.add_argument('--lr', type=float, default=0.00002)
        parser.add_argument('--num_steps', type=int, default=100000)
        parser.add_argument('--batch_size', type=int, default=6)
        # parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
        parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

        parser.add_argument('--iters', type=int, default=12)
        parser.add_argument('--wdecay', type=float, default=.00005)
        parser.add_argument('--epsilon', type=float, default=1e-8)
        parser.add_argument('--clip', type=float, default=1.0)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
        parser.add_argument('--add_noise', action='store_true')

        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--log_every', type=int, default=100)
        parser.add_argument('--validation_every', type=int, default=1000)
        parser.add_argument('--max_flow', type=int, default=400)

        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--pin_memory', action='store_true')
        parser.add_argument('--shuffle', action='store_true')
        parser.add_argument('--drop_last', action='store_true')

        parser.add_argument('--early_stop_patience', type=int, default=10)
        parser.add_argument('--early_stop_threshold', type=float, default=50)
        args = parser.parse_args()
    
    if args.restore_ckpt is not None and args.wandb_run_id is not None or args.wandb_resume is True:
        args.wandb_resume = 'allow'

    if args.wandb_run_id is None:
        args.wandb_run_id = wandb.util.generate_id()

    if args.name is None:
        args.name = f'{args.wandb_project}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.wandb_run_id}'
    
    return args

if __name__ == '__main__':
    args = get_args()

    train(args)