#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodologi defined in:

    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]

"""

# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import numpy as np

# Our Modules
import datasets
from datasets import utils

from models.MPNN import MPNN
from LogMetric import AverageMeter, Logger
from pathlib import Path, PurePath, PurePosixPath
import os 
import uuid 
import wandb
from datetime import datetime as dt
import sys
import argparse
import tqdm 

__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"


os.environ['WANDB_MODE'] = 'dryrun'
global PROJECT
PROJECT = "MPNN-Displace-Reaction"
logging = True

def is_rank_zero(args):
    return args.rank == 0

def log_images(params):
    raise NotImplementedError
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)

# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

global args, best_er1
PROJECT = "MPNN-Displace-Reaction-Training-Tuning"
logging = True

def main_worker(gpu, ngpus_per_node, args):
    # state gloabl variable for wandb and train 
    global PROJECT
    args.gpu = gpu 
    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Load data
    root = args.datasetPath

    if ngpus_per_node == 1:
        args.gpu = 0

    args.last_epoch = -1
    args.epoch = 0
    args.rank = 0

    # main worker
    print('Prepare files')
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    idx = np.random.permutation(len(files))
    idx = idx.tolist()

    index = int(len(files)*(1-args.split))
    valid_ids = [files[i] for i in idx[index+1:]]
    train_ids = [files[i] for i in idx[0:index]]
    del index 

    data_train = datasets.Qm9(root, args, train_ids, edge_transform=utils.qm9_edges, e_representation='raw_distance')
    data_valid = datasets.Qm9(root, args, valid_ids, edge_transform=utils.qm9_edges, e_representation='raw_distance')

    # Define model and optimizer
    print('Define model')

    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    print('\tStatistics')
    stat_dict = datasets.utils.get_graph_stats(data_valid, ['target_mean', 'target_std'])

    data_train.set_target_transform(lambda x: datasets.utils.normalize_data(x,stat_dict['target_mean'],
                                                                            stat_dict['target_std']))
    data_valid.set_target_transform(lambda x: datasets.utils.normalize_data(x, stat_dict['target_mean'],
                                                                            stat_dict['target_std']))
    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size, shuffle=True,
                                               collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)


    print('\tCreate model')
    in_n = [len(h_t[0]), len(list(e.values())[0])]
    hidden_state_size = 73
    message_size = 73
    n_layers = 3
    l_target = 1
    type ='regression'
    model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type=type)
    del in_n, hidden_state_size, message_size, n_layers, l_target, type

    args.multigpu = False
    print('Check cuda')
    if args.cuda:
        print('\t* Cuda')
        model = model.cuda()

    if args.gpu is not None:
        args.multigpu = True
        model = model.cuda(args.gpu)
        model = model.cuda(args.gpu)
    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # train for one epoch
    train(model, train_loader, valid_loader, args, optimizer, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root, experiment_name=args.name)


def train(model, train_loader, valid_loader, args, optimizer, epochs, lr=0.0001, device=None, experiment_name="Hyperparameter_tuning", root="."):
    
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        print(f"Training {args.name}.")

    # new parameters for documenting on W&B
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-bs{args.batch_size}-tep{args.epochs}-{uuid.uuid4()}"
    name = f"{args.name}_{run_id}"
    should_write = (args.rank == 0)
    
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'Displace-Reaction': # first change it into 'Displace-Reaction'
            PROJECT = PROJECT + f"-{args.dataset}"
            wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes, id=run_id)
            wandb.watch(model)


    print('Optimizer')
    criterion = nn.MSELoss()
    evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))

    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()

    # switch to train mode
    model.train()
    # leanring rate change strategy
    lr_step = (args.lr-args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])
    # if args.same_lr:
    #     print("Using same LR")
    #     params = model.parameters()
    # else:
    #     print("Using diff LR")
    #     params = [{"params": model.get_1x_lr_params(), "lr": lr / 10},
    #               {"params": model.get_10x_lr_params(), "lr": lr}]

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr_step, epochs=epochs, steps_per_epoch=len(train_loader),
                                            cycle_momentum=True,
                                            base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                            div_factor=args.div_factor,
                                            final_div_factor=args.final_div_factor)

    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)

    # Train loop
    for epoch in range(args.last_epoch+1, epochs):
        if should_log: wandb.log({"Epoch": epoch}, step=step)

        # for i, (g, h, e, target) in enumerate(train_loader):
        for i, (g, h, e, target) in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train\n",
                             total=len(train_loader)) if is_rank_zero(args) else enumerate(train_loader):

            # Prepare input data
            if args.cuda:
                g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
            g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

            # Measure data loading time
            data_time.update(time.time() - end)

            optimizer.zero_grad()

            # Compute output
            output = model(g, h, e)
            train_loss = criterion(output, target)

            # Logs
            losses.update(train_loss.data.item(), g.size(0))
            error_ratio.update(evaluation(output, target).data.item(), g.size(0))

            # compute gradient and do SGD step
            train_loss.backward()
            # cut big gradient
            # nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if should_log and step% 5 == 0:
                wandb.log({f"Train/{train_loss.name}": train_loss.item()}, step=step)
                wandb.log({f"Train/{error_ratio.name}": error_ratio.item()}, step=step)
            step += 1
            scheduler.step()

            if i % args.log_interval == 0 and i > 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                    .format(epoch, i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, err=error_ratio))
            # validate 
            if should_write and step % args.validate_every == 0:
                model.eval()
                metrics, val_si, er1 = validate(valid_loader, model, criterion, evaluation)

                print("Validated: {}".format(metrics))
                if should_log:
                    wandb.log({
                       f"Validate/{losses.name}": val_si.get_value()}, step=step)
                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)

                    # go into the wandb and file the right file to save the model
                    wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
                    utils.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_latest.pt",
                                             root=os.path.join(wandb.run.dir, "checkpoints"))
                if metrics['abs_rel'] < best_loss and should_write:
                    utils.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_best.pt",
                                             root=os.path.join(wandb.run.dir, "checkpoints"))
                    best_loss = metrics['abs_rel']

                # get the best checkpoint and test it with test set
                if args.resume:
                    checkpoint_dir = args.resume
                    best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
                    if not os.path.isdir(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    if os.path.isfile(best_model_file):
                        print("=> loading best model '{}'".format(best_model_file))
                        checkpoint = torch.load(best_model_file)
                        args.start_epoch = checkpoint['epoch']
                        best_acc1 = checkpoint['best_er1']
                        model.load_state_dict(checkpoint['state_dict'])
                        if args.cuda:
                            model.cuda()
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
                    else:
                        print("=> no best model found at '{}'".format(best_model_file))
                model.train()

        is_best = er1 > best_er1
        best_er1 = min(er1, best_er1)
        utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_er1': best_er1,
                        'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.resume)

    print('Epoch: [{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, err=error_ratio, loss=losses, b_time=batch_time))

    return model

def validate(val_loader, model, criterion, evaluation):
    with torch.no_grad():
        val_si = utils.RunningAverage()
        metrics = utils.RunningAverageDict()
        batch_time = AverageMeter()
        losses = AverageMeter()
        error_ratio = AverageMeter()

        # switch to evaluate mode
        model.eval()

        # for i, (g, h, e, target) in enumerate(val_loader):
        for i, (g, h, e, target) in tqdm(val_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else val_loader:

            # Prepare input data
            if args.cuda:
                g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
            g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

            # Compute output
            output = model(g, h, e)

            # Logs
            losses.update(criterion(output, target).data.item(), g.size(0))
            error_ratio.update(evaluation(output, target).data.item(), g.size(0))

            if i % args.log_interval == 0 and i > 0:
                
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                    .format(i, len(val_loader), batch_time=batch_time,
                            loss=losses, err=error_ratio))

        print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
            .format(err=error_ratio, loss=losses))
        metrics.update(utils.compute_errors(output, target))

    # return error_ratio.avg
    return metrics, val_si, error_ratio.avg

    
if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Neural message passing')

    parser.add_argument('--dataset', default='Displace-Reaction-03302021', help='dataset ID')
    parser.add_argument('--datasetPath', default='data/data/data_xyz-00227f42-3874-4de0-9abe-eb0f953a0a80', help='dataset path')
    parser.add_argument('--logPath', default='./log/qm9/mpnn/', help='log path')
    parser.add_argument('--plotLr', default=False, help='allow plotting the data')
    parser.add_argument('--plotPath', default='./plot/qm9/mpnn/', help='plot path')
    parser.add_argument('--resume', default='./checkpoint/qm9/mpnn/',
                        help='path to latest checkpoint')
    parser.add_argument("--root", default=".", type=str,
                            help="Root folder to save data in")
    parser.add_argument("--name", default="MPNN-Displace-Reaction")
    parser.add_argument("--split", default=0.15, type=float)

    # Optimization Options
    parser.add_argument('--bs', '--batch-size', type=int, default=20, metavar='N',
                        help='Input batch size for training (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=360, metavar='N',
                        help='Number of epochs to train (default: 360)')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-3, metavar='LR',
                        help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.6)')
    parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                        help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument("--same-lr", "--same_lr", default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")


    # i/o
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

    # train 
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--tags", default="tuning dataset splition", type=str, help="Wandb tags.")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")


    best_er1 = 0

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        print(arg_filename_with_prefix)
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # standardize variable
    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'

    # Folder to save 
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    # Configurate gpu
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if ngpus_per_node == 1:
        args.gpu = 0

    main_worker(args.gpu, ngpus_per_node, args)
