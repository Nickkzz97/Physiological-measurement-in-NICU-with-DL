import argparse
import json
import sys
import random
import shutil
import functools
import numpy as np
import os
import pathlib
from pathlib import Path
import time
import logging
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import scipy.io
from scipy.signal import butter
from tensorboardX import SummaryWriter
import pdb
from matplotlib import pyplot as plt

# #from utils import  get_nframe_video
from loss import NegPeaLoss
from dataGenerator import rPPG_Dataset
from model import PhysNet
#from metrics import calculate_metric
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# torch.manual_seed(100)
# np.random.seed(100)

def create_datasets(args):
    train_data = rPPG_Dataset(args.data_dir,args.ppg_dir, args.train_txt, dim=(128,128))
    dev_data = rPPG_Dataset(args.data_dir,args.ppg_dir, args.dev_txt,  dim=(128,128))

    return dev_data, train_data

def create_data_loaders(args):

    dev_data, train_data = create_datasets(args)
    
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]
    print('len:',len(dev_data),len(train_data),len(display_data))
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True

    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        shuffle=False

        )

    display_loader = DataLoader(
        dataset=display_data,
        batch_size=args.disp_size,

    )

    return train_loader, dev_loader, display_loader

def train_epoch(args, epoch, model,data_loader, optimizer, writer):
    
    model.train()
    running_loss = 0.0
    avg_loss = 0
    tr_loss = 0
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    for iter, data in enumerate(tqdm(data_loader)):
        
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        inputs = inputs.float()
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(inputs)
#         criterion = nn.MSELoss()
        criterion = NegPeaLoss()
        loss = criterion(outputs.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()
        #running_loss += loss.item()
        tr_loss += loss
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )


        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
#         break

    return avg_loss, time.perf_counter() - start_epoch
def evaluate(args, epoch, model, data_loader, writer):

    model.eval()
    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            inputs,labels = data[0].to(args.device), data[1].to(args.device)
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
#             criterion = nn.MSELoss()
            criterion = NegPeaLoss()
            loss = criterion(outputs.squeeze(), labels.squeeze())           
            losses.append(loss.item())
#             break
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
    
       
    return np.mean(losses), time.perf_counter() - start

def visualize(args, epoch, model, data_loader, writer):

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            inputs,labels = data[0].to(args.device), data[1].to(args.device)
#             inputs = inputs.view(-1, 3, 32, 32)
#             labels = labels.view(-1, 1)
            inputs = inputs.float()
            labels = labels.float()

            output = model(inputs)
            output = output.cpu()
            labels = labels.cpu()
#             print(output.shape)
            output = output.view(args.disp_size,-1, 1)
            labels = labels.view(args.disp_size,-1, 1)
            
            #print("predicted: ", torch.min(output), torch.max(output))

            for ii in range(output.shape[0]):

                fig = plt.figure()
                plt.plot(labels[ii], "r-")
                plt.plot(output[ii], "b-")
                writer.add_figure("sample_{}".format(ii), fig, epoch)

            break

def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        
def build_model(args):
    
    model = PhysNet().to(args.device)
    
    return model
def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer 


def build_optim(args, params):
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer

def main(args):
    
    if Path(str(args.exp_dir)+'best_model.pt').exists():
        print('check point exists')
        checkpointFile = Path(args.exp_dir,'best_model.pt')
        checkpoint, model, optimizer = load_model(checkpointFile)
    else:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        model = build_model(args)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))
    print ("Model Built")
    
    
    args.data_parallel = False
#     model = build_model(args)
     
    optimizer = build_optim(args, model.parameters())
    best_dev_loss = 1e9
    start_epoch = 0
    print('start..')
#     logging.info(args)
#     logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    print('data loading done!!')
    train_loader, dev_loader, display_loader = create_data_loaders(args)
    checkpoint_file = Path(args.exp_dir / 'summary')
#     if (checkpoint_file).exists():
#         print('check point file exists')
#         checkpoint, model, optimizer = load_model(sorted(checkpoint_file.glob('event*'))[0])
    if args.data_parallel:
        model = torch.nn.DataParallel(model)   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        
        train_loss,train_time = train_epoch(args, epoch, model,train_loader,optimizer,writer)
        print('trainloss:',train_loss)
#         pdb.set_trace()
        dev_loss,dev_time = evaluate(args, epoch, model, dev_loader, writer)
        visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Train setup for PhysNet')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--disp-size', default=5, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default= 2, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('-exp', '--exp_name', type=str,
                    help='experiment name')
    
    parser.add_argument('-o', '--exp_dir', type=pathlib.Path, default='CAN3Dcheckpoints_refined',
                        help='Location for parameter checkpoints and samples')

    parser.add_argument('-tr_txt', '--train_txt', type=str, help='train csv file')
    parser.add_argument('-dev_txt', '--dev_txt', type=str, help='dev csv file')
    parser.add_argument('-i', '--data_dir', type=str,help='Location for the dataset')
    parser.add_argument('-ppg_dir', '--ppg_dir', type=str,help='Location for the ppg')


    parser.add_argument('--eval_only', type=int, default=0,
                                        help='if eval only')
    parser.add_argument('--signal', type=str, default='pulse')
    parser.add_argument('-fd', '--frame_depth', type=int, default=64,
                    help='frame_depth for 3DCNN')
    parser.add_argument('-freq', '--fs', type=int, default=25,
                help='shuffle samples')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #print (args)
    main(args)
    
    
    
