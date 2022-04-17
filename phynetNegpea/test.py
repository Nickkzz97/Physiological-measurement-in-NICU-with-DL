import numpy as np
import os
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import argparse
from pathlib import Path
import pathlib
import sys
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader
from model import PhysNet
from metrics import calculate_metric,calculate_metric_Hci
from metrics import Hci_ExtractHeartRate_Ecg, Hci_ExtractHeartRate_ppg
#from utils import  read_from_single_txt  #get_nframe_video,
from dataGenerator import rPPG_Dataset
import neurokit2 as nk
import pandas as pd
import seaborn as sns

def save_signal(signals, patnum,partnum, out_dir,Dataset, vidnum=0):
    
    print("out dir",out_dir)
    if vidnum:
        opdir = Path(str(out_dir) +'/'+Dataset + "/{}/{}".format(patnum[0], vidnum[0]))
    else:
        opdir = Path(str(out_dir) +'/'+Dataset+ "/{}/{}".format(patnum[0], partnum[0]))
    if not opdir.exists():
        print("Output dir not exist")
        opdir.mkdir(parents=True)
    fname = 'ppgsignal.h5'#.format(patnum[0],vidnum[0])
    

    path = os.path.join(opdir/fname)
    ppgSignal = signals.reshape(-1,1).squeeze()
    if os.path.exists(path):
        print("fname exist")
        with h5py.File(opdir/fname , 'a') as f:
            f["ppgsignal"].resize((f["ppgsignal"].shape[0] + ppgSignal.shape[0]), axis = 0)
            f["ppgsignal"][-ppgSignal.shape[0]:] = ppgSignal
    else:
        with h5py.File(opdir/fname , 'w') as hf:
            hf.create_dataset('ppgsignal', data=ppgSignal, maxshape=(None,), chunks=True)
def saveGtsignal(signals, patnum,partnum, out_dir,Dataset, vidnum=0):
    
    print("out dir",out_dir)
    if vidnum:
        opdir = Path(str(out_dir)+'/'+Dataset + "/{}/{}".format(patnum[0], vidnum[0]))
    else:
        opdir = Path(str(out_dir) +'/'+Dataset+ "/{}/{}".format(patnum[0], partnum[0]))
    if not opdir.exists():
        print("Output dir not exist")
        opdir.mkdir(parents=True)
    fname = 'ppgGtsignal.h5'#.format(patnum[0],vidnum[0])
    

    path = os.path.join(opdir/fname)
    ppgSignal = signals.reshape(-1,1).squeeze()
    if os.path.exists(path):
        print("fname exist")
        with h5py.File(opdir/fname , 'a') as f:
            f["ppgsignal"].resize((f["ppgsignal"].shape[0] + ppgSignal.shape[0]), axis = 0)
            f["ppgsignal"][-ppgSignal.shape[0]:] = ppgSignal
    else:
        with h5py.File(opdir/fname , 'w') as hf:
            hf.create_dataset('ppgsignal', data=ppgSignal, maxshape=(None,), chunks=True)

def create_datasets(args):
   # test_data = rPPG_Dataset(data_dir, dev_txt,Dataset,  dim=(36,36), batch_size=32, frame_depth=10, signal='pulse')
    
    test_data = rPPG_Dataset(args.data_path,args.ppg_dir,args.valid_csv_path, dim=(128,128))
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False
        )

    return test_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = PhysNet().to(args.device)

    model.load_state_dict(checkpoint['model'])
    return model



def runTest(args, model, data_loader, Dataset, dim=(128,128)):
    model.eval()
    #print('fs,window_size:',fs,window_size)
#     fs,window_size =fs,window_size
    with torch.no_grad():
        print('Evaluate...')
        criterion = nn.MSELoss()
        ts_loss = 0.0


        for i, ts_data in enumerate(data_loader):
            ts_inputs, ts_labels,patnum, partnum  =  ts_data[0].to(args.device), ts_data[1].to(args.device), ts_data[2], ts_data[3]
            print(partnum)
            ts_inputs, ts_labels = ts_inputs.float(), ts_labels.float()
            ts_outputs = model(ts_inputs)
            loss = criterion(ts_outputs.squeeze(), ts_labels.squeeze())
            ts_loss += loss
            ts_outputs_numpy = ts_outputs.cpu().numpy().squeeze()
            ts_labels_numpy = ts_labels.cpu().numpy()
            save_signal(ts_outputs_numpy, patnum,partnum, args.out_dir,Dataset)
            saveGtsignal(ts_labels_numpy, patnum,partnum, args.out_dir,Dataset)

    
def main(args):
    
    data_loader = create_datasets(args)
    model = load_model(args.checkpoint)
    runTest(args, model, data_loader,args.dataset, dim=(128,128))

def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for RPPG extraction")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the physNet model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the signal to')
    parser.add_argument('--batch-size', default=1, type=int, help='Mini-batch size')
#     parser.add_argument('--data-parallel',de)
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')
    parser.add_argument('-ppg_dir', '--ppg_dir', type=str,help='Location for the ppg')
    parser.add_argument('--valid-csv-path',type=str,help='Path to test h5 files')
    parser.add_argument('--dataset',type=str,help='Name of dataset')
    parser.add_argument('--signal', type=str, default='pulse')
    parser.add_argument('-fd', '--framedepth', type=int, default=64,
                    help='frame_depth for physnet')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)