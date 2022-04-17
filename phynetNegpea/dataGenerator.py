import torch
import math
import h5py
import pandas as pd
import os
import scipy.io
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.signal import butter
from skimage import transform as tfm
from utils import* #ReferenceProcessor

class rPPG_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, patpath,ppgpath,csvpath, dim=(128,128),fs = 25, signal='pulse'):
        self.dim = dim  ###36
        self.task_data_path = []
        self.signal = 'pulse'
        self.patpath = patpath
        self.ppgpath = ppgpath
#         self.Dataset = Dataset
#         self.fs = fs
        self.csvdata = pd.read_csv(csvpath,header=None)
        self.ppglen = 148 #frame_depth
        self.fs = 25
        subjectlist = self.csvdata[0].to_list()
        partlist = self.csvdata[1].to_list()
        nfiles = self.csvdata.shape[0]
        for ii in range(nfiles):
            subnum = str(subjectlist[ii]) 
            partnum = str(partlist[ii])
            print(subnum, partnum)
            patPath =sorted(Path(self.patpath).glob('{}/*/*/{}'.format(subnum,partnum)))[0]
#             print(ppgpath,subnum, partnum)

            ppgPath =sorted(Path(self.ppgpath).glob('{}/*/*/{}'.format(subnum,partnum)))[0]


            facePath = sorted(Path(patPath).glob('facecrop.h5'))[0]
            
            ppgfile = h5py.File (sorted(Path(ppgPath).glob('*CHROM*'))[0],'r')

            if (facePath.exists()):
#                 print(facePath)
                f1 = h5py.File(Path(facePath), 'r')
                facedata = np.array(f1["facecrop"])
            else:
                print('.h5 file not created')
            label = ppgfile['ppg']
#             print(facedata.shape, label.shape)
            seglen = len(label)//self.ppglen
            
#             trainlen = int((seglen//5)*3)
#             validlen = (seglen//5)
            
#             facemin = np.min(facedata)
#             facemax = np.max(facedata)
            for jj in range(0,seglen):
                    idx = [jj*self.ppglen,(jj+1)*self.ppglen]
                    self.task_data_path.append([idx, facePath,ppgfile, subnum, partnum])
#             if state =='train':
#                 for jj in range(0,trainlen):
#                     trainidx = [jj*self.ppglen,(jj+1)*self.ppglen]
#                     self.task_data_path.append([trainidx, facePath,ppgPath, subnum, facemin, facemax])
#             elif state =='valid':
#                 for jj in range(trainlen,trainlen+validlen):
#                     valididx = [jj*self.ppglen,(jj+1)*self.ppglen]
#                     self.task_data_path.append([valididx, facePath,ppgPath, subnum, facemin, facemax])
#             elif state =='test':
#                 for jj in range(trainlen+validlen,seglen):
#                     testidx = [jj*self.ppglen,(jj+1)*self.ppglen]
#                     self.task_data_path.append([testidx, facePath,ppgPath, subnum, facemin, facemax])
    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.task_data_path))#/self.batch_size


    def __getitem__(self, index):
        
#         if self.Dataset =='UBFC':
        idx, facePath,ppgPath,patnum,partnum = self.task_data_path[index]
#         print(partnum)
        
        if (facePath.exists()):
                f1 = h5py.File(facePath, 'r')
                facedata = np.array(f1["facecrop"])
#                     label = np.array(f1["ppg"])
#                     labeldiff = np.array(f1["ppgdiff"])
        else:
            print('.h5 file not created')
        label = ppgPath['ppg']
#         facedata = (facedata-facemin)/(facemax-facemin)
#         refproc = ReferenceProcessor(label)
#         refproc.calculate()
#         labelnorm = refproc.training_label
#         labelnorm = (label-labelmin)/(labelmax-labelmin)
#         print('label:',min(labelnorm),max(labelnorm))
        faceidx = facedata[idx[0]:idx[1],:,:,:]
        label = label[idx[0]:idx[1]]
        facenorm = normalize(faceidx)


        if self.signal == 'pulse':
            [b, a] = butter(1, [1.3 / self.fs * 2, 4 / self.fs * 2], btype='bandpass')
            label = scipy.signal.filtfilt(b, a, np.squeeze(label))
#             label = np.float32(label-np.min(label))/(np.max(label)-np.min(label))
        labelnorm = normalize(label)
        faceData = np.transpose(facenorm,  [3,0, 1, 2])  ##frame,channel,w,h  
        faceData = tfm.resize(faceData,[faceData.shape[0],faceData.shape[1],self.dim[0],self.dim[1]])
#         print('min max:',np.min(faceData), np.max(faceData), np.min(label), np.max(label))
        inputset = [faceData,labelnorm,patnum, partnum]
        return inputset