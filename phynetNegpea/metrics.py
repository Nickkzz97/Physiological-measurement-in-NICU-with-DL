# %% Import
import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
import math

from utils import detrend, mag2db

import matplotlib.pyplot as plt
import neurokit2 as nk
import seaborn as sns

# %% Helper Function

def calculate_HR(pxx_pred, frange_pred, fmask_pred, pxx_label, frange_label, fmask_label):
    pred_HR = np.take(frange_pred, np.argmax(np.take(pxx_pred, fmask_pred), 0))[0] * 60
    ground_truth_HR = np.take(frange_label, np.argmax(np.take(pxx_label, fmask_label), 0))[0] * 60
    return pred_HR, ground_truth_HR


def calculate_SNR(pxx_pred, f_pred, currHR, signal):
    currHR = currHR/60
    f = f_pred
    pxx = pxx_pred
    gtmask1 = (f >= currHR - 0.1) & (f <= currHR + 0.1)
    gtmask2 = (f >= currHR * 2 - 0.1) & (f <= currHR * 2 + 0.1)
    sPower = np.sum(np.take(pxx, np.where(gtmask1 | gtmask2)))
    if signal == 'pulse':
        fmask2 = (f >= 0.75) & (f <= 4)
    else:
        fmask2 = (f >= 0.08) & (f <= 0.5)
    allPower = np.sum(np.take(pxx, np.where(fmask2 == True)))
    SNR_temp = mag2db(sPower / (allPower - sPower))
    return SNR_temp
# %%  Processing


def calculate_metric(predictions, labels, signal='pulse', window_size=360, fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass') # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

    data_len = len(predictions)
    HR_pred = []
    HR0_pred = []
    mySNR = []
    for j in range(0, data_len, window_size):
        if j == 0 and (j+window_size) > data_len:
            pred_window = predictions
            label_window = labels
        elif (j + window_size) >  data_len:
            break
        else:
            pred_window = predictions[j:j + window_size]
            label_window = labels[j:j + window_size]
        if signal == 'pulse':
            pred_window = detrend(np.cumsum(pred_window), 100)
        else:
            pred_window = np.cumsum(pred_window)

        label_window = np.squeeze(label_window)
        if bpFlag:
            pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))

        pred_window = np.expand_dims(pred_window, 0)
        label_window = np.expand_dims(label_window, 0)
        # Predictions FFT
        f_prd, pxx_pred = scipy.signal.periodogram(pred_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse':
            fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        else:
            fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
        pred_window = np.take(f_prd, fmask_pred)
        # Labels FFT
        f_label, pxx_label = scipy.signal.periodogram(label_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse':
            fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        else:
            fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
        label_window = np.take(f_label, fmask_label)

        # MAE
        temp_HR, temp_HR_0 = calculate_HR(pxx_pred, pred_window, fmask_pred, pxx_label, label_window, fmask_label)
        temp_SNR = calculate_SNR(pxx_pred, f_prd, temp_HR_0, signal)
        HR_pred.append(temp_HR)
        HR0_pred.append(temp_HR_0)
        mySNR.append(temp_SNR)

    HR = np.array(HR_pred)
    HR0 = np.array(HR0_pred)
    mySNR = np.array(mySNR)

    MAE = np.mean(np.abs(HR - HR0))
    RMSE = np.sqrt(np.mean(np.square(HR - HR0)))
    meanSNR = np.nanmean(mySNR)
    return MAE, RMSE, meanSNR, HR0, HR


def Hci_ExtractHeartRate_Ecg(label,frameRate = 64,windowSize =150):
    ecg_dn_arr = np.array(np.squeeze(label))
    signals, info = nk.ecg_process(ecg_dn_arr, frameRate)
    ECG_R_peaks = signals["ECG_R_Peaks"]
    ECG_R_peaks.shape
    ECG_Y_dn_axis = []
    for i in range(np.array(ECG_R_peaks).shape[0]):
        #ecgIndex = np.where()
        if (np.array(ECG_R_peaks))[i] ==1:ECG_Y_dn_axis.append(signals['ECG_Clean'][i])#j = j+1
    ecgIndex_dn_Xaxis = np.array(np.where(np.array(ECG_R_peaks)==1))
    peaklist = np.array(ecgIndex_dn_Xaxis[0])
    ecgIndex_dn_Yaxis = np.array(ECG_Y_dn_axis)
#     plt.plot(signals['ECG_Clean'][:200])
#     plt.scatter(ecgIndex_dn_Xaxis[0][0:3],ecgIndex_dn_Yaxis[0:3])
    ECGclean =signals['ECG_Clean']
    data_len = signals['ECG_Clean'].shape[0]
    fs =frameRate
    BPMlist =[]
    window_size =windowSize
    for j in range(0, data_len, window_size):
        if j == 0 and (j+window_size) > data_len:
            label_window = ECGclean

        elif (j + window_size) >  data_len:
            break
        else:
            label_window = ECGclean[j:j + window_size]
        peakSublist = []
        for i in range(len(peaklist)):
            if j<=peaklist[i]<j+window_size:
                peakSublist.append(peaklist[i])
        RR_list=[]
        cnt = 0
        while (cnt < (len(peakSublist)-1)):
            RR_interval = (peakSublist[cnt+1] - peakSublist[cnt]) #Calculate distance between beats in # of samples
            ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
            RR_list.append(ms_dist) #Append to list
            cnt += 1

        bpm = 60000 / np.mean(RR_list)
       # print(peakSublist,bpm)
        BPMlist.append(bpm)
        HR0arr = np.array(BPMlist)
        #break
#         plt.figure(figsize=(7,3))        
#         plt.plot(HR0arr)
       
    return HR0arr
def Hci_ExtractHeartRate_ppg(predBP,frameRate=61,windowSize = 150,signal ='pulse',bpFlag=True):
    predictions = np.array(predBP)
    window_size=windowSize
    fs = frameRate
    signal ='pulse'
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass') # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

    HR_pred = []
    data_len = predictions.shape[0]
    for j in range(0, data_len, window_size):
        if j == 0 and (j+window_size) > data_len:
            pred_window = predictions
    #         label_window = labels

        elif (j + window_size) >  data_len:
            break
        else:
            pred_window = predictions[j:j + window_size]
            
        if signal == 'pulse':
            pred_window = detrend(np.cumsum(pred_window), 100)  ### removes noisy high frequency signals
        else:
            pred_window = np.cumsum(pred_window)

        if bpFlag:
            pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))   ### removes ripples

        pred_window = np.expand_dims(pred_window, 0)
    #     # Predictions FFT
        f_prd, pxx_pred = scipy.signal.periodogram(pred_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse':
            fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        else:
            fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
        pred_window = np.take(f_prd, fmask_pred)

        pred_HR = np.take(pred_window, np.argmax(np.take(pxx_pred, fmask_pred), 0))[0] * 60
    #     ground_truth_HR = np.take(label_window, np.argmax(np.take(pxx_label, fmask_label), 0))[0] * 60
        HR_pred.append(pred_HR)
        HR_pred_arr = np.array(HR_pred)

    return HR_pred_arr
def calculate_metric_Hci(HR_pred_arr,BPMarr):
    
    MAE = np.mean(np.abs(HR_pred_arr - BPMarr))
    RMSE = np.sqrt(np.mean(np.square(HR_pred_arr - BPMarr)))
    
    return MAE, RMSE