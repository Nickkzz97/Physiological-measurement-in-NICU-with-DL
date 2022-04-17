import numpy as np
import pandas as pd
from scipy.signal import butter
# import math
import scipy
import scipy.io
from utils import detrend, mag2db


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
        fmask2 = (f >= 1.5) & (f <= 3.5)
    else:
        fmask2 = (f >= 0.08) & (f <= 0.5)
    allPower = np.sum(np.take(pxx, np.where(fmask2 == True)))
    SNR_temp = mag2db(sPower / (allPower - sPower))
    return SNR_temp
def meanHR(gtPath,PRlength):
    gtFile = pd.read_excel(gtPath,index_col='Frame', usecols = "A,B")
    frames, PR = gtFile.index,gtFile['PR']
#     print('max',np.max(PR))
    MeanPR = []
    for j in range(0,PRlength,200):
        PRmean = np.mean(PR[j:(j+200)])
        MeanPR.append(PRmean)
    return np.array(MeanPR)
def meanHR_06(gtPath,PRlength):
    gtFile = pd.read_excel(gtPath,index_col='Frame', usecols = "A,B")
    frames, PR = gtFile.index,gtFile['PR']
#     print('max',np.max(PR))
    MeanPR = []
    for j in range(1400,1400+PRlength,200):
        PRmean = np.mean(PR[j:(j+200)])
        MeanPR.append(PRmean)
    return np.array(MeanPR)
def scatter_plot_with_correlation_line(x, y,i,n,patnum):
    # Create scatter plot
    
    plt.subplot(2,n,i).scatter(x, y)

    # Add correlation line
    axes = plt.gca()
    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
#     plt.subplot(n//2,n-(n//2),i)
    plt.subplot(2,n,i).plot(X_plot, m*X_plot + b, '-')
    plt.title(patnum)
def bland_altman_plot(data1, data2,i,n,patnum, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    
    plt.subplot(2,n//2,(i)).scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.title(patnum)
    


# predictions, labels = ts_outputs_numpy.reshape(-1, 1), ts_labels_numpy.reshape(-1, 1)
def evluate(predictions, labels,window_size =300, fs = 25, signal = 'pulse',bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [1.5 / fs * 2, 4 / fs * 2], btype='bandpass') # 2.5 -> 1.7
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
            fmask_pred = np.argwhere((f_prd >= 1.5) & (f_prd <= 3.5))  # regular Heart beat are 0.75*60 and 2.5*60
        else:
            fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
        pred_window = np.take(f_prd, fmask_pred)
        # Labels FFT
        f_label, pxx_label = scipy.signal.periodogram(label_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse':
            fmask_label = np.argwhere((f_label >= 1.5) & (f_label <= 3.5))  # regular Heart beat are 0.75*60 and 2.5*60
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
    return HR, HR0, MAE, RMSE, meanSNR