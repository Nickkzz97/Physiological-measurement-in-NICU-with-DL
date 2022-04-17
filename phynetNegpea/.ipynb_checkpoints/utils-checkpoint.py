import glob
import os
import cv2
import h5py
import numpy as np
from scipy.sparse import spdiags
import scipy.io
from skimage.util import img_as_float

def CAN_preprocess_raw_video(videoFilePath, dim=36):

    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath);
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    dims = img.shape
    #########################################################################
    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
        vidLxL = cv2.resize(img_as_float(img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim, dim), interpolation = cv2.INTER_AREA)
        vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1/255)] = 1/255
        Xsub[i, :, :, :] = vidLxL
        success, img = vidObj.read() # read the next one
        i = i + 1
#     plt.imshow(Xsub[0])
#     plt.title('Sample Preprocessed Frame')
#     plt.show()
    #########################################################################
    # Normalized Frames in the motion branch
    normalized_len = len(t) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)
    #########################################################################
    # Normalize raw frames in the apperance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub  / np.std(Xsub)
    Xsub = Xsub[:totalFrames-1, :, :, :]
    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis = 3);
    return dXsub
def get_nframe_video(path):
    temp_f1 = h5py.File(path, 'r')
    temp_dysub = np.array(temp_f1["dysub"])
    nframe_per_video = temp_dysub.shape[0]
    return nframe_per_video

def normalize(mat_in):
    mat= mat_in.astype('float')
    return (2*(mat-mat.min())/(mat.max()-mat.min()))-1
def normalize01(mat_in):
    mat= mat_in.astype('float')
    return (mat-mat.min())/(mat.max()-mat.min())

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def mag2db(mag):
    """Convert a magnitude to decibels (dB)
    If A is magnitude,
        db = 20 * log10(A)
    Parameters
    ----------
    mag : float or ndarray
        input magnitude or array of magnitudes
    Returns
    -------
    db : float or ndarray
        corresponding values in decibels
    """
    return 20. * np.log10(mag)
class ReferenceProcessor:
    """
    Reference pre-processor for DeepPhys architecture.
    Derivates and normalizes the reference signal.
    """

    def __init__(self, signal):
        self.signal = signal.astype(np.float)
        self.n = signal.size-1
#         print(f"The length of training label: {self.n}")
        self.training_label = np.empty(shape=(self.n,), dtype=np.float)

    def calculate(self):
#         self.__derivative()
        self.__scale()

    def __derivative(self):
#         print("Derivating the signal...")
        for i in range(self.n):
            self.training_label[i] = self.signal[i+1]-self.signal[i]

    def __scale(self):
#         print("Scaling the signal...")

        part = 0
        window = 30

        while part < (len(self.training_label) // window) - 1:
            self.training_label[part*window:(part+1)*window] /= np.std(self.training_label[part*window:(part+1)*window])
            part += 1

        if len(self.training_label) % window != 0:
            self.training_label[part * window:] /= np.std(self.training_label[part * window:])