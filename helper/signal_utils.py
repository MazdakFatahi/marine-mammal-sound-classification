from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import argparse
# import os
# from random import choice
# from time import sleep, time
import numpy as np
from scipy.signal import welch, decimate

from pycochleagram import cochleagram as cgram
from pycochleagram import erbfilter as erb
from pycochleagram import utils

import matplotlib.pyplot as plt

import librosa  as lr
import librosa.display
import numpy as np

import soundfile as sf
from scipy import signal


def resample(sig_array, orig_sr, target_sr):
    return librosa.resample(sig_array, orig_sr=orig_sr, target_sr=target_sr)


def bandpass_filter(sig_array, sig_sr, start_frq, end_frq, filter_order = 6):
    sos = signal.butter(filter_order, [start_frq, end_frq], 'bandpass', fs=sig_sr, output='sos')
    return signal.sosfiltfilt(sos, sig_array)

# def bandpass_filter(sig_array, sig_sr, start_frq, end_frq, filter_order = 6):
#     sos = signal.butter(filter_order, [start_frq, end_frq], 'bandpass', fs=sig_sr, output='sos')
#     return signal.sosfiltfilt(sos, sig_array)

def melspectrogram_db(signal_array, samplig_rate, fmax, out_spec_path = None, cmap = 'auto', hop_length = 1024, n_fft=2048, n_mels=128, axis_off = True):
    sig, sr = signal_array, samplig_rate

    S = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
    fig = 0
    if cmap != 'auto':
        fig = lr.display.specshow(lr.amplitude_to_db(S, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    else:
        fig = lr.display.specshow(lr.amplitude_to_db(S, ref=np.max), cmap=None, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')


    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
    # plt.colorbar(format='%+2.0f dB');

    if axis_off:
        plt.axis('off')
    else:
        plt.title(f'Mel-Spectrogram-librosa')
        plt.xlabel("Time")
        plt.ylabel("Frequencies")
    
    # plt.show()

    if out_spec_path:
        fig.figure.savefig(out_spec_path,bbox_inches='tight',
            pad_inches = 0)
        fig.figure.clear()
        plt.close()
    return fig, S


def spectogram(signal_array, samplig_rate, out_spec_path = None, package = 'librosa-linear' , cmap = 'auto', hop_length = 1024, n_fft=2048, axis_off = True): # 
    sig, sr = signal_array, samplig_rate
    fig=0
    tf_sig=0
    if package == 'pyplot':
        fig = plt.figure()
        fig = fig.add_subplot(111)

        if cmap == 'auto':
            fig.specgram(sig,  cmap='viridis')
        else:
            fig.specgram(sig,  cmap=cmap)

    if package == 'librosa-log':
        tf_sig = lr.stft(sig, n_fft=n_fft, hop_length=hop_length)
        tf_sig = np.abs(tf_sig)

        if cmap != 'auto':
            fig = lr.display.specshow(lr.amplitude_to_db(tf_sig, ref=np.max), y_axis='log', x_axis='time', hop_length=hop_length, sr=sr)
        else:
            fig = lr.display.specshow(lr.amplitude_to_db(tf_sig, ref=np.max), cmap=None, y_axis='log', x_axis='time', hop_length=hop_length, sr=sr)



    if package == 'librosa-linear':
        tf_sig = lr.stft(sig, n_fft=n_fft)
        tf_sig = np.abs(tf_sig)

        if cmap != 'auto':
            fig = lr.display.specshow(lr.amplitude_to_db(tf_sig, ref=np.max), y_axis='linear', x_axis='time', sr=sr)
        else:
            fig = lr.display.specshow(lr.amplitude_to_db(tf_sig, ref=np.max), cmap=None, y_axis='linear', x_axis='time', sr=sr)


    if axis_off:
        plt.axis('off')
    else:
        plt.title(f'Spectrogram-{package}')
        plt.xlabel("Time")
        plt.ylabel("Frequencies")
    
    # plt.show()

    if out_spec_path:
        fig.figure.savefig(out_spec_path,bbox_inches='tight',
            pad_inches = 0)
        fig.figure.clear()
        plt.close()
    return fig, tf_sig




def cochleagram(signal_array, samplig_rate, out_coch_path = None, cmap='magma', n_subbands = 38, sample_factor=2, downsample=None, nonlinearity=None,  strict=False, axis_off=True):# show = False,
    sig, sr = signal_array, samplig_rate 

    human_coch = cgram.human_cochleagram(sig, sr, n=n_subbands, sample_factor=sample_factor,
        downsample=downsample, nonlinearity=nonlinearity, strict=strict)
    img = np.flipud(human_coch)  # the cochleagram is upside down (i.e., in image coordinates)

    if axis_off:
        plt.axis('off')
    else:
        plt.title("Cochleagram")
        plt.xlabel("Time")
        plt.ylabel("Frequencies")

    # fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.imshow(img, aspect='auto', cmap=cmap, origin='lower', interpolation='nearest')
    plt.gca().invert_yaxis()

        # plt.show()
    # else:
    #     if axis_off:
    #         plt.axis('off')
    #     else:
    #         plt.title("Cochleagram")
    #         plt.xlabel("Time")
    #         plt.ylabel("Frequencies")


    if out_coch_path:
        plt.savefig(out_coch_path,bbox_inches='tight',
            pad_inches = 0)
        # plt.imsave(out_coch_path, img)
        # plt.figure.clear()
        plt.close()
        
    return img

def multi_channel_plot(signal_array, samplig_rate, out_coch_path = None, cmap='magma', n_subbands = 38, sample_factor=2, downsample=None, nonlinearity=None,  strict=False, axis_off=True):
    pass
