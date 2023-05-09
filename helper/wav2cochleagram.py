from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from random import choice
from time import sleep, time
import numpy as np
from scipy.signal import welch, decimate

from pycochleagram import cochleagram as cgram
from pycochleagram import erbfilter as erb
from pycochleagram import utils

import matplotlib.pyplot as plt

def gen_cochleagram(wav_file_path, out_coch_path, n_subbands = 38, sample_factor=2, downsample=None, nonlinearity=None,  strict=False):
    signal, sr = utils.wav_to_array(wav_file_path)

    human_coch = cgram.human_cochleagram(signal, sr, n=n_subbands, sample_factor=sample_factor,
        downsample=downsample, nonlinearity=nonlinearity, strict=strict)
    img = np.flipud(human_coch)  # the cochleagram is upside down (i.e., in image coordinates)

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.imshow(img, aspect='auto', cmap='magma', origin='lower', interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()
    fig.savefig(out_coch_path,bbox_inches='tight',
        pad_inches = 0)

    return img


    """
    Args:

    default values:  n=None, low_lim=50, hi_lim=20000,
        sample_factor=2, padding_size=None, downsample=None, nonlinearity=None,
        fft_mode='auto', ret_mode='envs'

    out_format= 'img' or 'arrays'    

    n (int): Number of filters (subbands) to be generated with standard
        sampling (i.e., using a sampling factor of 1). Note, the actual number of
        filters in the generated filterbank depends on the sampling factor, and
        will also include lowpass and highpass filters that allow for
        perfect reconstruction of the input signal (the exact number of lowpass
        and highpass filters is determined by the sampling factor).

    low_lim (int): Lower limit of frequency range. Filters will not be defined
        below this limit.

    hi_lim (int): Upper limit of frequency range. Filters will not be defined
        above this limit.

    sample_factor (int): Positive integer that determines how densely ERB function
        will be sampled to create bandpass filters. 1 represents standard sampling;
        adjacent bandpass filters will overlap by 50%. 2 represents 2x overcomplete sampling;
        adjacent bandpass filters will overlap by 75%. 4 represents 4x overcomplete sampling;
        adjacent bandpass filters will overlap by 87.5%.

    padding_size (int, optional): If None (default), the signal will not be padded
        before filtering. Otherwise, the filters will be created assuming the
        waveform signal will be padded to length padding_size+signal_length.

    downsample (None, int, callable, optional): The `downsample` argument can
        be an integer representing the upsampling factor in polyphase resampling
        (with `sr` as the downsampling factor), a callable
        (to perform custom downsampling), or None to return the
        unmodified cochleagram; see `apply_envelope_downsample` for more
        information. If `ret_mode` is 'envs', this will be applied to the
        cochleagram before the nonlinearity, otherwise no downsampling will be
        performed. Providing a callable for custom downsampling is suggested.

    nonlinearity ({None, 'db', 'power', callable}, optional): The `nonlinearity`
        argument can be an predefined type, a callable
        (to apply a custom nonlinearity), or None to return the unmodified
        cochleagram; see `apply_envelope_nonlinearity` for more information.
        If `ret_mode` is 'envs', this will be applied to the cochleagram after
        downsampling, otherwise no nonlinearity will be applied. Providing a
        callable for applying a custom nonlinearity is suggested.

    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
        to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
        will fallback to numpy, if necessary.

    ret_mode ({'envs', 'subband', 'analytic', 'all'}): Determines what will be
        returned. 'envs' (default) returns the subband envelopes; 'subband'
        returns just the subbands, 'analytic' returns the analytic signal provided
        by the Hilber transform, 'all' returns all local variables created in this
        function.

    strict (bool, optional): If True (default), will throw an errors if this
        function is used in a way that is unsupported by the MATLAB implemenation.

    Returns:
    array:
    **out**: The output, depending on the value of `ret_mode`. If the `ret_mode`
        is 'envs' and a downsampling and/or nonlinearity
        operation was requested, the output will reflect these operations.
    """


    pass