import pandas as pd
import numpy as np
from .filter import SignalFilter
from . import PeakDetector

def median_mad(df, axis=0):
    """
    Compute along axis the median and the med.
    Note: median is already included in pandas (df.median()) but not the mad
    This take care of constructing a Series for the mad.
    
    Arguments
    ----------------
    df : pandas.DataFrame
    
    
    Returns
    -----------
    med: pandas.Series
    mad: pandas.Series
    
    
    """
    med = df.median(axis=axis)
    mad = np.median(np.abs(df-med),axis=axis)*1.4826
    mad = pd.Series(mad, index = med.index)
    return med, mad

# this function enables to get the data frame in one function. 
def get_data_frame(x, t_start, sampling_rate, ch_names):
    """
    get data_frame compatible with the pandas representation of data. 
    
    Syntax
    
    signals = get_data_frame(x, t_start, sampling_rate)

    Input
    
    x: (nObs, nDim) ndarray
    t_start: time of start in seconds
    sampling_rate: sampling frequency in Hz 
    
    Output
    
    signals: a data_frame
    
    Example
    
    >>> np.random.seed(0)
    >>> x = np.random.randn(1000, 2)
    >>> t_start = 0
    >>> sampling_rate = 100
    >>> signals = get_data_frame(x, t_start, sampling_rate, np.arange(2))
    >>> times = signals.index.to_native_types()
    >>> print(signals[0:0.1])
                 0         1
    0.00  1.764052  0.400157
    0.01  0.978738  2.240893
    0.02  1.867558 -0.977278
    0.03  0.950088 -0.151357
    0.04 -0.103219  0.410599
    0.05  0.144044  1.454274
    0.06  0.761038  0.121675
    0.07  0.443863  0.333674
    0.08  1.494079 -0.205158
    0.09  0.313068 -0.854096
    0.10 -2.552990  0.653619
         
    """
    times = np.arange(x.shape[0], dtype = 'float64') / sampling_rate + t_start
    signals = pd.DataFrame(x, index = times, columns = ch_names)
    return signals

def filter_signal(signals, highpass_freq=300./30000., box_smooth=5): 
    """
    filter the signals with a highpass
    """
    h =  SignalFilter(signals, highpass_freq=highpass_freq, 
        box_smooth=box_smooth)
    filtered_sigs = h.get_filtered_data()
    return filtered_sigs
    
def find_peak(x, list_threshold): 
    """
    find_peak: get different values in function of the threshold
    
    """
    peakdetector = PeakDetector(x)
    n_peak = []
    for thresh in list_threshold: 
        peaks_pos = peakdetector.detect_peaks(threshold=thresh, peak_sign='-', 
            n_span=15)
        peaks_index = x.index[peaks_pos]
        n_peak.append(peaks_index.size)
    return n_peak

    