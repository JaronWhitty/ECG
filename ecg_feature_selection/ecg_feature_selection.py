import scipy.signal as sig
import numpy as np
#import pywt

def usable(signal):
    if rythmRegularity(signal)[0] > .2:
        return False
    else:
        return True


def filter_ecg(signal, cut_off_frequency = .6, Q = 30, baseline_width = 101, 
               baseline_order = 3, points = 11, num_peak_points = 5,
               drop_first = True, preserve_peak = True):
    """
    filter and detrend a raw ECG signal 
    
    Args:
        signal (numpy array): The raw ECG data 
        cut_off_frequency (float): the frequency we wish to filter out, must be between 0 and 1, 
        with 1 being half the sampling frequency. Default .6
        Q (int): Quality factor for the notch filter. Default 30
        baseline_width (int): How wide a window to use for the baseline removal. Default 101
        baseline_order (int): Polynomial degree to use for the baseline removal. Default 3
        
    Returns:
        detrended_signal (numpy array): The filtered and detrended ECG signal
    """
    #filter out some specific frequency noise
    b, a = sig.iirnotch(cut_off_frequency, Q)
    filt_signal = sig.filtfilt(b, a, signal, axis = 0)
    #remove baseline wander
    baseline = sig.savgol_filter(filt_signal, baseline_width, baseline_order, axis = 0)
    detrended_signal = filt_signal - baseline
    #wiener filter
    filt_signal = sig.wiener(detrended_signal)
    #smooth signal some more for cleaner average heartbeat
    bart = np.bartlett(points)
    smooth_signal = np.convolve(bart/bart.sum(), filt_signal, mode = 'same')
    
    #preserve the r-peak amplitude (if desired)
    if preserve_peak:
        r_peaks = get_r_peaks(detrended_signal)
        #get rid of first peak which algorithm incorectly grabs (if desired)
        if drop_first:
            r_peaks = r_peaks[1:]
    
        for peak in r_peaks:
            for i in range(num_peak_points):
                try:
                    smooth_signal[peak + i] = detrended_signal[peak + i]
                    smooth_signal[peak - i] = detrended_signal[peak - i]
                except IndexError:
                    continue
    
    
    return smooth_signal

def get_r_peaks(signal, exp = 3, peak_order = 80, high_cut_off = .8, low_cut_off = .5):
    """
    get the r peaks from a filtered de-trended ecg signal 
    
    Args:
        signal (numpy array): The signal from which to find the r-peaks
        exp (int): exponent that we take the signal data to, find peaks easier
        peak_order (int): number of data points on each side to compare when finding peaks
        high_cut_off (float): percent above the median r-peak amplitude that constitues an invalid r-peak
        low_cut_off (float): percent below the median r-peak amplitude that constitutes an invalid r-peak
        
    Returns:
        peaks (numpy array): The indexes of the detected r-peaks
    """
    #exentuate the r peaks
    r_finder = signal**exp
    peaks = sig.argrelextrema(r_finder, np.greater, order = peak_order, mode = 'wrap')
    #convert peaks to 1D numpy array
    peaks = peaks[0]
    #throw out bogus peaks due to not good enough signal (bogus peaks have very high amplitude)
    median = np.median(signal[peaks])
    valid = []
    for i in range(len(peaks)):
        if signal[peaks[i]] <= median + median * high_cut_off and signal[peaks[i]] >= median - median * low_cut_off:
            valid.append(i)
    
    return peaks[valid]

def segmenter(signal, fs = 200, r_peak_split = .60, returns = 'avg' ):
    """
    get the average ECG heartbeat in waveform
    
    Args:
        signal (numpy array): The raw ECG signal
        fs (int): The sampling frequency. Default 200 Hz 
        r_peak_split (float): proportion of heart beat we wish to show after the r-peak. Default .6
        returns (str): specifies what should be returned. if 'avg' then returns the average beat
        along with the domain in seconds. If 'beats' then returns the segemented beats and domain. Default 'avg'
        
    Returns:
        domain_avg (numpy array): The time domain (in seconds) for the average heartbeat
        avg_beat (numpy array): The average heart beat in wave form
        full_beats (numpy array): A list of the segmented beats (only returned when returns = 'beats'). 
        domain_beats (numpy array): the domain for the full_beast (in seconds)
    """
    if returns not in ['avg', 'beats']:
        raise ValueError('returns must either be avg or beats')
        
    signal = filter_ecg(signal)
    signal = filter_ecg(signal, drop_first = False, preserve_peak = False)
    r_peaks = get_r_peaks(signal)
    #find the average distance between r peaks (in counts)
    r_distance = []
    last = r_peaks[0]
    for i in range(1, len(r_peaks)):
        if (r_peaks[i] - last) <= 2*fs and (r_peaks[i] - last) >= .3*fs: #make sure not bogus distance from cut off data
            r_distance.append(r_peaks[i] - last)
        last = r_peaks[i]
        
    avg_distance = int(np.mean(r_distance))
    #calculate how many counts to go forward and backward from each r peak
    forward = int(avg_distance*r_peak_split)
    backward = avg_distance - forward
    #seperate signal into individual beats
    beats = []
    for peak in r_peaks:
        #cut off first beat if not a full wave form
        if peak - backward < 0:
            continue
        #cuts off last beat if not a full wave form
        elif peak + forward > len(signal):
            continue
        beat = signal[peak - backward:peak + forward]
        #check to make sure it's a full heart beat
        if len(beat) == avg_distance:
            beats.append(beat)
    #create an average singal beat in waveform
    avg_beat = np.array(beats)
    avg_beat = avg_beat.mean(axis = 0)
    domain_avg = np.linspace(0, len(avg_beat)/fs, len(avg_beat))
    #create a full signal of all the valid beats
    full_beats = []
    for beat in beats:
        for value in beat:
            full_beats.append(value)
    domain_beats = np.linspace(0, len(full_beats)/fs, len(full_beats))
    
    if returns == 'avg':
        return domain_avg, avg_beat
    else:
        return domain_beats, full_beats
    
def get_bpm(signal, fs = 200):
    """
    get bpm from raw signal
    
    Args:
        signal (numpy array): The raw ECG signal 
        fs (int): The sampling frequency. Default 200 Hz
        
    Returns:
        bpm (float): The average bpm from the signal
    """
    #filter signal
    signal = filter_ecg(signal)
    #get r peaks
    peaks = get_r_peaks(signal)
    #calculate average time difference between peaks
    if len(peaks) != 0:
        last = peaks[0]
    r_distance = []
    for i in range(1, len(peaks)):
        if (peaks[i] - last) <= 2*fs and (peaks[i] - last) >= .3*fs: #make sure not bogus distance from cut off data
            r_distance.append(peaks[i] - last)
        last = peaks[i]
  
    avg_distance = np.mean(r_distance)
    avg_distance_sec = avg_distance / fs
    #convert to bpm
    bpm = (1/avg_distance_sec)*60
    
    return bpm

def rythmRegularity(signal, fs = 200):
    """
    gives a metric for the regularity of your heart rythm
    
    Args:
        signal (numpy array): the raw ECG signal
        fs (int): The sampling frequency. Default 200 Hz
        
    Returns:
        std (float): the standard deviation (in seconds) of the r-r intervals
        max_dif (float): the largerst difference (in seconds) between r-r intervals
    """
    
    #filter signal
    filt = filter_ecg(signal)
    peaks = get_r_peaks(filt)
    #get the distacnes of each r-r interval
    r_distance = []
    if len(peaks) != 0:
        last = peaks[0]
    for i in range(1, len(peaks)):
        if (peaks[i] - last) <= 2*fs and (peaks[i] - last) >= .3*fs: #make sure not bogus distance from cut off data
            r_distance.append(peaks[i] - last)
        last = peaks[i]
    
    if len(r_distance) != 0:
        #get standard deviation
        std = np.std(r_distance)
        # convert to time 
        std = std / fs
        #get difference between the longest and shortest r-r intervals 
        max_dif = np.max(r_distance) - np.min(r_distance)
        #convert to time
        max_dif = max_dif /fs
    
    return std, max_dif
    
#def wavelet(signal, C = 2, wavelet = 'db4', mode = 'soft'):
    #ca, cd = pywt.dwt(signal, wavelet, axis = 0)
    #cat = pywt.threshold(ca, (np.std(ca)*np.sqrt(C*np.log(len(signal)))), mode)
    #cdt = pywt.threshold(cd, (np.std(cd)*np.sqrt(C*np.log(len(signal)))), mode)
    #thresh = C*np.sqrt(np.std((signal/np.std(cd))*len(signal)))
    #cat = pywt.threshold(ca, thresh, mode)
    #cdt = pywt.threshold(cd, thresh, mode)
    #signal_rec = pywt.idwt(cat, cdt, wavelet, axis = 0)
    
    #return signal_rec

#def slope(signal):
    #slope = []
    #for i in range(len(signal)):
        #if i == 0:
            #slope.append(0)
            #continue
        #slope.append(signal[i] - signal[i -1])
        
    #slope = np.array(slope)
        
    #return slope
