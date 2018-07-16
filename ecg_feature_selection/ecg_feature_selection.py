from __future__ import division
import scipy.signal as sig
import numpy as np
#import pywt
def snr(signal, points = 200):
    """
    determines the signal to noise ratio of the ECG signal by looking at the standard deviation of the noise
    
    Args:
        signal (numpy array): the raw signal we wish to determine the signal to noise ratio of
        points (int): Number of points on each side of the middle we wish to look at to detrmine the snr
    Returns:
        noise_std (float): The standard deviation of the noise
        
    """
    filt = filter_ecg(signal)
    #take the middle section of the data to test (generally the beginning and end are when the user is not touching the electrodes)
    middle = len(filt)//2
    noise = signal - filt
    noise_std = np.std(noise[middle - points:middle + points])
    return noise_std

def usable(signal, fail_point = .1, snr_too_high = 250):
    """
    determines if the raw signal is usable to extract features
    
    Args:
        signal (numpy array): raw ECG signal we wish to determine the usability of
        fail_point (float): The standard deviation of the heart beat times, past which means algorithm didn't perform well enough. Default .1
        snr_too_high (int): Int above which we say the signal to noise ratio as defined in the snr function is too high
    Returns: 
        bool: True if usable, False if unusable
    """
    #We find the standard deviation of the distances between r-peaks. An unusable signal will still get peaks,
    #but these peaks will not be in any regualar form like an ECG signal's peaks would be
    try:
        if rythmRegularity(signal)[0] < fail_point or snr(signal) < snr_too_high :
            return True
        else:
            return False
    except: #for various reasons such as no peaks found it could be invalid data throwing an error 
        return False

def filter_ecg(signal, normalized_frequency = .6, Q = 30, baseline_width = 301, 
               baseline_order = 3, baseline_freq_low = .1, baseline_freq_high = 1, fs = 200, butter_order = 2,
               points = 11, num_peak_points = 5, preserve_peak = False):
    """
    filter and detrend a raw ECG signal 
    
    Args:
        signal (numpy array): The raw ECG data 
        normalized_frequency (float): the normalized frequency we wish to filter out, must be between 0 and 1, with 1 being half the sampling frequency. Default .6
        Q (int): Quality factor for the notch filter. Default 30
        baseline_width (int): How wide a window to use for the baseline removal. Default 301
        baseline_order (int): Polynomial degree to use for the baseline removal. Default 3
        baseline_freq_low (float): low end of frequency to cut off to eliminate baseline drift. Default .01 Hz
        baseline_freq_high (float): high end frequency to cut off to eliminate baseline drift. Default .1 Hz
        butter_order (int): The order of the butter filter used to eliminate baseline drift. Defualt 2
        points (int): The number of points to use for the bartlett window. Default 11
        num_peak_points (int): The number of points around each r-peak to keep at their original amplitude. Default 5
        
    Returns:
         numpy array: The filtered and detrended ECG signal
    """
    #filter out some specific frequency noise
    b, a = sig.iirnotch(normalized_frequency, Q)
    filt_signal = sig.filtfilt(b, a, signal, axis = 0)
    #remove baseline wander
    #baseline = sig.savgol_filter(filt_signal, baseline_width, baseline_order, axis = 0)
    #detrended_signal = filt_signal - baseline
    #using a zero phase iir filter based off a butterworth of order 2 cutting off low frequencies
    """ Other Option (Use a much higher baseline_width, 1301 perhaps) """
    nyquist = fs / 2
    bb, ba = sig.iirfilter(butter_order, [baseline_freq_low / nyquist, baseline_freq_high / nyquist])
    trend = sig.filtfilt(bb, ba, filt_signal, axis = 0)
    #center trendline onto signal
    together = np.median(trend) - np.median(filt_signal)
    trend_center = trend - together
    baseline_removed = filt_signal - trend_center
    trend2 = sig.savgol_filter(baseline_removed, baseline_width, baseline_order)
    baseline_removed = baseline_removed - trend2 
    #wiener filter
    #filt_signal = sig.wiener(detrended_signal)
    filt_signal = sig.wiener(baseline_removed)
    #smooth signal some more for cleaner average heartbeat
    bart = np.bartlett(points)
    smooth_signal = np.convolve(bart/bart.sum(), filt_signal, mode = 'same')
    
    #preserve the r-peak amplitude (if desired)
    #When smoothing the curve with np.convolve, we destroy amplitude in the r-peaks. We use the detrended signal's 
    #peak amplitude to preserve the r-peak amplitude to stay consistent with a normal ECG waveform. 
    if preserve_peak:
        #r_peaks = get_r_peaks(detrended_signal)
        r_peaks = get_r_peaks(baseline_removed)
        for peak in r_peaks:
            for i in range(num_peak_points):
                try:
                    #smooth_signal[peak + i] = detrended_signal[peak + i]
                    #smooth_signal[peak - i] = detrended_signal[peak - i]
                    smooth_signal[peak + i] = baseline_removed[peak + i]
                    smooth_signal[peak - i] = baseline_removed[peak - i]
                except IndexError:
                    continue
    
    
    return smooth_signal

def get_r_peaks(signal, exp = 3, peak_order = 50, high_cut_off = .8, low_cut_off = .5, med_perc = .55, too_noisy = 1.6, noise_level = 5000, noise_points = 10):
    """
    get the r peaks from a filtered de-trended ecg signal 
    
    Args:
        signal (numpy array): The signal from which to find the r-peaks
        exp (int): exponent that we take the signal data to, find peaks easier. Default 3
        peak_order (int): number of data points on each side to compare when finding peaks. Default 80
        high_cut_off (float): percent above the median r-peak amplitude that constitues an invalid r-peak. Default .8
        low_cut_off (float): percent below the median r-peak amplitude that constitutes an invalid r-peak. Dfeault .5
        med_perc (float): percent of the median time one peak back and one peak forward that would surely not be an r peak. Defualt = .55
        too_noisy (float): How many times the median standard deviation around an R peak that flags noise instead of acutal heart beat. Default 1.6
        noise_level (float): Number above which we would consider noise from the original signal. Default 1000
        noise_points (int): Number of points on each side of the peaks to check for the noise level. Default 10
    Returns:
        numpy array: The indexes of the detected r-peaks
    """
    #exentuate the r peaks
    og_signal = signal
    signal = filter_ecg(signal)
    r_finder = signal**exp
    peaks = sig.argrelextrema(r_finder, np.greater, order = peak_order, mode = 'wrap')
    #convert peaks to 1D numpy array
    peaks = peaks[0]
    #when user is not touching the electrodes correctly, the sensor gives very high amplitude spikes, we ignore these
    #ocassionaly there are higher amplitude t-waves then normal. These are still shorter amplitude to the r-peaks. We ignore these as well
    median = np.median(signal[peaks])
    valid = []
    for i in range(len(peaks)):
        if signal[peaks[i]] <= median + median * high_cut_off and signal[peaks[i]] >= median - median * low_cut_off:
            valid.append(i)
    peaks = peaks[valid]        
    #often times noise is filtered down to around the same level as r peaks and the standard deviation filter isn't good enough
    #To cover these cases we look at the original unfiltered signal to take out peaks that are noise
    valid = []
    for i in range(len(peaks)):
        if not any(og_signal[peaks[i] - noise_points: peaks[i] + noise_points] > noise_level):
            valid.append(i)
    peaks = peaks[valid]
    #some t-waves are still caught in r-peak detection to filter those out look at the distance between peaks
    #we look at the distances from one peak back to one peak forward, thus to single out t peaks
    dist = []
    for i in range(1, len(peaks) - 1):
        dist.append(peaks[i+1] - peaks[i-1])
    median = np.median(dist)
    #from the way we look at the distance we skipped the first and last, so add them back in
    if len(peaks) != 0:
        not_t = [0]
    else: 
        return peaks
    for i in range(len(dist)):
        if dist[i] > median*med_perc:
            not_t.append(i + 1)

    not_t.append(len(peaks) -1)
    #occasionally there happens to be noise at a similar amplitude and similar distances as r-peaks 
    #to get rid of these we can eliminate the detected peaks that have unusally high standard deviations around them
    peaks = peaks[not_t]
    not_noise = []
    #find the distance before and after each peak to look at
    dist = []
    last = peaks[0]
    for i in range(1, len(peaks)):
        dist.append(peaks[i] - last)
        last = peaks[i]
    med_distance = np.median(dist)
    look_distance = int(med_distance / 2)
    #get the standard deviation around each peak
    stds = []
    for peak in peaks:
        if peak - look_distance < 0:
            stds.append(np.std(signal[:look_distance]))
            continue
        else:
            stds.append(np.std(signal[peak - look_distance:peak + look_distance]))
            
    med_std = np.median(stds)
    #accept only the peaks with more normal standard deviation around it
    for i in range(len(stds)):
        if stds[i] < too_noisy* med_std and stds[i] > 1/too_noisy * med_std:
            not_noise.append(i)
            
    peaks = peaks[not_noise]
    
    return peaks

def segmenter(signal, fs = 200, r_peak_split = .60, returns = 'avg', too_long = 1.7, too_short = .3):
    """
    get the average ECG heartbeat in waveform
    
    Args:
        signal (numpy array): The raw ECG signal
        fs (int): The sampling frequency. Default 200 Hz 
        r_peak_split (float): proportion of heart beat we wish to show after the r-peak. Default .6
        returns (str): specifies what should be returned. if 'avg' then returns the average beat along with the domain in seconds. If 'beats' then returns the segemented beats and domain. Default 'avg'
        too_long(float): specifies number of seconds that would be too long a distance between peaks. Default 1.7 (30 bpm)
        too_short(float): sepcifies number of seconds that would be too short a distance between peaks. Default .3 (200 bpm)
        
    Returns:
            2-element tuple containing
        
            - **domain_avg** (*numpy array*): The time domain (in seconds) for the average heartbeat
            - **avg_beat** (*numpy array*): The average heart beat in wave form
            - **domain_beats** (*numpy array*): A list of the segmented beats (only returned when returns = 'beats'). 
            - **full_beats** (*numpy array*): the domain for the full_beast (in seconds) (only returned when returns = 'beats')
    """
    if returns not in ['avg', 'beats']:
        raise ValueError('returns must either be avg or beats')
    #split up between r-peaks
    r_peaks = get_r_peaks(signal)
    #filter the raw signal    
    signal = filter_ecg(signal)
    #smooth signal more for a cleaner more viewable waveform 
    signal = filter_ecg(signal, preserve_peak = False)
    #find the average distance between r peaks (in counts)
    r_distance = []
    last = r_peaks[0]
    for i in range(1, len(r_peaks)):
        if (r_peaks[i] - last) <= too_long*fs and (r_peaks[i] - last) >= too_short*fs: #make sure not bogus distance from cut off data
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
    
def get_bpm(signal, fs = 200, too_long = 1.7, too_short = .3):
    """
    get bpm from raw signal
    
    Args:
        signal (numpy array): The raw ECG signal 
        fs (int): The sampling frequency. Default 200 Hz
        too_long(float): specifies number of seconds that would be too long a distance between peaks. Default 1.7 (35 bpm)
        too_short(float): sepcifies number of seconds that would be too short a distance between peaks. Default .3 (200 bpm)
        
    Returns:
        float: The average bpm from the signal
    """
    #get r peaks
    peaks = get_r_peaks(signal)
    #filter signal
    signal = filter_ecg(signal)
    #calculate average time difference between peaks
    if len(peaks) != 0:
        last = peaks[0]
    r_distance = []
    for i in range(1, len(peaks)):
        if (peaks[i] - last) <= too_long*fs and (peaks[i] - last) >= too_short*fs: #make sure not bogus distance from cut off data
            r_distance.append(peaks[i] - last)
        last = peaks[i]
    #sometimes the algorithm skips peaks because they are too noisy
    #We eliminate those skips from the distances
    med_diff = np.median(r_distance)
    non_skip = []
    for dist in r_distance:
        if dist < 1.8*med_diff:
            non_skip.append(dist)
    r_distance = non_skip
  
    avg_distance = np.mean(r_distance)
    avg_distance_sec = avg_distance / fs
    #convert to bpm
    bpm = (1/avg_distance_sec)*60
    
    return bpm

def rythmRegularity(signal, fs = 200, too_long = 1.7, too_short = .3):
    """
    gives a metric for the regularity of your heart rythm
    
    Args:
        signal (numpy array): the raw ECG signal
        fs (int): The sampling frequency. Default 200 Hz
        too_long(float): specifies number of seconds that would be too long a distance between peaks. Default 1.7 (35 bpm)
        too_short(float): sepcifies number of seconds that would be too short a distance between peaks. Default .3 (200 bpm)
        
    Returns:
            2 element tuple containing
            
            - **std** (*float*): the standard deviation (in seconds) of the r-r intervals
            - **max_dif** (*float*): the largerst difference (in seconds) between r-r intervals
    """
    #split up between beats
    peaks = get_r_peaks(signal)
    #get the distacnes of each r-r interval
    r_distance = []
    if len(peaks) != 0:
        last = peaks[0]
    for i in range(1, len(peaks)):
        if (peaks[i] - last) <= too_long*fs and (peaks[i] - last) >= too_short*fs: #make sure not bogus distance from cut off data
            r_distance.append(peaks[i] - last)
        last = peaks[i]
    #sometimes the algorithm skips peaks because they are too noisy
    #We eliminate those skips from the distances
    med_diff = np.median(r_distance)
    non_skip = []
    for dist in r_distance:
        if dist < 1.8*med_diff:
            non_skip.append(dist)
    r_distance = non_skip
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

def interval(signal, mode, perc_p = .10, perc_t = .20, ends_perc = .10, invert_thresh = 1/3):
    """
    returns the time interval (in seconds) between the p and r peak
    
    Args:
        signal (numpy array): The raw ECG signal
        mode (str): Specifies which interval, either 'pr' or 'rt'
        perc_p (int): percent of points before the r peak to start looking for the p peak. Default .10
        perc_t (int): percent of points after the r peak to start looking for the t peak. Default .20
        ends_perc (float): Percent of data to cut off from the ends when looking for peaks. Default .10
        invert_thresh (int): What percent of the r peak to be used as a threshold for an inverted t wave. Default 1/3
    Returns:
        float: Time (in secons) Between the specified peaks (pr or rt). 
    """
    if mode not in ['pr', 'rt']:
        raise ValueError('mode must be \'pr\' or \'rt\'')
    domain, beat = segmenter(signal)
    ends_cut = int(ends_perc * len(beat))
    points_p = int(perc_p * len(beat))
    points_t = int(perc_t * len(beat))
    r = list(beat).index(max(beat))
    p = list(beat).index(max(beat[ends_cut:r - points_p]))
    #in some cases there can be an inverted t wave
    if min(beat[r + points_t:]) < np.median(beat[r + points_t:]) -invert_thresh*max(beat):
       t = list(beat).index(min(beat[r+ points_t:len(beat) - ends_cut]))
    else:
        t = list(beat).index(max(beat[r+ points_t:len(beat) - ends_cut]))
    if mode == 'pr':
        return domain[r] - domain[p]
    elif mode == 'rt':
        return domain[t] - domain[r]
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