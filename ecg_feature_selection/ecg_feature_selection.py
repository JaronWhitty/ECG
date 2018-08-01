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

def filter_ecg(signal, normalized_frequency = .6, Q = 30, baseline_width = 101, 
               baseline_order = 3, baseline_freq_low = .1, baseline_freq_high = 1, fs = 200, butter_order = 2,
               points = 11, num_peak_points = 5, preserve_peak = False):
    """
    Filter and detrend a raw ECG signal.
    
    The ecg signal from the remote is extremelly noisy, in many cases the naked eye can't pick
    out the waveform from the raw signal. As such, we go through multiple filtering processes to 
    ge the the signal into a workable form. First we use a notch filter to filter out a specific
    noise frequency. We then remove the baseline wander using a savitsky golay filter to get the 
    baseline, and then subtract that off of the signal. We then use a wiener filter to reduce the 
    noise some more, and then use convolution with a bartlett window to smooth the signal further.
    Due to all the processing the resulting signal has a much smaller amplitude than it started with,
    if we wish to preserve the amplitude of the r-peaks we keep the amplitude from the point after we 
    get rid of the baseline wander.
    
    Args:
        signal (numpy array): The raw ECG data 
        normalized_frequency (float): the normalized frequency we wish to filter out, must be between 0 and 1, with 1 being half the sampling frequency. Default .6
        Q (int): Quality factor for the notch filter. Default 30
        baseline_width (int): How wide a window to use for the baseline removal. Default 101
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
    baseline = sig.savgol_filter(filt_signal, baseline_width, baseline_order, axis = 0)
    detrended_signal = filt_signal - baseline
    #using a zero phase iir filter based off a butterworth of order 2 cutting off low frequencies
    #Other Option (Use a much higher baseline_width, 1301 perhaps)
    #nyquist = fs / 2
    #bb, ba = sig.iirfilter(butter_order, [baseline_freq_low / nyquist, baseline_freq_high / nyquist])
    #trend = sig.filtfilt(bb, ba, filt_signal, axis = 0)
    #center trendline onto signal
    #together = np.median(trend) - np.median(filt_signal)
    #trend_center = trend - together
    #baseline_removed = filt_signal - trend_center
    #trend2 = sig.savgol_filter(baseline_removed, baseline_width, baseline_order)
    #baseline_removed = baseline_removed - trend2 
    #wiener filter
    filt_signal = sig.wiener(detrended_signal)
    #filt_signal = sig.wiener(baseline_removed)
    #smooth signal some more for cleaner average heartbeat
    bart = np.bartlett(points)
    smooth_signal = np.convolve(bart/bart.sum(), filt_signal, mode = 'same')
    
    #preserve the r-peak amplitude (if desired)
    #When smoothing the curve with np.convolve, we destroy amplitude in the r-peaks. We use the detrended signal's 
    #peak amplitude to preserve the r-peak amplitude to stay consistent with a normal ECG waveform. 
    if preserve_peak:
        r_peaks = get_r_peaks(signal, detrended_signal)
        #r_peaks = get_r_peaks(baseline_removed)
        for peak in r_peaks:
            for i in range(num_peak_points):
                #smooth_signal[peak + i] = detrended_signal[peak + i]
                #smooth_signal[peak - i] = detrended_signal[peak  - i]
                if peak + i > len(detrended_signal)-1 or peak - i < 0:
                    continue
                else:
                    smooth_signal[peak + i] = detrended_signal[peak + i]
                    smooth_signal[peak - i] = detrended_signal[peak - i]
                
    
    
    return smooth_signal

def get_r_peaks(og_signal, signal, exp = 3, peak_order = 80, high_cut_off = .8, low_cut_off = .5, med_perc = .55, 
                too_noisy = 1.6, noise_level = 5000, noise_points = 10, r_peak_points = 10):
    """Get the r peaks from a filtered de-trended ecg signal.
    
    All feature extraction methods for the ECG are dependant on the r-peaks, as these are 
    by far the mose well-defined feature of the ECG waveform. Our process for detecting the
    r-peaks is as follows:
        
    1) Take the gradient of the signal.
    
    2) Use peak detect algorithm to find the local minimums of the gradient.
    
    3) For each of these local minimums find the maximum in that area of the signal.
    
    4) Get rid of peaks that are unusually high or low amplitude. 
    
    5) Get rid of peaks that were the result of noise when the user is not touching the electrodes
    This is done by checking if there are any points around the peak that are beyond a certain specified
    noise level.
    
    6) Get rid of t peaks that may have been detected by checking to see if there are any peaks that 
    are in between two other peaks of the median distance between peaks.
    
    7) Get rid of peaks that have unusally high standard deviations around it, this both gets rid
    of potential peaks from noise that was not eliminated before, but also gets rid of extra noisy
    peaks that would throw off other feature extraction methods. 
    
    8) Return the peaks that survived all the rules.
    
    Args:
        og_signal (numpy array): The raw unfiltered signal, to be used in noise detection
        signal (numpy array): The filtered signal from which to find the r-peaks
        exp (int): exponent that we take the signal data to, find peaks easier. Default 2
        peak_order (int): number of data points on each side to compare when finding peaks. Default 80
        high_cut_off (float): percent above the median r-peak amplitude that constitues an invalid r-peak. Default .8
        low_cut_off (float): percent below the median r-peak amplitude that constitutes an invalid r-peak. Dfeault .5
        med_perc (float): percent of the median time one peak back and one peak forward that would surely not be an r peak. Defualt = .55
        too_noisy (float): How many times the median standard deviation around an R peak that flags noise instead of acutal heart beat. Default 1.6
        noise_level (float): Number above which we would consider noise from the original signal. Default 5000
        noise_points (int): Number of points on each side of the peaks to check for the noise level. Default 10
        r_peak_points (int): Number of points from the derivative critical point to look for the acutal r peak. Default 10
    Returns:
        numpy array: The indexes of the detected r-peaks
    """
    #exentuate the r peaks
    #r_finder = signal**exp
    #peaks = sig.argrelextrema(r_finder, np.greater, order = peak_order, mode = 'wrap')
    #convert peaks to 1D numpy array
    #peaks = peaks[0]
    #use derivative and find mins to find general location of r peaks
    deriv = np.gradient(signal)
    peak_areas = sig.argrelmin(deriv, order = peak_order)
    peak_areas = peak_areas[0]
    #now find the maximum around each peak area
    peaks = []
    for area in peak_areas:
        try:
            peaks.append(list(signal).index(max(signal[area - r_peak_points:area])))
        except ValueError: #if the area is right at the beginning 
            peaks.append(list(signal).index(max(signal[:area])))
    peaks = np.array(peaks)   
    #when user is not touching the electrodes correctly, the sensor gives very high amplitude spikes, we ignore these
    #ocassionaly there are higher amplitude t-waves then normal. These are still shorter amplitude to the r-peaks. We ignore these as well
    median = np.median(signal[peaks])
    valid = []
    for i in range(len(peaks)):
        if abs(signal[peaks[i]]) <= abs(median + median * high_cut_off) and abs(signal[peaks[i]]) >= abs(median - median * low_cut_off):
            valid.append(i)
    peaks = peaks[valid]        
    #often times noise is filtered down to around the same level as r peaks and the standard deviation filter isn't good enough
    #To cover these cases we look at the original unfiltered signal to take out peaks that are noise
    valid = []
    for i in range(len(peaks)):
        if not any(og_signal[peaks[i] - noise_points: peaks[i] + noise_points] > noise_level):
            valid.append(i)
    peaks = peaks[valid]
    #when the signal is all noise this will get rid of all the peaks
    if len(peaks) == 0:
        return peaks
    #some t-waves are still caught in r-peak detection to filter those out look at the distance between peaks
    #we look at the distances from one peak back to one peak forward, thus to single out t peaks
    dist = []
    for i in range(1, len(peaks) - 1):
        dist.append(peaks[i+1] - peaks[i-1])
    median = np.median(dist)
    #from the way we look at the distance we skipped the first and last, so add them back in
    not_t = [0]
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

def segmenter(signal, r_peaks, fs = 200, r_peak_split = .60, returns = 'avg', too_long = 1.7, too_short = .3):
    """Get the average ECG heartbeat in waveform.
    
    Even after the filtering process, the singal is still not clean enough to detect other features of 
    the ECG waveform, such as the p,q,s, and t points. To get a clean waveform, we first split up the signal
    into individual beats. This is done by finding the average distance between r peaks to use as the length
    of the the averaged beat. We then use the r_peak_split variable to decide what percent of the beat should
    be after the r-peak, Defaulted to .60. Going through each r-peak we take the calculated number of points
    before and after the r-peak and put it into an array of beats. Once we have all the beats we take the average
    along each column, resulting in the averaged beat. 
    
    Args:
        signal (numpy array): The filtered ECG signal
        r_peaks (numpy array): The r peaks of the ECG signal
        fs (int): The sampling frequency. Default 200 Hz 
        r_peak_split (float): proportion of heart beat we wish to show after the r-peak. Default .6
        returns (str): specifies what should be returned. if 'avg' then returns the average beat along with the domain in seconds. If 'beats' then returns the segemented beats and domain. Default 'avg'
        too_long(float): specifies number of seconds that would be too long a distance between peaks. Default 1.7 (30 bpm)
        too_short(float): sepcifies number of seconds that would be too short a distance between peaks. Default .3 (200 bpm)
        
    Returns:
            2-element tuple containing domain and specified wave, or None
        
            - **domain_avg** (*numpy array*): The time domain (in seconds) for the average heartbeat
            - **avg_beat** (*numpy array*): The average heart beat in wave form
            - **domain_beats** (*numpy array*): A list of the segmented beats (only returned when returns = 'beats'). 
            - **full_beats** (*numpy array*): the domain for the full_beast (in seconds) (only returned when returns = 'beats')
            - **None**: Returns None when there are no r-peaks detected
            
    Raises:
        ValueError: When returns is specified as somthing other than 'avg' or 'beats'.
    """
    if returns not in ['avg', 'beats']:
        raise ValueError('returns must either be avg or beats')
    #smooth signal more for a cleaner more viewable waveform 
    signal = filter_ecg(signal, preserve_peak = False)
    #find the average distance between r peaks (in counts)
    r_distance = []
    if len(r_peaks) != 0:
        last = r_peaks[0]
    else: 
        return None
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
    
def get_bpm(peaks, fs = 200, too_long = 1.7, too_short = .3, skip_factor = 1.8):
    """Get bpm from the r-peaks of the ECG signal.
    
    To get bpm, we use the r-peaks. First we find the time deltas in between each peak, filtering out
    those that are less than the time specified by too_short or longer than the time specified by too_long. 
    We then filter out any time deltas that where due to a skipped r-peak by taking out time deltas that were longer 
    than the median by a factor of the specified skip_factor. Once we have the time deltas we take the median of them,
    and convert that to the time domain and convert that to bpm.
    
    Args:
        peaks (numpy array): The raw ECG signal 
        fs (int): The sampling frequency. Default 200 Hz
        too_long (float): specifies number of seconds that would be too long a distance between peaks. Default 1.7 (35 bpm)
        too_short (float): sepcifies number of seconds that would be too short a distance between peaks. Default .3 (200 bpm)
        skip_factor (float): How many times greater than the median distance between peaks we would classify as a skipped peak. Default 1.8
    Returns:
        float: The median bpm from the signal
    """
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
        if dist < skip_factor*med_diff:
            non_skip.append(dist)
    r_distance = non_skip
  
    med_distance = np.median(r_distance)
    med_distance_sec = med_distance / fs
    #convert to bpm
    bpm = (1/med_distance_sec)*60
    
    return bpm

def rythmRegularity(peaks, fs = 200, too_long = 1.7, too_short = .3, skip_length = 1.8):
    """Gives a metric for the regularity of your heart rythm.
    
    Following the same process used to calculate bpm we get the valid time deltas between r-peaks.
    We then take the standard deviation and the difference between the max time delta and the min time delta.
    
    Args:
        peaks (numpy array): the r-peaks from the ECG signal
        fs (int): The sampling frequency. Default 200 Hz
        too_long (float): specifies number of seconds that would be too long a distance between peaks. Default 1.7 (35 bpm)
        too_short (float): sepcifies number of seconds that would be too short a distance between peaks. Default .3 (200 bpm)
        skip_lengt (float): specifies how many times longer than a normal interval would classify as a skipped peak. Default 1.8
    Returns:
            2 element tuple containing
            
            - **std** (*float*): the standard deviation (in seconds) of the r-r intervals
            - **max_dif** (*float*): the largerst difference (in seconds) between r-r intervals
    """
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
        if dist < skip_length*med_diff:
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
    else:
        return None
    
    return std, max_dif

def interval(domain, beat, mode, perc_p = .10, perc_t = .20, ends_perc = .10, invert_thresh = 1/3, r_area = 20, 
             qs_thresh = 1, s_points_after_r = 5, p_points_before_r = 15, p_peak_points = 7):
    """Returns the time interval (in seconds) between the specified points. Either 'pr', 'rt', or 'qrs'. 
    
    Time deltas between features of the ecg waveform are commonly used to detect various heart problems.
    We use the averaged beat waveform from the segmenter function to pick out the ecg waveform features,
    namely the P, Q, R, S, and T points of the waveform. To get the r point we simply take the maximum. 
    For the Q and S points we look at the derivative of the waveform. Q and S points are the valleys on 
    either side of the R peak. Once the derivative gets below the parameter qs_thresh, we specify the point
    as Q if it was on the left of the R peak, and S, if it was on the right of the r-peak. To detect the P peak,
    we also use the derivative. We find the the max point of the derivative to the left of the r peak to find the
    area of the P peak, and then find the point closest to zero in that area. To get the T peak we simply look at the
    max to the right of the r peak as the T peak has a large amplitude. If however there is a minimum point 
    within the invert threshold we classify the T wave as inverted, and instead look for the minimum.
    After the appropriate points are detected, we return the time delta in between the specified interval.
    
    Args:
        domain (numpy array): The time domain (in seconds) of the average beat
        beat (numpy array): The average beat in waveform
        mode (str): Specifies which interval, either 'pr' or 'rt'
        perc_p (int): percent of points before the r peak to start looking for the p peak. Default .10
        perc_t (int): percent of points after the r peak to start looking for the t peak. Default .20
        ends_perc (float): Percent of data to cut off from the ends when looking for peaks. Default .10
        invert_thresh (int): What percent of the r peak to be used as a threshold for an inverted t wave. Default 1/3
        r_area (int): Number of points on each side of the r peak to looking for q and s points. Default 20
        qs_thresh (float): When the derivative gets below this threshold we identify the q or s point. Default 1
        s_points_after_r (int): Number of points after the r peak to start looking for the s point. Default 5
        p_points_before_r (int): Number of points before the r peak to be start looking for the p wave. Default 15
        p_peak_points (int): Number of points after the derivative peak to look for the actual p peak. Default 7
    Returns:
        float: Time (in secons) Between the specified peaks (pr, qrs, or rt).
        
    Raises:
        ValueError: When the mode parameter is not specified as 'pr', 'qrs', or 'rt'.
    """
    if mode not in ['pr', 'rt', 'qrs']:
        raise ValueError('mode must be \'pr\', \'qrs\', or \'rt\'')
    if beat is None:
        return None
    ends_cut = int(ends_perc * len(beat))
    #points_p = int(perc_p * len(beat))
    points_t = int(perc_t * len(beat))
    r = list(beat).index(max(beat))
    #p = list(beat).index(max(beat[ends_cut:r - points_p]))
    #to get q, p and s points we look at the derivative
    #initialize q and s
    q = r - r_area
    s = r + r_area
    deriv = np.gradient(beat)
    for i in np.arange(r - r_area, r)[::-1]:
        if deriv[i] < qs_thresh:
            q = i
            break
    for i in range(r + s_points_after_r, r + r_area):
        if deriv[i] > -qs_thresh:
            s = i
            break
    area = list(deriv).index(np.max(deriv[:r - p_points_before_r]))
    p = list(deriv).index(min(deriv[area:area + p_peak_points], key = abs))
    #q = list(beat).index(min(beat[r-10:r]))
    #s = list(beat).index(min(beat[r:r+20]))
    #in some cases there can be an inverted t wave
    if min(beat[r + points_t:]) < np.median(beat[r + points_t:]) -invert_thresh*max(beat):
       t = list(beat).index(min(beat[r+ points_t:len(beat) - ends_cut]))
    else:
        t = list(beat).index(max(beat[r+ points_t:len(beat) - ends_cut]))
    if mode == 'pr':
        return domain[r] - domain[p]
    elif mode == 'rt':
        return domain[t] - domain[r]
    else:
        return domain[s] - domain[q]
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