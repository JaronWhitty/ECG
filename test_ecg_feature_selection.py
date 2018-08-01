#Unit Testing for ecg_feature_selection
import pytest
from ecg_feature_selection import ecg_feature_selection as ecg
import numpy as np
import pickle

with open ('test_data/testData.pkl', 'rb') as f:
    test_data = pickle.load(f)
    
@pytest.fixture
def set_up_usable():
    nothing = test_data['nothing']
    chicken = test_data['chicken'] 
    unusable = np.array((1,4000,1,4000,1,4000,1,4000,1,4000,1)*500)
    for i in range(0, len(unusable), 200):
        for j in range(0, 15):
            unusable[i-j] = 500 +2*j
            unusable[i+j] = 500 + 2*j
    for i in range(0, len(unusable), 70):
        for j in range(0, 15):
            unusable[i-j] = 500 +2*j
            unusable[i+j] = 500 + 2*j
    return nothing, chicken, unusable
    
    
def test_usable(set_up_usable):
    nothing, chicken, unusable = set_up_usable
    assert ecg.usable(nothing) == False
    assert ecg.usable(chicken) == True
    assert ecg.usable(unusable) == False


@pytest.fixture
def set_up_filter():
    nothing = test_data['nothing'] 
    chicken = test_data['chicken']
    peaker = np.array(([1,1000] + [1]*100)*50 + [1000])
    return nothing, chicken, peaker

def test_snr(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    nothing_snr = ecg.snr(nothing)
    chicken_snr = ecg.snr(chicken)
    peaker_snr = ecg.snr(peaker)
    assert chicken_snr < nothing_snr
    assert peaker_snr < nothing_snr
    
def test_filter_ecg(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    filt_nothing = ecg.filter_ecg(nothing)
    filt_chicken = ecg.filter_ecg(chicken)
    assert len(nothing) == len(filt_nothing)
    assert len(chicken) == len(filt_chicken)
    assert np.std(nothing) > 10*np.std(filt_nothing)
    assert np.std(chicken) < 10*np.std(filt_chicken)
    filt_chicken_preserve = ecg.filter_ecg(chicken, preserve_peak = True)
    assert np.std(filt_chicken_preserve) > np.std(filt_chicken)
    filt_peaker = ecg.filter_ecg(peaker)
    assert np.std(filt_peaker) < np.std(peaker)
    too_many_peak_points = ecg.filter_ecg(peaker, num_peak_points = 10000, preserve_peak = True)
    assert np.std(too_many_peak_points) < np.std(peaker)
    
    
def test_get_r_peaks(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    filt_nothing = ecg.filter_ecg(nothing)
    filt_chicken = ecg.filter_ecg(chicken)
    filt_peaker = ecg.filter_ecg(peaker)
    filt_chicken_cut = ecg.filter_ecg(chicken[:4000])
    peaks_nothing = ecg.get_r_peaks(nothing, filt_nothing)
    peaks_chicken = ecg.get_r_peaks(chicken, filt_chicken)
    peaks_peaker = ecg.get_r_peaks(peaker, filt_peaker)
    peaks_cut = ecg.get_r_peaks(chicken[:4000], filt_chicken_cut)
    noise = np.array([-10000, 10000] * 2500)
    assert len(ecg.get_r_peaks(noise, ecg.filter_ecg(noise))) == 0
    assert max(peaker[peaks_peaker] - np.median(peaker[peaks_peaker])) == 0
    assert len(peaks_nothing) == 0
    assert len(peaks_chicken) > 0
    assert len(peaks_chicken) == len(peaks_cut)
    
def test_segmenter(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    #filt_nothing = ecg.filter_ecg(nothing)
    filt_chicken = ecg.filter_ecg(chicken)
    filt_peaker = ecg.filter_ecg(peaker)
    #peaks_nothing = ecg.get_r_peaks(filt_nothing)
    peaks_chicken = ecg.get_r_peaks(chicken, filt_chicken)
    peaks_peaker = ecg.get_r_peaks(peaker, filt_peaker)
    chicken_wave = ecg.segmenter(filt_chicken, peaks_chicken)
    chicken_beats = ecg.segmenter(filt_chicken, peaks_chicken, returns = 'beats')
    peaker_wave = ecg.segmenter(filt_peaker, peaks_peaker)
    assert len(chicken_beats) < len(chicken)
    assert len(chicken_beats[0]) == len(chicken_beats[1])
    assert len(chicken_wave) < len(chicken)
    assert len(chicken_wave[0]) == len(chicken_wave[1])
    assert len(peaker_wave[0]) == len(peaker_wave[0])
    with pytest.raises(ValueError) as excinfo:
        ecg.segmenter(filt_peaker, peaks_peaker, returns = 'bad')
    assert excinfo.value.args[0] == 'returns must either be avg or beats'
    filt_chicken_cut = ecg.filter_ecg(chicken[:4000])
    peaks_chicken_cut = ecg.get_r_peaks(chicken[:4000], filt_chicken_cut)
    assert len(ecg.segmenter(filt_chicken_cut, peaks_chicken_cut)) < len(chicken[:4000])
    
def test_get_bpm(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    filt_chicken = ecg.filter_ecg(chicken)
    filt_peaker = ecg.filter_ecg(peaker)
    peaks_chicken = ecg.get_r_peaks(chicken, filt_chicken)
    peaks_peaker = ecg.get_r_peaks(peaker, filt_peaker)
    chicken_bpm = ecg.get_bpm(peaks_chicken)
    #nothing_bpm = ecg.get_bpm(nothing)
    peaker_bpm = ecg.get_bpm(peaks_peaker)
    assert round(chicken_bpm) == 80.0
    #assert nothing_bpm > 30 and nothing_bpm < 200
    assert peaker_bpm > 110
    
    
def test_rythmRegularity(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    filt_chicken = ecg.filter_ecg(chicken)
    filt_peaker = ecg.filter_ecg(peaker)
    peaks_chicken = ecg.get_r_peaks(chicken, filt_chicken)
    peaks_peaker = ecg.get_r_peaks(peaker, filt_peaker)
    chicken_regularity = ecg.rythmRegularity(peaks_chicken)
    #nothing_regularity = ecg.rythmRegularity(nothing)
    peaker_regularity = ecg.rythmRegularity(peaks_peaker)
    assert peaker_regularity[0] < 0.001
    assert peaker_regularity[1] < 0.01
    assert chicken_regularity[0] < .01
    assert chicken_regularity[1] < .1
    #assert nothing_regularity[0] > .2
    #assert nothing_regularity[1] > .25
    
@pytest.fixture
def set_up_interval():
    nothing = test_data['nothing']
    chicken = test_data['chicken']
    invert = np.array(([0]*50 + [250] + [0]*30 + [1500] + [0]*70 + [-1000] + [0] * 40)*20)
    not_invert = np.array(([0]*50 + [250] + [0]*30 + [1500] + [0]*70 + [-500] + [0] * 40)*20)
    return nothing, chicken, invert, not_invert
    
    
def test_interval(set_up_interval):
    nothing, chicken, invert, not_invert = set_up_interval
    filt_nothing = ecg.filter_ecg(nothing)
    filt_chicken = ecg.filter_ecg(nothing)
    filt_invert = ecg.filter_ecg(invert)
    #filt_not_invert = ecg.filter_ecg(not_invert)
    peaks_nothing = ecg.get_r_peaks(nothing, filt_nothing)
    peaks_chicken = ecg.get_r_peaks(chicken, filt_chicken)
    peaks_invert = ecg.get_r_peaks(invert, filt_invert)
    #peaks_not_invert = ecg.get_r_peaks(not_invert, filt_not_invert)
    nothing_wave = ecg.segmenter(filt_nothing, peaks_nothing)
    chicken_wave = ecg.segmenter(filt_chicken, peaks_chicken)
    invert_wave = ecg.segmenter(filt_invert, peaks_invert)
    #not_invert_wave = ecg.segmenter(filt_not_invert, peaks_not_invert)
    with pytest.raises(ValueError) as excinfo:
        ecg.interval(chicken_wave[0], chicken_wave[1], 'st')
    assert excinfo.value.args[0] == 'mode must be \'pr\', \'qrs\', or \'rt\''
    pr = ecg.interval(chicken_wave[0], chicken_wave[1], 'pr')
    rt = ecg.interval(chicken_wave[0], chicken_wave[1], 'rt')
    assert pr > .05
    assert rt > .1
    rt = ecg.interval(invert_wave[0], invert_wave[1], 'rt')
    assert rt > 0
    rt = ecg.interval(nothing_wave, nothing_wave, 'rt')
    assert rt is None
    assert ecg.interval(chicken_wave[0], chicken_wave[1], 'qrs') > .05
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    