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
    unusable = np.array(([1,2,3,4,5,100000,5,4,3,1]+[0]*100+[1,2,3,4,5,100000,5,4,3,2,1]+[0]*50)*50)
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
    peaks_nothing = ecg.get_r_peaks(filt_nothing)
    peaks_chicken = ecg.get_r_peaks(filt_chicken)
    peaks_peaker = ecg.get_r_peaks(filt_peaker)
    peaks_cut = ecg.get_r_peaks(filt_chicken_cut)
    assert len(peaks_peaker) == 50
    assert len(peaks_nothing) > 0
    assert len(peaks_chicken) > 0
    assert len(peaks_chicken) == len(peaks_cut)
    
def test_segmenter(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    chicken_wave = ecg.segmenter(chicken)
    chicken_beats = ecg.segmenter(chicken, returns = 'beats')
    peaker_wave = ecg.segmenter(peaker)
    assert len(chicken_beats) < len(chicken)
    assert len(chicken_beats[0]) == len(chicken_beats[1])
    assert len(chicken_wave) < len(chicken)
    assert len(chicken_wave[0]) == len(chicken_wave[1])
    assert len(peaker_wave[0]) == len(peaker_wave[0])
    with pytest.raises(ValueError) as excinfo:
        ecg.segmenter(peaker, returns = 'bad')
    assert excinfo.value.args[0] == 'returns must either be avg or beats'
    chicken_cut = chicken[:4000]
    assert len(ecg.segmenter(chicken_cut)) < len(chicken_cut)
    
def test_get_bpm(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    chicken_bpm = ecg.get_bpm(chicken)
    #nothing_bpm = ecg.get_bpm(nothing)
    peaker_bpm = ecg.get_bpm(peaker)
    assert round(chicken_bpm) == 80.0
    #assert nothing_bpm > 30 and nothing_bpm < 200
    assert peaker_bpm > 110
    
    
def test_rythmRegularity(set_up_filter):
    nothing, chicken, peaker = set_up_filter
    chicken_regularity = ecg.rythmRegularity(chicken)
    #nothing_regularity = ecg.rythmRegularity(nothing)
    peaker_regularity = ecg.rythmRegularity(peaker)
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
    with pytest.raises(ValueError) as excinfo:
        ecg.interval(nothing, 'st')
    assert excinfo.value.args[0] == 'mode must be \'pr\' or \'rt\''
    pr = ecg.interval(chicken, 'pr')
    rt = ecg.interval(chicken, 'rt')
    assert pr > .05
    assert rt > .1
    rt = ecg.interval(invert, 'rt')
    assert rt > 0
    rt = ecg.interval(nothing, 'rt')
    assert rt is None
    
    
    