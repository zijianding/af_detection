import numpy as np
import scipy.io as sio
import wfdb
#import os
import matplotlib.pyplot as plt
import pandas
#import pickle
from IPython.display import display
import itchat, time
from itchat.content import *


####### input signal #########
def read_signal(file_path, record_name):
	file = file_path + '/' + record_name
	signal = wfdb.rdsamp(recordname = file)
	hea = wfdb.rdheader(file)
	return signal, hea
 

######## detect beats ##############
#import sys
#import numpy as np
import scipy.signal
import scipy.ndimage
#import pickle


def detect_beats(
        ecg,  # The raw ECG signal
        rate,  # Sampling rate in HZ
        # Window size in seconds to use for
        ransac_window_size=5.0,
        # Low frequency of the band pass filter
        lowfreq=5.0,
        # High frequency of the band pass filter
        highfreq=15.0,
):
    """
    ECG heart beat detection based on
    http://link.springer.com/article/10.1007/s13239-011-0065-3/fulltext.html
    with some tweaks (mainly robust estimation of the rectified signal
    cutoff threshold).
    """

    ransac_window_size = int(ransac_window_size * rate)

    lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    # TODO: Could use an actual bandpass filter
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)

    # Square (=signal power) of the first difference of the signal
    decg = np.diff(ecg_band)
    decg_power = decg ** 2

    # Robust threshold and normalizator estimation
    thresholds = []
    max_powers = []
    for i in range(len(decg_power) / ransac_window_size):
        sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
        d = decg_power[sample]
        thresholds.append(0.5 * np.std(d))
        max_powers.append(np.max(d))

    threshold = np.median(thresholds)
    max_power = np.median(max_powers)
    decg_power[decg_power < threshold] = 0

    decg_power /= max_power
    decg_power[decg_power > 1.0] = 1.0
    square_decg_power = decg_power ** 2

    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0

    mean_window_len = int(rate * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
    # lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)

    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 8.0)
    lp_energy_diff = np.diff(lp_energy)

    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1
    return zero_crossings


def plot_peak_detection(ecg, rate):
    import matplotlib.pyplot as plt
    dt = 1.0 / rate
    t = np.linspace(0, len(ecg) * dt, len(ecg))
    #plt.plot(t, ecg)
    plt.plot(ecg)
    peak_i = detect_beats(ecg, rate)
    qrs_amp = [ecg[x] for x in peak_i]
    plt.plot(peak_i,qrs_amp,'ro')
    #plt.scatter(t[peak_i], ecg[peak_i], color='red')
    plt.show()

######## QRS detection #######                   

def indice2time(peak_indices, fs=300.):
    time_gap = 1. / float(fs)
    time_seq = np.zeros(len(peak_indices))
    for i in range(0, len(peak_indices)):
        time_seq[i] = peak_indices[i] * time_gap
    return time_seq
    

def qrs_detection(signal, min_bpm=20, max_bpm=230, smooth_window=150):
    #peak indices    
    d_sig = signal.adc()[:,0] #A/D convert necessary?
    peak_indices = wfdb.processing.gqrs_detect(x=d_sig, fs = signal.fs, adcgain=signal.adcgain[0], adczero=signal.adczero[0])
    
    #hr
    hrs = wfdb.processing.compute_hr(siglen=d_sig.shape[0], peak_indices=peak_indices, fs=signal.fs)
    
    #correct peaks
    min_gap = signal.fs * 60 / min_bpm
    max_gap = signal.fs * 60 / max_bpm
    peak_indices = wfdb.processing.correct_peaks(d_sig, peak_indices=peak_indices, min_gap=min_gap, 
                                                 max_gap = max_gap, smooth_window = smooth_window)
    
    peak_time = indice2time(peak_indices, fs = signal.fs)
                                                
    return peak_indices, peak_time
    
    

####### RR interval ##########
#input qrs file#
def read_qrs(path, name, suffice='_qrs.txt'):
    file = path + '/' + name + suffice
    qrs = pandas.read_table(file, sep='\s+', header=None)
    return qrs


def time2sec(qrs_time):
    '''
    convert [00:01:00]
    to 60
    '''
    qrs_time = qrs_time.strip('[')
    qrs_time = qrs_time.strip(']')
    x = qrs_time.split(':')
    y = np.zeros(len(x))
    for i in range(0,len(x)):
        y[i] = float(x[i])
    qrs_sec = y[0]*3600 + y[1]*60 + y[2]
    return qrs_sec
    
def qrs_convert(qrs):
    '''
    a series of [**:**:**] to seconds   
    '''
    x = qrs[0]
    qrs_time = np.zeros((len(x)))
    for i in range(0, len(x)):
        qrs_time[i] = time2sec(x[i])
    return qrs_time
    

def rr_interval( qrs, thresh = 32+2, section = 'sub' ):
    #x = qrs[0]
    x = qrs  
    #    
    if len(x) < thresh:
        print('The lengh of this record is too short, unreadbale!')
        return -1
    if section == 'sub':
        x = x[range(0,thresh)]
        # rr
        curr_rr = np.zeros(thresh-1)
        for i in range(1, thresh):
            curr_rr[i-1] = qrs_time[i] - qrs_time[i-1]
        return curr_rr
    elif section == 'all':
        # rr
        curr_rr = np.zeros(len(x)-1)
        for i in range(0, len(x)):
            curr_rr[i-1] = qrs_time[i] - qrs_time[i-1]
        return curr_rr

def drr_interval(rr):
    drr = np.zeros(len(rr)-1)
    for i in range(1, len(rr)):
        drr[i-1] = rr[i] - rr[i-1] 
    return drr

######## calculate NEC #######

def nec_calc(rr, drr, gap = 0.025):
    if ( len(rr) - len(drr) ) != 1:
        print('Length of rr and rr is not equal')
        return -1
    # same length 
    rr = np.delete(rr, 0, axis=0)
    
    xc = int( (np.max(rr) - np.min(rr) + gap*2) // gap + 1 )
    yc = int( (np.max(drr) - np.min(drr) + gap*2 ) // gap + 1 )

    xmin = np.min(rr) - gap
    ymin = np.min(drr) - gap

    mat = np.zeros((xc, yc))
    for i in range(0, xc):
        xleft = xmin + i * gap
        xright = xmin + (i+1) * gap
        new_rr_1 = (xleft <= rr)
        new_rr_2 = (rr < xright)
        new_rr = (new_rr_1 == new_rr_2)
        for j in range(0, yc):
            ydown = ymin + j * gap
            yup = ymin + (j+1) * gap
            new_drr_1 = (ydown <= drr)
            new_drr_2 = (drr < yup)
            new_drr = (new_drr_1 == new_drr_2)
            for k in range(0, len(new_rr)):
                if (new_rr[k] == True) & (new_drr[k] == True):
                    mat[i,j] += 1

    mat = (mat>0)
    nec = sum(sum(mat))
    normalized_nec = nec/float(len(rr))

    return nec, normalized_nec

####### classification #######

def classify( nec_norm, thresh, words = True ):
    prediction = 'non-AF'    
    if nec_norm > thresh:
        prediction = 'AF'
    quote = 'This is a ' + prediction + ' patient'
    if words == True:
        print( quote )
    
    return prediction, quote
    
    
def comparison(name, label, prediction):
    if (label == 'A') and (prediction == 'AF'):
        return 'TP'
    elif (label != 'A') and (prediction == 'AF'):
        return 'FP'
    elif (label == 'A') and (prediction != 'AF'):
        return 'FN'
    else:
        return 'TN'

def roc_curve(train_data, nec_dict):
    m = len(train_data)
    gap = 1 / float(m)

    res = np.zeros((m+1,4)) #tp, fp , tn, fn
    sensitivity = np.zeros((m+1))
    specificity = np.zeros((m+1))
    gaps = []
    for i in range(0, m+1):
        curr_thresh = i * gap
        gaps.append(curr_thresh)

        tp = 0.
        tn = 0.
        fp = 0.
        fn = 0.
        for j in range(0, len(train_data)):
            curr_sample = train_data[j][0]
            curr_label = train_data[j][1]
            pred = classify( nec_dict[curr_sample], curr_thresh, words = False)
            comp = comparison( curr_sample, curr_label, pred)
            if comp == 'TP':
                tp += 1.
            elif comp == 'TN':
                tn += 1.
            elif comp == 'FP':
                fp += 1.
            elif comp == 'FN':
                fn += 1.

        res[i, 0] = tp
        res[i, 1] = fp
        res[i, 2] = tn
        res[i, 3] = fn

        sensitivity[i] = tp / (tp + fn)
        specificity[i] = tn / (fp + tn)

    return res, sensitivity, specificity,  gaps



######## draw ECG ############
def plot_ecg(signal, hea, text):
    wfdb.plotrec(signal, title=text)
    display(signal.__dict__)
    display(hea.__dict__)


def plot_save_ecg(ecg_sig, save_path, png_name, fs = 300,mm_per_mv = 10.0, mm_per_s = 25.0, max_mv = 2.5, min_mv = -2.5):
    '''
    ecg_sig: ecg signal
    fs: ecg sample rate
    save_path: ecg file save path
    '''
    subplot_RowNum = int( len(ecg_sig) / (fs * 10.) )
    if len(ecg_sig) % (fs * 10.) > 0:
        subplot_RowNum += 1
        
        
    pixel_per_mm = np.int(np.floor(1 / mm_per_s * fs))
    max_point_time_count = int(mm_per_s * 10 * pixel_per_mm)
    max_point_amp_count = int(max_mv * mm_per_mv * pixel_per_mm)
    min_point_amp_count = int(min_mv * mm_per_mv * pixel_per_mm)
    
    if subplot_RowNum == 1:
        plt.figure(1,figsize=(75,15),dpi=96)
        plt.clf()

        plt.plot(ecg_sig * 10 * pixel_per_mm, color='k', label='ecg', linewidth=2, alpha=0.9)
        for xx in range(0, max_point_time_count+pixel_per_mm, pixel_per_mm):
            if xx % (pixel_per_mm*5) == 0:
                plt.plot([xx, xx], [min_point_amp_count,max_point_amp_count], color='r', linestyle='-', linewidth=2,alpha=0.6)
            else:
                plt.plot([xx, xx], [min_point_amp_count,max_point_amp_count], color='r', linestyle='-', linewidth=1,alpha=0.4)

        for yy in range(min_point_amp_count, max_point_amp_count,pixel_per_mm):
            if yy % (5*pixel_per_mm) == 0 and yy != 0:
                plt.plot([0, max_point_time_count], [yy, yy], color='r', linestyle='-', linewidth=2,alpha=0.6)
            else:
                if yy != 0:
                    plt.plot([0, max_point_time_count], [yy, yy], color='r', linestyle='-', linewidth=1,alpha=0.4)
                else:
                    plt.plot([0, max_point_time_count], [0, 0], color='r', linestyle='-', linewidth=2,alpha=0.6)
        plt.xlabel('time/s',fontsize=40,horizontalalignment='right')
        plt.yticks(fontsize=25)
        plt.ylabel('amp/mv',fontsize=25,verticalalignment='top',rotation='vertical')
        plt.xticks(range(0, max_point_time_count, int( mm_per_s * pixel_per_mm)), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=40)
        plt.yticks(range(min_point_amp_count, max_point_amp_count, int(0.5 * mm_per_mv * pixel_per_mm)), [-2.5, -2.0, -1.5, -1.0, 0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=40)
        plt.axis('equal')
        
        plt.savefig(save_path + '/' + png_name)  
        
    elif subplot_RowNum > 1:
        # signal longer than 10s
        plt.figure(1, figsize=(75,45), dpi = 96)
        plt.clf()
        
        for i in range(0, subplot_RowNum - 1):
            curr_ecg = ecg_sig[range(i * fs * 10, (i+1) * fs * 10)]
            plt.subplot(subplot_RowNum, 1, i+1)
            plt.plot(curr_ecg * 10 * pixel_per_mm, color='k', label='ecg', linewidth=2, alpha=0.9)
            for xx in range(0, max_point_time_count+pixel_per_mm, pixel_per_mm):
                if xx % (pixel_per_mm*5) == 0:
                    plt.plot([xx, xx], [min_point_amp_count,max_point_amp_count], color='r', linestyle='-', linewidth=2,alpha=0.6)
                else:
                    plt.plot([xx, xx], [min_point_amp_count,max_point_amp_count], color='r', linestyle='-', linewidth=1,alpha=0.4)

            for yy in range(min_point_amp_count, max_point_amp_count,pixel_per_mm):
                if yy % (5*pixel_per_mm) == 0 and yy != 0:
                    plt.plot([0, max_point_time_count], [yy, yy], color='r', linestyle='-', linewidth=2,alpha=0.6)
                else:
                    if yy != 0:
                        plt.plot([0, max_point_time_count], [yy, yy], color='r', linestyle='-', linewidth=1,alpha=0.4)
                    else:
                        plt.plot([0, max_point_time_count], [0, 0], color='r', linestyle='-', linewidth=2,alpha=0.6)
            plt.xlabel('time/s',fontsize=40,horizontalalignment='right')
            plt.yticks(fontsize=25)
            plt.ylabel('amp/mv',fontsize=25,verticalalignment='top',rotation='vertical')
            plt.xticks(range(0, max_point_time_count, int( mm_per_s * pixel_per_mm)), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=40)
            plt.yticks(range(min_point_amp_count, max_point_amp_count, int(0.5 * mm_per_mv * pixel_per_mm)), [-2.5, -2.0, -1.5, -1.0, 0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=40)
            plt.axis('equal')
        
        curr_ecg = ecg_sig[range( (subplot_RowNum - 1) * fs * 10, len(ecg_sig) )]
        plt.subplot(subplot_RowNum, 1, subplot_RowNum)
        plt.plot(curr_ecg * 10 * pixel_per_mm, color='k', label='ecg', linewidth=2, alpha=0.9)
        for xx in range(0, max_point_time_count+pixel_per_mm, pixel_per_mm):
            if xx % (pixel_per_mm*5) == 0:
                plt.plot([xx, xx], [min_point_amp_count,max_point_amp_count], color='r', linestyle='-', linewidth=2,alpha=0.6)
            else:
                plt.plot([xx, xx], [min_point_amp_count,max_point_amp_count], color='r', linestyle='-', linewidth=1,alpha=0.4)
        
        for yy in range(min_point_amp_count, max_point_amp_count,pixel_per_mm):
            if yy % (5*pixel_per_mm) == 0 and yy != 0:
                plt.plot([0, max_point_time_count], [yy, yy], color='r', linestyle='-', linewidth=2,alpha=0.6)
            else:
                if yy != 0:
                    plt.plot([0, max_point_time_count], [yy, yy], color='r', linestyle='-', linewidth=1,alpha=0.4)
                else:
                    plt.plot([0, max_point_time_count], [0, 0], color='r', linestyle='-', linewidth=2,alpha=0.6)
        plt.xlabel('time/s',fontsize=40,horizontalalignment='right')
        plt.yticks(fontsize=25)
        plt.ylabel('amp/mv',fontsize=25,verticalalignment='top',rotation='vertical')
        plt.xticks(range(0, max_point_time_count, int( mm_per_s * pixel_per_mm)), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=40)
        plt.yticks(range(min_point_amp_count, max_point_amp_count, int(0.5 * mm_per_mv * pixel_per_mm)), [-2.5, -2.0, -1.5, -1.0, 0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=40)
        plt.axis('equal')
        
        plt.savefig(save_path + '/' + png_name)
    
    


    
    

######## send results ########


####### main #######
if __name__ == 'main':
    
    ###test on aliveCor data###    
    path = '/home/dingzj/workspace/ECG/data/training2017'
    name = 'A00001'
    # read data    
    signal, hea = read_signal(path, name)
    d_signal = signal.adc()[:,0]
    
    # calc qrs
    qrs_indices, qrs_time = qrs_detection(signal, min_bpm=20, max_bpm=230, smooth_window=150)
    
    # rr and drr
    rr = rr_interval(qrs_time, thresh=16+2, section='sub')
    drr = drr_interval(rr)
    
    #nec
    nec, nec_norm = nec_calc(rr, drr)    
    
    #classify
    pred, word = classify(nec_norm, thresh=0.75, words=True)
    
   
    # plot
    #name_file = 'A00001.mat'
    #record = sio.loadmat(path+'/'+name_file,squeeze_me=True)
    #sig = record['keypoint']['ecg'].tolist()
    #sig = record['val'].tolist()
    plot_save_ecg(signal.p_signals, save_path = '/home/dingzj/workspace/ECG/Test_bin/pipeline',
                  png_name = name+'.png', fs = signal.fs)

    
    #send
    itchat.auto_login(True)
    @itchat.msg_register(TEXT, MAP, CARD, NOTE)
    def text_reply(msg, words=word):
        itchat.send('%s' % (words), msg['FromUserName'])
        itchat.send('@img@%s' % (path+'/ecg.png'), msg['FromUserName'])
    itchat.run()
    
    ###OR read from qrs files
#    #read qrs
#    qrs = read_qrs('/home/dingzj/workspace/ECG/data/challenge2017_qrs', 'A00001')
#    qrs = qrs_convert(qrs)   
#    
#    #calculate rr and drr
#    
#    rr = rr_interval(qrs, thresh=16+2, section='sub')
#    drr = drr_interval(rr) 
    
    ### test data from SunLi###
    path = '/home/dingzj/workspace/ECG/Test_bin/pipeline'
    name = 'test_20171219_125sample_5smooth.mat'
    
    record = sio.loadmat(path+'/'+name,squeeze_me=True)
    d_sig = record['So']
    fs = 125.
    
    beats = detect_beats(d_sig, rate = fs)
    
    
    


