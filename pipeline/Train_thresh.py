import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
import soundfile as sf
from sys import argv
import pandas
import wfdb

####### input signal #########
def read_signal(file_path, record_name):
	file = file_path + '/' + record_name
	signal = wfdb.rdsamp(recordname = file)
	hea = wfdb.rdheader(file)
	return signal, hea

def read_all_signal(file_path, records):
    n_sample = records.shape[0]
    samples = {}
    for i in range(n_sample):
        name = records[0][i]
        signal, hea = read_signal(file_path, name)
        samples[name + 'signal'] = signal
        samples[name + 'header'] = hea
    return samples

######## detect beats ##############

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
    


####### RR and DRR interval ##########

def indice2time(peak_indices, fs=300.):
    time_gap = 1. / float(fs)
    time_seq = np.zeros(len(peak_indices))
    for i in range(0, len(peak_indices)):
        time_seq[i] = peak_indices[i] * time_gap
    return time_seq

def define_usebeats(qrs_indices):
    peaks = len(qrs_indices)
    if peaks >= 18 and peaks < 34:
        return 16
    if peaks >= 34:
        return 32

def rr_interval( qrs, thresh = 32+2, section = 'sub' ):
    #x = qrs[0]
    x = qrs  
    #    
    if len(x) < thresh:
        print('Too short!')
        return -1
    if section == 'sub':
        x = x[range(0,thresh)]
        # rr
        curr_rr = np.zeros(thresh-1)
        for i in range(1, thresh):
            curr_rr[i-1] = qrs[i] - qrs[i-1]
        return curr_rr
    elif section == 'all':
        # rr
        curr_rr = np.zeros(len(x)-1)
        for i in range(1, len(x)):
            curr_rr[i-1] = qrs[i] - qrs[i-1]
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
    
    xmin = np.floor(10* (np.min(rr) - gap)) / 10.
    ymin = np.floor(10*  (np.min(drr) - gap)) / 10.

    xmax = np.ceil(10* (np.max(rr) + gap)) / 10. 
    ymax = np.ceil(10* (np.max(drr) + gap)) / 10. 
    
    
    xc = int( (xmax - xmin) // gap + 1 )
    yc = int( (ymax - ymin) // gap + 1 )

    mat = np.zeros((xc, yc))
    for i in range(0, xc):
        # the ith column        
        xleft = xmin + i * gap
        xright = xmin + (i+1) * gap
        new_rr_1 = (xleft <= rr)
        new_rr_2 = (rr < xright)
        new_rr = (new_rr_1 == new_rr_2)
        if np.sum(new_rr) > 0:
            for j in range(0, yc):
                # the jth                 
                ydown = ymin + j * gap
                yup = ymin + (j+1) * gap
                new_drr_1 = (ydown <= drr)
                new_drr_2 = (drr < yup)
                new_drr = (new_drr_1 == new_drr_2)
                if np.sum(new_drr) > 0:                
                    for k in range(0, len(new_rr)):
                        if (new_rr[k] == True) & (new_drr[k] == True):
                            mat[i,j] += 1
        
    mat = (mat>0)
    nec = sum(sum(mat))
    normalized_nec = nec/float(len(rr))

    return nec, normalized_nec


####### classification #######

def define_thresh(usebeats):
    if usebeats == 16:
        return 0.8125222340803984
    if usebeats == 32:
        return 0.6876852600377257


def classify( nec_norm, thresh, words = True ):
    prediction = 'Normal'    
    if nec_norm > thresh:
        prediction = 'Possible Atrial Fibrillation' 
    if words == True:
        print( prediction )
    return prediction


def comparison(name, label, prediction):
    if (label == 'A') and (prediction == 'Possible Atrial Fibrillation'):
        return 'TP'
    elif (label != 'A') and (prediction == 'Possible Atrial Fibrillation'):
        return 'FP'
    elif (label == 'A') and (prediction != 'Possible Atrial Fibrillation'):
        return 'FN'
    else:
        return 'TN'

def roc_curve(train_data, nec_dict):
    """
    train_data: pandas data frame
    nec_dict: a dictionary
    """    
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
        for j in list(train_data.index):
            curr_sample = train_data[0][j]
            curr_label = train_data[1][j]
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




def find_threshold(sensitivity, specificity,
                   threshold, sensi_thresh=None,
                   speci_thresh=None):
    index = 0.
    min_dist = 1000000.
    if (sensi_thresh == None) and (speci_thresh == None):
        #find the dot with a shortest distance to (0,1)
        for i in range(0, len(sensitivity)):
            curr_dist = np.sqrt( (1-specificity[i])*(1-specificity[i]) +
                                 (sensitivity[i]-1)*(sensitivity[i]-1) )
            if curr_dist < min_dist:
                min_dist = curr_dist
                index = i
    elif (sensi_thresh != None) and (speci_thresh != None):
        for i in range(0, len(sensitivity)):
            curr_dist = np.sqrt((sensitivity[i]-sensi_thresh) * (sensitivity[i]-sensi_thresh) +
                                (specificity[i]-speci_thresh) * (specificity[i]-speci_thresh))
            if curr_dist < min_dist:
                min_dist = curr_dist
                index = i
    elif (sensi_thresh != None) and (speci_thresh == None):
        for i in range(0, len(sensitivity)):
            curr_dist = np.abs(sensitivity[i] - sensi_thresh)
            if curr_dist < min_dist:
                min_dist = curr_dist
                index = i
    elif (sensi_thresh == None) and (speci_thresh != None):
        for i in range(0, len(specificity)):
            curr_dist = np.abs(specificity[i] - speci_thresh)
            if curr_dist < min_dist:
                min_dist = curr_dist
                index = i

    return threshold[index], sensitivity[index], specificity[index], index
      
      
def threshold(path='/home/dingzj/workspace/ECG/data/training2017', use_beat = 32,
                   sensi_thresh = None, speci_thresh = None):  
    records = pandas.read_table(path + '/RECORDS', header=None) #records[0][0], [0][1], a col
    samples = read_all_signal(path, records)
    labels = pandas.read_csv(path + '/REFERENCE-v3.csv', header=None) #
        
    nec_dict = dict()
    ix = list()
    for i in range(len(labels)):
        name = labels[0][i]
        curr_label = labels[1][i]
        if curr_label is 'N' or curr_label is 'A':
            signal = samples[name+'signal']
            d_signal = signal.adc()[:,0]
            #qrs indices        
            #qrs_indices, qrs_time = qrs_detection(signal, min_bpm=20, max_bpm=230, smooth_window=150)
            qrs_indices = detect_beats(d_signal, 300)
            qrs_time = indice2time(qrs_indices, 300)
            # rr and drr
            rr = rr_interval(qrs_time, thresh=use_beat+2, section='sub')
            if (type(rr) is np.ndarray) and (len(rr) > use_beat):
                ix.append(i)            
                drr = drr_interval(rr)
                #nec
                nec, nec_norm = nec_calc(rr, drr)  
                nec_dict[name] = nec_norm
        
   
    labels = labels.iloc[ix]
    results, sensitivity, specificity,  gaps = roc_curve(labels, nec_dict)
    plotROC(sensitivity, specificity)
    final_thresh, sensi_level, speci_level, index= find_threshold(sensitivity, 
                                                                  specificity,
                                                                  gaps, 
                                                                  sensi_thresh,
                                                                  speci_thresh)
    return final_thresh, sensi_level, speci_level       



###### check final threshold ########

def check_thresh(path='/home/dingzj/workspace/ECG/data/training2017', 
                 use_beat = 32, final_thresh = 0.75):
    
    records = pandas.read_table(path + '/RECORDS', header=None) #records[0][0], [0][1], a col
    samples = read_all_signal(path, records)
    labels = pandas.read_csv(path + '/REFERENCE-v3.csv', header=None) #
                    
    nec_dict = dict()
    ix = list()
    for i in range(len(labels)):
        name = labels[0][i]
        curr_label = labels[1][i]
        if curr_label is 'N' or curr_label is 'A':
            signal = samples[name+'signal']
            d_signal = signal.adc()[:,0]
            #qrs indices        
            #qrs_indices, qrs_time = qrs_detection(signal, min_bpm=20, max_bpm=230, smooth_window=150)
            qrs_indices = detect_beats(d_signal, 300)
            qrs_time = indice2time(qrs_indices, 300)
            # rr and drr
            rr = rr_interval(qrs_time, thresh=use_beat+2, section='sub')
            if (type(rr) is np.ndarray) and (len(rr) > use_beat):
                ix.append(i)            
                drr = drr_interval(rr)
                #nec
                nec, nec_norm = nec_calc(rr, drr)  
                nec_dict[name] = nec_norm
    
    labels = labels.iloc[ix]
    
    
    tp = 0.
    fp = 0.
    tn = 0.
    fn = 0.    
    for i in list(labels.index):
        curr_sample = labels[0][i]
        curr_label = labels[1][i]
        pred = classify( nec_dict[curr_sample], final_thresh, words = False)
        comp = comparison( curr_sample, curr_label, pred)
        if comp == 'TP':
            tp += 1.
        elif comp == 'TN':
            tn += 1.
        elif comp == 'FP':
            fp += 1.
        elif comp == 'FN':
            fn += 1.
    
    return tp, fp, tn, fn

####### draw ROC #######
def plotROC(sensitivity, specificity):
    plt.figure()
    plt.clf()
    plt.plot(1-specificity, sensitivity)
    plt.xlabel('1-specificity')
    plt.ylabel('sensitivity')
    plt.title('ROC curve for Train Data')
    auc = np.trapz(sensitivity, 1-specificity)
    print("The AUC is", str(auc))


###### draw NEC #######
def draw_nec(rr, drr, gap = 0.025):
    if ( len(rr) - len(drr) ) != 1:
        print('Length of rr and rr is not equal')
        return -1
    # same length 
    rr = np.delete(rr, 0, axis=0)
    
    fig = plt.figure(1)
    fig.clf()
    plt.scatter(rr, drr, linewidths = 0.5)
    
    xmin = np.floor(10* (np.min(rr) - gap)) / 10.
    ymin = np.floor(10*  (np.min(drr) - gap)) / 10.

    xmax = np.ceil(10* (np.max(rr) + gap)) / 10. 
    ymax = np.ceil(10* (np.max(drr) + gap)) / 10. 
        
    xc = int( (xmax - xmin) // gap + 1 )
    yc = int( (ymax - ymin) // gap + 1 )

    mat = np.zeros((xc, yc))
    for i in range(0, xc):
        # the ith column        
        xleft = xmin + i * gap
        xright = xmin + (i+1) * gap
        new_rr_1 = (xleft <= rr)
        new_rr_2 = (rr < xright)
        new_rr = (new_rr_1 == new_rr_2)
        
        plt.axvline(xright, linewidth = 0.1)
        plt.axvline(xleft, linewidth = 0.1)
        
        for j in range(0, yc):
            # the jth                 
            ydown = ymin + j * gap
            yup = ymin + (j+1) * gap
            new_drr_1 = (ydown <= drr)
            new_drr_2 = (drr < yup)
            new_drr = (new_drr_1 == new_drr_2)
            plt.axhline(yup, linewidth = 0.1)
            plt.axhline(ydown, linewidth = 0.1)
            for k in range(0, len(new_rr)):
                if (new_rr[k] == True) & (new_drr[k] == True):
                    mat[i,j] += 1
        
    mat = (mat>0)
    nec = sum(sum(mat))
    normalized_nec = nec/float(len(rr))
    
    return nec, normalized_nec
###### main ######
if __name__ == '__main__':
    
    final_thresh, sensi_level, speci_level = threshold(path='/home/dingzj/workspace/ECG/data/training2017',
                                                       use_beat = 16,
                                                       sensi_thresh = None, speci_thresh = None)
                                                       
    tp, fp, tn, fn = check_thresh(path='/home/dingzj/workspace/ECG/data/training2017', 
                                  use_beat = 32, final_thresh = final_thresh)
    
    draw_nec(rr, drr)