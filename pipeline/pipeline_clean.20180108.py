import numpy as np
import scipy.io as sio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.ndimage
import soundfile as sf
import math
import control
from sys import argv


####### input wav ##########
def demod(Yf, fc = 19000., fs = 48000.):
    lensig = float(Yf.shape[0])
    t = np.arange(0, (lensig-1)/fs + 1/fs, 1/fs)
    yq = np.multiply( scipy.signal.hilbert(Yf), np.exp(-1j * 2 * math.pi * fc * t) )
    yq = np.insert(np.diff(control.unwrap(np.angle(yq))), 0, 0.)
    return yq

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))   

def cutoff(sig, cutoff_thresh = 0.005, index = 100):
    # 1 derivative
    deriv1 = list()
    for i in range(1, len(sig)):
        tmp = sig[i] - sig[i-1]
        deriv1.append(tmp)
    # 2 derivative
    deriv2 = list()
    for i in range(1, len(deriv1)):
        tmp = deriv1[i] - deriv1[i-1]
        deriv2.append(tmp)
    #find spot
    tmp = np.where(np.abs(deriv2)>=cutoff_thresh)
    tmp = tmp[0]
    tmp2 = np.where(tmp <= index)
    tmp2 = tmp2[0]
    spot = tmp[ tmp2[len(tmp2) - 1] ] + 1
    #return value
    tmp = list()
    for i in range(spot, len(sig)):
        tmp.append(sig[i])
    return np.array(tmp)
    
def del_phaseShift(sig, shift_time = 6 * 300):
    tmp = list()
    for i in range(shift_time, len(sig)):
        tmp.append(sig[i])
    return np.array(tmp)
    
def add_sig(sig, add_time = 6 * 300):
    tmp = list()
    for i in range(len(sig)):
        tmp.append(sig[i])
    for i in range(len(sig), len(sig) + add_time):
        tmp.append(0.0)
    return np.array(tmp)

def keep_sig(sig, fs = 300, keep_time = 30):
    if len(sig) <= fs * keep_time:
        return sig
    else:
        tmp = int(len(sig) - fs * keep_time)
        return sig[tmp:len(sig)]

def audio2mat(path, name, param_file):
    #read wav
    sig, fs = sf.read( path + '/' + name)
    sig = sig[:,0]
    sig_time = np.floor(len(sig) / fs)
    # read parameters
    hd_15k = sio.loadmat(param_file)
    ahd = hd_15k['aHd_15k']
    bhd = hd_15k['bHd_15k']
    ahigh = hd_15k['ahighpassFIR1MM']
    bhigh = hd_15k['bhighpassFIR1MM']
    # filter
    Yf = scipy.signal.lfilter(bhd[0,:], ahd[0,:], sig)
    # demodulate 
    result_test = demod(Yf, 19000., float(fs))
    # smooth
    result_20k = smooth(result_test, 961)
    result_20k = smooth(result_20k, 959)
    # down sampling
    val2 = result_20k[range(0, result_20k.shape[0] , int(fs/300))]
    val2 -= np.mean(val2)
    #cut off signal
    val2 = cutoff(val2, 0.005, 100)
    # add 0
    val2 = add_sig(val2, add_time = 6 * 300)
    # baseline wander
    val = scipy.signal.lfilter(bhigh[0,:], ahigh[0,:], val2)
    # delete phase shift
    val = del_phaseShift(val, shift_time = 6 * 300)
    # keep 30s long sig
    val = keep_sig(val, 300, sig_time)
    
    return val

######### inverted ecg preprocessing ########
def inverted_ecg(sig, qrs_indices, thresh = 0.4):
    qrs_minus = 0.
    minus2plus = 0.
    for i in range(len(qrs_indices)):
        # val
        if sig[qrs_indices[i]] < 0.:
            qrs_minus += 1.
        #minus to plus
        bckmax = 0.
        bckix = qrs_indices[i]
        fwdmax = sig[qrs_indices[i]]
        fwdix = qrs_indices[i]
        index = 6
        if qrs_indices[i] - index < 0 :
            index = qrs_indices[i]
        if qrs_indices[i] + index > len(sig) - 1:
            index = len(sig) - 1 - (qrs_indices[i] + 1)
        for j in range(index):
            if np.abs(sig[qrs_indices[i]-j]) >= bckmax:
                bckmax = np.abs(sig[qrs_indices[i]-j])
                bckix = qrs_indices[i] - j
            if sig[qrs_indices[i]+(j+1)] >= fwdmax:
                fwdmax = sig[qrs_indices[i]+(j+1)]
                fwdix = qrs_indices[i] + (j+1)
        if sig[bckix] < 0. and sig[fwdix] > 0.:
            minus2plus += 1.
            
    thresh = thresh * float(len(qrs_indices))

    if qrs_minus >= thresh and minus2plus >= thresh:
        return True
    else:
        return False

######## QRS detection #######                   
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
    for i in range(int(len(decg_power) / ransac_window_size)):
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
    if peaks < 18:
        return 0
    elif peaks >= 18 and peaks < 34:
        return 16
    else:
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
        for i in range(0, len(x)):
            curr_rr[i-1] = qrs[i] - qrs[i-1]
        return curr_rr

def drr_interval(rr):
    drr = np.zeros(len(rr)-1)
    for i in range(1, len(rr)):
        drr[i-1] = rr[i] - rr[i-1] 
    return drr


##### calculate sqi for a segment #######
def intersect(a, b):
    c = list(set(a) & set(b))
    c = np.array(c)
    return np.sort(c)
    
def calculate_sqi(seg, qrs, fs=300, IN=0.5, w=10 ):
    """
    calculate SQI for single segment of data
    Inputs: seg -- 1d ecg array
            qrs -- r peak indices
            fs -- ecg sampling rate (default: 300 for alivecor)
            IN -- threshold for average correlation coefficient (default: 0.5)
    Outputs: R2 -- average correlation coefficient
             SQI -- 0 if bad, 1 if good
    """
    SQI = 0
    R2 = 0.
    avtempl = 0.
    ts = list()
       
    rr = np.diff(qrs)
    hrs = np.floor(np.mean(rr))
        
    # Apply Rules: (1) <40 || >180 (2) max(RR) > 3 (3) max(RR) / min(RR) < 2.2
    if len(qrs)<(40*w)/60 or len(qrs)>(180*w)/60 or max(rr)>3*fs or (max(rr)/min(rr))>=2.2:
        SQI = 0
        R2 = None
        return SQI, R2
    else:
        #calculate
        hr = 60 * fs / hrs
        j = np.where( qrs > hrs/2 )[0]
        l = np.where( qrs + hrs/2 < len(seg) )[0]
        jl = intersect(j, l)
        if len(l) == 0:
            SQI = 0
            R2 = None
            return SQI, R2
        else:
            for k in jl:
                t = seg[(qrs[k]-int(hrs/2)):(qrs[k]+int(hrs/2))]
                t /= np.linalg.norm(t, ord=2)
                ts.append(t)
                avtempl += t
 
    # mean curve
    avtempl /= len(ts)    
    # correlation
    cor = list()
    for i in range(len(ts)):
        curr_cor = scipy.stats.pearsonr(ts[i], avtempl)
        cor.append(curr_cor[0])
    R2 = np.mean(cor)
    if R2 > IN:
        SQI = 1
    else:
        SQI = 0
    
    return SQI, R2
    
##### calculate sqi for a signal #######

def signal_sqi(d_sig, qrs_indices, fs = 300, IN = 0.5, w=10):
    """
    calculate SQI for a ecg signal
    input: d_sig -- digital signal, numpy array
           qrs_indices -- indices of r peak
           fs -- ecg sampling rate (default: 300 for alivecor)
           IN -- threshold for average correlation coefficient (default: 0.5)
           w -- window length (default: 10s)
    output: R2 and SQI for each segment (w long) of d_sig
    """
    windsize = fs * w
    seg_num = int(len(d_sig) / (fs*w))
    if len(d_sig) % (fs * w) > 0:
        seg_num += 1
    
    sqi_list = list()
    r2_list = list()
    for i in range(0, seg_num):
        min_p = i*windsize
        if i < seg_num -1:
            max_p = (i+1) * windsize
        else:
            max_p = len(d_sig)-1
        
        #curr ecg
        curr_ecg = d_sig[range(min_p, max_p)]
        #current qrs_indices
        m1 = qrs_indices >= min_p
        m2 = qrs_indices <= max_p
        curr_qrs = qrs_indices[m1*m2]
        curr_qrs -= min_p
        
        # calculate sqi
        sqi, r2 = calculate_sqi(curr_ecg, curr_qrs, fs, IN, w)
        sqi_list.append(sqi)
        r2_list.append(r2)
         
    return sqi_list, r2_list


######## calculate NEC #######

def nec_calc(rr, drr, gap = 0.025):
    if ( len(rr) - len(drr) ) != 1:
        print('Length of rr and rr is not equal')
        return -1
    # same length 
    rr = np.delete(rr, 0, axis=0)
    
    # range
    xmin = np.floor(10. * (np.min(rr) - gap)) / 10.
    ymin = np.floor(10. *  (np.min(drr) - gap)) / 10.

    xmax = np.ceil(10. * (np.max(rr) + gap)) / 10. 
    ymax = np.ceil(10. * (np.max(drr) + gap)) / 10. 
    
    
    #cells
#    xrange = np.arange(xmin, xmax + gap, gap)
#    yrange = np.arange(ymin, ymax + gap, gap)
    
    
#    # calculate non-empty cells
#    mat = np.zeros( ( len(yrange), len(xrange) ) )
#    for i in range(len(yrange)):
#        # the ith row, bottom to up
#        ydown = yrange[i]
#        yup = yrange[i] + gap
#        DRRdown = (ydown <= drr)
#        DRRup = (drr < yup)
#        DRR = (DRRdown == DRRup)
#        
#        if np.sum(DRR) > 0:
#            for j in range(len(xrange)):
#                # the jth column, left to right
#                xleft = xrange[j]
#                xright = xrange[j] + gap
#                RRleft = (xleft <= rr)
#                RRright = (rr < xright)
#                RR = (RRleft == RRright)
#                if np.sum(RR) > 0:
#                    a = np.where(RR == True)
#                    b = np.where(DRR == True)
#                    c = list(set(a[0]) & set(b[0]))
#                    if len(c) > 0:
#                        mat[i,j] = float(len(c))
#    mat = (mat>0)
#    nec = sum(sum(mat))
#    normalized_nec = nec/float(len(rr))
    
    
    # the original method
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
    
 
     

######## draw ECG ############

def define_figsize(d_sig, fs = 300):
    sig_time = float(len(d_sig)) / float(fs) 
    if  sig_time <= 10:
        figsize = (40,10)
        return figsize
    if sig_time > 10 and sig_time <= 20:
        figsize = (40,25)
        return figsize
    if sig_time > 20 and sig_time <= 30:
        figsize = (40,35)
        return figsize
    if sig_time > 30 and sig_time <= 40:
        figsize = (40,40)
        return figsize
    if sig_time > 40 and sig_time <= 50:
        figsize = (40, 50)
        return figsize
    if sig_time > 50 and sig_time <= 60:
        figsize = (40, 60)
        return figsize
    if sig_time > 60 and sig_time <=70:
        figsize = (40, 70)
        return figsize
    if sig_time > 70 and sig_time <= 80:
        figsize = (40, 80)
        return figsize
    if sig_time > 80 and sig_time <= 90:
        figsize = (40, 90)
        return figsize
    if sig_time > 90 and sig_time <= 100:
        figsize = (40, 100)
        return figsize
    if sig_time > 100 and sig_time <= 110:
        figsize = (40, 110)
        return figsize
    if sig_time > 110 and sig_time <= 120:
        figsize = (40, 120)
        return figsize

def adjust_ylim(ecg, max_mv = 2.5, min_mv = -2.5):
    maxval = np.max(ecg)
    minval = np.min(ecg)
    # totally not in the same range
    if minval > max_mv or maxval < min_mv:
        maxval += 0.5
        min_mv += -0.5
        return maxval, minval
    else:
        return max_mv, min_mv
    
    
def plot_red_lines(fig, pixel_per_mm, max_point_time_count, max_point_amp_count, min_point_amp_count  ):
    for xx in range(0, max_point_time_count+pixel_per_mm, pixel_per_mm):
        if xx % (pixel_per_mm*5) == 0:    
            fig.plot([xx, xx], [min_point_amp_count,max_point_amp_count], color='r',
                      linestyle='-', linewidth=1, alpha=0.6)
        else:
            fig.plot([xx, xx], [min_point_amp_count,max_point_amp_count], color='r', 
                     linestyle='-', linewidth=0.5, alpha=0.4)
    for yy in range(min_point_amp_count, max_point_amp_count,pixel_per_mm):
        if yy % (5*pixel_per_mm) == 0 and yy != 0:
            fig.plot([0, max_point_time_count], [yy, yy], color='r', linestyle='-', linewidth=1,alpha=0.6)
        else:
            if yy != 0:
                fig.plot([0, max_point_time_count], [yy, yy], color='r', linestyle='-',
                         linewidth=0.5,alpha=0.4)
            else:
                fig.plot([0, max_point_time_count], [0, 0], color='r', linestyle='-', 
                         linewidth=1,alpha=0.6)

def plot_curve(fig, sig, mm_per_mv, pixel_per_mm, max_mv, min_mv):
    if (np.max(sig) <= max_mv) and (np.min(sig) >= min_mv):
        fig.plot(sig * mm_per_mv * pixel_per_mm, color='k', label='ecg', linewidth=1.5, alpha=1)
    else:
        ix_up = np.where(sig > max_mv)
        ix_up = ix_up[0]        
        ix_down = np.where(sig < min_mv)
        ix_down = ix_down[0]
        ix = np.concatenate((ix_up, ix_down))
        ix = np.sort(ix)
        #find sub-curves out of bounds
        tmp = list()
        if len(ix) >= 2:
            start = ix[0]
            end = ix[1]
            for i in range(1,len(ix)):
                if i == len(ix) - 1:
                    if ix[i] - ix[i-1] > 1:
                        end = ix[i-1]
                        tmp.append([start, end])
                        tmp.append([ix[i]-1, ix[i]])
                    else:
                        end = ix[i]
                        tmp.append([start, end])   
                else:
                    if ix[i] - ix[i-1] > 1:
                        end = ix[i-1]
                        tmp.append([start, end])
                        start = ix[i]
                        end = ix[i+1]  
        elif len(ix) == 1:
            out = ix[0]
            if out > 0 and out < (len(sig)-1):
                tmp.append([out-1, out])
            elif out == 0:
                tmp.append([out, out+1])
            elif out == len(sig)-1:
                tmp.append([out-1, out])
        #find sub-curves in bounds
        curves = list()
        if tmp[0][0] > 0:
            start = 0
            end = tmp[0][0]
            for i in range(0, len(tmp)):
                if i < len(tmp) - 1:
                    curves.append([start, end])
                    start = tmp[i][1] + 1
                    end = tmp[i+1][0]
                else:
                    curves.append([start, end])
                    if tmp[i][1] == len(sig) - 1:
                        continue
                    else:
                        start = tmp[i][1] + 1
                        end = len(sig) - 1
                        curves.append([start,end])   
        else:
            start = tmp[0][1] + 1
            end = tmp[1][0]
            for i in range(1, len(tmp)):
                if i < len(tmp) - 1:
                    curves.append([start, end])
                    start = tmp[i][1] + 1
                    end = tmp[i+1][0]
                else:
                    curves.append([start, end])
                    if tmp[i][1] >= len(sig) - 2:
                        continue
                    else:
                        start = tmp[i][1] + 1
                        end = len(sig) - 1
                        curves.append([start,end])  

        #draw curves
        for i in range(len(curves)):
            start = curves[i][0]
            end = curves[i][1]
            if i != 0 and i != len(curves) - 1:
                # add some dots                
                if sig[start] < 0:
                    fig.plot(range(start-1, start+1), 
                             [min_mv * mm_per_mv * pixel_per_mm, 
                              sig[start] * mm_per_mv * pixel_per_mm],
                             color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)
                if sig[start] > 0:
                    fig.plot(range(start-1, start+1), 
                             [max_mv * mm_per_mv * pixel_per_mm, 
                              sig[start] * mm_per_mv * pixel_per_mm],
                             color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)
                
                if sig[end] < 0:
                    fig.plot(range(end-1, end+1), 
                             [sig[end-1] * mm_per_mv * pixel_per_mm, 
                              min_mv * mm_per_mv * pixel_per_mm],
                             color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)
                if sig[end] > 0:
                    fig.plot(range(end-1, end+1), 
                             [sig[end-1] * mm_per_mv * pixel_per_mm, 
                              max_mv * mm_per_mv * pixel_per_mm],
                             color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)    
                #the sub-curve
                fig.plot(range(start,end), sig[start:end] * mm_per_mv * pixel_per_mm, 
                         color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)
            elif i == 0:
                if sig[end] < 0:
                    fig.plot(range(end-1, end+1), 
                             [sig[end-1] * mm_per_mv * pixel_per_mm, 
                              min_mv * mm_per_mv * pixel_per_mm],
                             color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)
                if sig[end] > 0:
                    fig.plot(range(end-1, end+1), 
                             [sig[end-1] * mm_per_mv * pixel_per_mm, 
                              max_mv * mm_per_mv * pixel_per_mm],
                             color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1) 
                fig.plot(range(start,end), sig[start:end] * mm_per_mv * pixel_per_mm, 
                         color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)
            else:
                if sig[start] < 0:
                    fig.plot(range(start-1, start+1), 
                             [min_mv * mm_per_mv * pixel_per_mm, 
                              sig[start] * mm_per_mv * pixel_per_mm],
                             color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)
                if sig[start] > 0:
                    fig.plot(range(start-1, start+1), 
                             [max_mv * mm_per_mv * pixel_per_mm, 
                              sig[start] * mm_per_mv * pixel_per_mm],
                             color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)
                fig.plot(range(start,end), sig[start:end] * mm_per_mv * pixel_per_mm, 
                         color = 'k', label = 'ecg', linewidth = 1.5, alpha = 1)
            
        
        


def plot_save_ecg(ecg_sig, qrs_indices, #np.array
                  #save_path, 
                  #png_name, 
                  figsize = (25,35),
                  fs = 300, 
                  mm_per_mv = 10.0, mm_per_s = 25.0, 
                  max_mv = 2.5, min_mv = -2.5):
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
        # define ylim
        max_mv, min_mv = adjust_ylim(ecg_sig, max_mv = max_mv, min_mv = min_mv)

        fig = plt.figure(1,figsize=figsize,dpi=96)
        fig.clf()
        
              
        plot_red_lines(fig, pixel_per_mm, max_point_time_count, max_point_amp_count, min_point_amp_count  ) 
        #fig.plot(ecg_sig * mm_per_mv * pixel_per_mm, color='k', label='ecg', linewidth=1.5, alpha=1)
        plot_curve(fig = fig, sig = ecg_sig, 
                   mm_per_mv = mm_per_mv, 
                   pixel_per_mm = pixel_per_mm, 
                   max_mv = max_mv, min_mv = min_mv)        
        # r peak annotations
        # horizontal black lines
        fig.plot( [0, max_point_time_count], [min_point_amp_count, min_point_amp_count], 
                 color = 'k',  linewidth = 2, alpha = 0.7 )
        # vertical r pkeas
        for xx in qrs_indices:
            fig.plot( [xx, xx], [min_point_amp_count, min_point_amp_count + pixel_per_mm * 2 ],
                     color = 'k', linewidth = 2, alpha=0.7 )
        plt.xlabel('time/s',fontsize=40,horizontalalignment='right')
        plt.xticks(range(0, max_point_time_count, int( mm_per_s * pixel_per_mm)), 
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=20)
        
        plt.ylabel('amp/mv',fontsize=40,
                   #verticalalignment='top',
                   rotation='vertical')
        ytick_val = np.arange(min_mv, max_mv, np.ceil((max_mv - min_mv) / 4) )
        plt.yticks(ytick_val * mm_per_mv * pixel_per_mm, ytick_val, fontsize=15)
        #plt.axis('equal')
        #fig.savefig(save_path + '/' + png_name)  
        #plt.close()
        return fig
        
    elif subplot_RowNum > 1:
        # signal longer than 10s
        
        fig = plt.figure(1, figsize=figsize, dpi = 96)
        fig.clf()
        
        for i in range(0, subplot_RowNum - 1):
            min_p = i*fs*10
            max_p = (i+1) * fs * 10
            #curr ecg
            curr_ecg = ecg_sig[range(min_p, max_p)]
            #current qrs_indices
            m1 = qrs_indices >= min_p
            m2 = qrs_indices <= max_p
            curr_qrs = qrs_indices[m1*m2]
            curr_qrs -= min_p
            
            # define ylim
            max_mv_tmp, min_mv_tmp = adjust_ylim(curr_ecg, max_mv = max_mv, min_mv = min_mv)
            max_point_amp_count = int(max_mv_tmp * mm_per_mv * pixel_per_mm)
            min_point_amp_count = int(min_mv_tmp * mm_per_mv * pixel_per_mm)
             
            ax = fig.add_subplot(subplot_RowNum, 1, i+1)
            #ax.clf()
            # the red boxes
            plot_red_lines(ax, pixel_per_mm, max_point_time_count, max_point_amp_count, min_point_amp_count  )              
            #ax.plot(curr_ecg * mm_per_mv * pixel_per_mm, color='k', label='ecg', linewidth=1, alpha=1)
            plot_curve(fig = ax, sig = curr_ecg, 
                       mm_per_mv = mm_per_mv, 
                       pixel_per_mm  = pixel_per_mm, 
                       #max_mv = max_mv, min_mv = min_mv, 
                       max_mv = max_mv_tmp, min_mv = min_mv_tmp) 
                              
            # r peak annotations
            # horizontal black lines
            ax.plot( [0, max_point_time_count], [min_point_amp_count, min_point_amp_count], 
                     color = 'k',  linewidth = 2, alpha = 0.7 )
            # vertical r pkeas
            for xx in curr_qrs:
                ax.plot( [xx, xx], [min_point_amp_count, min_point_amp_count + pixel_per_mm * 3 ], 
                         color = 'k', linewidth = 2, alpha=0.7 ) 
            #plt.xlabel('time/s',fontsize=10,horizontalalignment='right')
            #plt.yticks(fontsize=10)     
            xtick = range( i*10, i*10 +10 )
            plt.xticks(range(0, max_point_time_count, int( mm_per_s * pixel_per_mm)), 
                       xtick, fontsize=15)
            #ytick_val = np.arange(min_mv, max_mv, np.ceil( (max_mv - min_mv) / 4) )
            ytick_val = np.arange(min_mv_tmp, max_mv_tmp, np.ceil( (max_mv_tmp - min_mv_tmp) / 4) )
            plt.yticks(ytick_val * mm_per_mv * pixel_per_mm , ytick_val, fontsize=15)
            plt.ylabel('amp/mv',fontsize=25,
                       #verticalalignment='top',
                       rotation='vertical')
            
            plt.axis('equal')
        
        min_p = (subplot_RowNum - 1) * fs * 10
        max_p = len(ecg_sig)
        curr_ecg = ecg_sig[range( min_p, max_p )]
        m1 = qrs_indices >= min_p
        m2 = qrs_indices <= max_p
        curr_qrs = qrs_indices[m1*m2]
        curr_qrs -= min_p
        
        # define ylim
        max_mv_tmp, min_mv_tmp = adjust_ylim(curr_ecg, max_mv = max_mv, min_mv = min_mv)           
        max_point_amp_count = int(max_mv_tmp * mm_per_mv * pixel_per_mm)
        min_point_amp_count = int(min_mv_tmp * mm_per_mv * pixel_per_mm)
        
        
        ax = fig.add_subplot(subplot_RowNum, 1, subplot_RowNum)
        #ax.clf()
        plot_red_lines(ax, pixel_per_mm, max_point_time_count, max_point_amp_count, min_point_amp_count  )          
        #ax.plot(curr_ecg * mm_per_mv * pixel_per_mm, color='k', label='ecg', linewidth=1, alpha=1)
        plot_curve(fig = ax, sig = curr_ecg, 
                   mm_per_mv = mm_per_mv, 
                   pixel_per_mm = pixel_per_mm, 
                   #max_mv = max_mv, min_mv = min_mv,
                   max_mv = max_mv_tmp, min_mv = min_mv_tmp) 
        # r peak annotations
        # horizontal black lines
        ax.plot( [0, max_point_time_count], [min_point_amp_count, min_point_amp_count],
                 color = 'k',  linewidth = 2, alpha = 0.7 )
        # vertical r pkeas
        for xx in curr_qrs:
            ax.plot( [xx, xx], [min_point_amp_count, min_point_amp_count + pixel_per_mm * 3 ], 
                     color = 'k', linewidth = 2, alpha=0.7 )
        
        xtick = range( ( subplot_RowNum -1 ) * 10, ( subplot_RowNum -1 ) * 10 + 10 )
        plt.xticks(range(0, max_point_time_count, int( mm_per_s * pixel_per_mm)), 
                   xtick, fontsize=15)
        plt.xlabel('time/s',fontsize=25,horizontalalignment='right')
        #plt.yticks(fontsize=10)
        #ytick_val = np.arange(min_mv, max_mv, np.ceil((max_mv - min_mv) / 4) )
        ytick_val = np.arange(min_mv_tmp, max_mv_tmp, np.ceil((max_mv_tmp - min_mv_tmp) / 4) )
        plt.yticks(ytick_val * mm_per_mv * pixel_per_mm, ytick_val, fontsize=15)
        plt.ylabel('amp/mv',fontsize=25,
                   #verticalalignment='top',
                   rotation='vertical')
        
        plt.axis('equal')
        
        #fig.savefig(save_path + '/' + png_name)
        #plt.close()
        return fig
 
def save_figure(figure, save_path, png_name):
    figure.savefig(save_path + '/' + png_name)

##### nec classification pipeline #####
def nec_pipeline(d_sig):
    # r peak #
    qrs_indices = detect_beats(d_sig, 300, ransac_window_size=2.0)
    qrs_time = indice2time(qrs_indices, 300)
    
    # inverted or not
    if inverted_ecg(d_sig, qrs_indices):
        d_sig = -d_sig
    
    # plot ecg #
    figsize = define_figsize(d_sig, 300)
    fig = plot_save_ecg( ecg_sig = d_sig*100., 
                        qrs_indices = np.array(qrs_indices), 
                        #save_path = save_path, 
                        #png_name = name+'.png', 
                        figsize = figsize, fs = 300,
                        mm_per_mv = 10.0, mm_per_s = 25.0,
                        max_mv = 2.5, min_mv = -2.5)
    
    # signal quality #
    sqi, r2 = signal_sqi(d_sig, qrs_indices, fs =300, IN = 0.5, w=10)
        
    # nec classification #
    usebeats = define_usebeats(qrs_indices)
    pred = "Unreadable"
    if usebeats != 0:
        # rr and drr
        rr = rr_interval(qrs_time, thresh=usebeats+2, section='sub') 
        drr = drr_interval(rr) 
    
        #nec
        nec, nec_norm = nec_calc(rr, drr)
        
        #classify
        thresh = define_thresh(usebeats)
        pred = classify(nec_norm, thresh=thresh, words=False) # 0.75, args
     
    else:
        pred = "Unreadable"

    return pred, fig, sqi, r2
    
####### main #######

if __name__ == '__main__':
    #script, path, name, param, save_path = argv
    """  
    usage:
    python script path name param save_path
        
    path: path to .wav file folder
    name: file name of the .wav file, "name.wav"
    param: parameters for wav processing, .mat file
    save_path: path to .png save folder
    """ 
    
    path = "F:/workspace/ECG/data/cluster"
    name = "test2066889762"
    param = "F:/workspace/ECG/Test_bin/pipeline/testcoefficient.mat"
    save_path = "F:/workspace/ECG/results/nec"
        
    #read signal .wav
    d_sig = audio2mat(path = path, name = name + '.pcm', param_file = param )
      
    #run non-empty cell classification and plot figure
    pred, fig, sqi, r2 = nec_pipeline(d_sig = d_sig) #pred: prediction, "normal" etc.; fig as figure
    
    # save figure
    save_figure(fig, save_path, png_name = name + '.png')                
    

    
