import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
import numpy as np

def plot_save_ecg(ecg_sig,save_path,fs = 300,mm_per_mv = 10.0, mm_per_s = 25.0, max_mv = 2.5, min_mv = -2.5):
    '''
    ecg_sig: ecg signal
    fs: ecg sample rate
    save_path: ecg file save path
    '''
    plt.figure(1,figsize=(75,15),dpi=96)
    plt.clf()
    if len(ecg_sig)>10*fs:
        ecg_sig = ecg_sig[:int(10*fs)]
    pixel_per_mm = np.int(np.floor(1 / mm_per_s * fs))

    max_point_time_count = int(mm_per_s * 10 * pixel_per_mm)
    max_point_amp_count = int(max_mv * mm_per_mv * pixel_per_mm)
    min_point_amp_count = int(min_mv * mm_per_mv * pixel_per_mm)

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
    plt.savefig(save_path)

if __name__ == "__main__":
    # record_dir = '/home/linjp/Documents/Projects/Physionet/preprocess/'
    # record_name = 'A00078'
    # record_path = record_dir + record_name + '.pickle'
    # with open(record_path,'rb') as fin:
    #     record = pickle.load(fin)
    # filtered_data = record['filtered_data']
    # save_path = './ecg.png'
    # ecg_sig = filtered_data[:3000]
    record_path = './A00001.mat'
    record = sio.loadmat(record_path,squeeze_me=True)
    sig = record['keypoint']['ecg'].tolist()
    seg_sig = sig[0:500]
    save_path = './ecg.jpg'
    plot_save_ecg(seg_sig,save_path, 300)