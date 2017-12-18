#############read_qrs.py##############
import pandas

file_path = '/home/dingzj/workspace/ECG/data/training2017/'

record_file = file_path + 'RECORDS'
qrs_suffix = '_qrs.txt'

records = pandas.read_table(record_file, sep='\t', header=None)

qrs_data = {}
exclude_list = set(['A03549','A04735','A05434','A07183'])
for i in range(0,records.shape[0]):
    if records[0][i] in exclude_list:
        continue
    else:
        curr_qrs_file = file_path + records[0][i] + qrs_suffix
        curr_data = pandas.read_table(curr_qrs_file, sep='\s+', header=None)
        qrs_data[records[0][i]] = curr_data

############rr_dr.py#################
import numpy as np
#import pandas

def time2sec(qrs_time):
    qrs_time = qrs_time.strip('[')
    qrs_time = qrs_time.strip(']')
    x = qrs_time.split(':')
    y = np.zeros(len(x))
    for i in range(0,len(x)):
        y[i] = float(x[i])
    qrs_sec = y[0]*3600 + y[1]*60 + y[2]
    return qrs_sec


rr_intervals = {}
rr_intervals_thresh = 32 #### filter records with less than 32 intervals
rr_method = 'sub' ### use 'all' rr intervals or 'sub'-part of the first rr_intervals_thresh intervals
for record in qrs_data.keys():
    curr_dat = qrs_data[record]
    x = curr_dat[0]
    qrs_time = np.zeros(len(x))
    for i in range(0, len(x)):
        qrs_time[i] = time2sec(x[i])
    if (len(x)-1 > rr_intervals_thresh) & (rr_method == 'sub'): ###filter records with less than 32 intervals
        curr_rr = np.zeros(rr_intervals_thresh) ### only need the first 32 RR intervals
        for i in range(1, rr_intervals_thresh+1): ### only need the first 32 RR intervals
            curr_rr[i - 1] = qrs_time[i] - qrs_time[i - 1]
        rr_intervals[record] = curr_rr
    elif (len(x) - 1 > rr_intervals_thresh) & (rr_method == 'all'):
        curr_rr = np.zeros(len(x)-1)
        for i in range(1, len(x)):
            curr_rr[i-1] = qrs_time[i] - qrs_time[i - 1]
        rr_intervals[record] = curr_rr
    else:
        exclude_list.add(record)

drr_intervals = {}
for record in rr_intervals.keys():
    new_arr = np.zeros(len(rr_intervals[record])-1)
    for i in range( 1, len(rr_intervals[record]) ):
        new_arr[i-1] = rr_intervals[record][i] - rr_intervals[record][i-1]
    drr_intervals[record] = new_arr


#############NEC_count.py#################
def nec_count(rr, drr, gap=0.025):
    rr = np.delete(rr, 0, axis=0) # same counts as drr
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


def NEC_info(rr_intervals, drr_intervals):
    nec = {}
    normalized_nec = {}

    records = rr_intervals.keys()
    for rec in records:
        rr = rr_intervals[rec]
        drr = drr_intervals[rec]
        curr_nec, curr_norm_nec = nec_count(rr, drr)
        nec[rec] = curr_nec
        normalized_nec[rec] = curr_norm_nec

    return nec, normalized_nec


nec, normalized_nec = NEC_info(rr_intervals, drr_intervals)


###########classify.python.py############
#read partition
import pickle
partition_file = open("/home/dingzj/workspace/ECG/data/record_patition.pickle",'rb')
partition = pickle.load(partition_file)
partition_file.close()

#filter train sample, keep intervals >= rr_intervals_thresh
train_data = []
record_set = set(rr_intervals.keys())
for i in range(0, len(partition['train'])):
    curr_sample = partition['train'][i][0]
    if curr_sample in record_set:
        train_data.append(partition['train'][i])


def roc_curve( train_data, normalized_nec ):
    train_sample = train_data

    m = len(train_sample)
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
        for j in range(0, len(train_sample)):
            curr_sample = train_sample[j][0]
            curr_label = train_sample[j][1]
            if curr_sample in exclude_list:
                if curr_label == 'A':
                    fn += 1
                else:
                    tn += 1
            else:
                if normalized_nec[curr_sample] > curr_thresh:
                    #predicted 'A'
                    if curr_label == 'A':
                        tp += 1
                    else:
                        fp += 1
                else:
                    #predictied non-'A'
                    if curr_label != 'A':
                        tn += 1
                    else:
                        fn += 1

        res[i, 0] = tp
        res[i, 1] = fp
        res[i, 2] = tn
        res[i, 3] = fn

        sensitivity[i] = tp / (tp + fn)
        specificity[i] = tn / (fp + tn)

    return res, sensitivity, specificity,  gaps


def roc_curve_4normal( train_data, normalized_nec ):
    train_sample = train_data

    m = len(train_sample)
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
        for j in range(0, len(train_sample)):
            curr_sample = train_sample[j][0]
            curr_label = train_sample[j][1]
            if curr_sample in exclude_list:
                if curr_label == 'N':
                    fn += 1.
                else:
                    tn += 1.
            else:
                if normalized_nec[curr_sample] < curr_thresh:
                    #predicted 'N'
                    if curr_label == 'N':
                        tp += 1.
                    else:
                        fp += 1.
                else:
                    #predictied non-'N'
                    if curr_label != 'N':
                        tn += 1.
                    else:
                        fn += 1.

        res[i, 0] = tp
        res[i, 1] = fp
        res[i, 2] = tn
        res[i, 3] = fn

        sensitivity[i] = tp / (tp + fn)
        specificity[i] = tn / (fp + tn)

    return res, sensitivity, specificity,  gaps



result, sensitivity, specificity, threshold = roc_curve(train_data, normalized_nec)

result, sensitivity, specificity, threshold = roc_curve_4normal(train_data, normalized_nec)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(1-specificity, sensitivity)
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.title('ROC curve for Train Data')
plt.show()


auc = np.trapz(sensitivity, 1-specificity)

##choose a threshold
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

final_thresh, sensitivity_level, specificity_level,  index = find_threshold(sensitivity, specificity, threshold)

'''
final_thresh, sensitivity_level, specificity_level,  index = find_threshold(sensitivity, specificity, threshold,
                                                                            sensi_thresh = 0.9, speci_thresh=0.55)

final_thresh, sensitivity_level, specificity_level, index = find_threshold(sensitivity, specificity,
                                                                           threshold, sensi_thresh=0.9)

final_thresh, sensitivity_level, specificity_level, index = find_threshold(sensitivity, specificity,
                                                                           threshold, speci_thresh=0.8)
'''

precision_level = result[index][0] / (result[index][0] + result[index][1])
acc_level = (result[index][0] + result[index][2]) / (np.sum(result[index]))

########################on test data########################
def classify(test_data, normalized_nec, final_thresh):
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    for j in range(0, len(test_data)):
        curr_sample = test_data[j][0]
        curr_label = test_data[j][1]
        if normalized_nec[curr_sample] > final_thresh:
            # predicted 'A'
            if curr_label == 'A':
                tp += 1.
            else:
                fp += 1.
        else:
            # predictied non-'A'
            if curr_label != 'A':
                tn += 1.
            else:
                fn += 1.


    classify_result = {}

    classify_result['accuracy'] = (tp+tn)/(tp+tn+fp+fn)
    classify_result['sensitivity'] = tp / (tp+fn)
    classify_result['specificity'] = tn / (tn+fp)
    classify_result['precision'] = tp / (tp+fp)
    classify_result['tp'] = tp
    classify_result['fp'] = fp
    classify_result['fn'] = fn
    classify_result['tn'] = tn

    return classify_result


val_data = []
record_set = set(rr_intervals.keys())
for i in range(0, len(partition['val'])):
    curr_sample = partition['val'][i][0]
    if curr_sample in record_set:
        val_data.append(partition['val'][i])

val_classification = classify(val_data, normalized_nec, final_thresh)


test_data = []
record_set = set(rr_intervals.keys())
for i in range(0, len(partition['test'])):
    curr_sample = partition['test'][i][0]
    if curr_sample in record_set:
        test_data.append(partition['test'][i])



test_classification = classify(test_data, normalized_nec, final_thresh)


def count_af_sample(data):
    count = 0
    for i in range(0, len(data)):
        if data[i][1] == 'A':
            count += 1
    return count


def hr_calc(rr_intervals):
    '''
    calculate the heart rate based on average QRS
    :param
    rr_intervals
    :return:
    hr = 60/mean(rr) + 1
    '''
    hr = {}
    for rec in rr_intervals.keys():
        hr[rec] = 60 / np.mean(rr_intervals[rec])

    return hr


def normal_identify(hr, low=50, high=110):
    normal_res = {}
    for rec in hr.keys():
        if (hr[rec] >= 50) & (hr[rec] <= 1100):
            normal_res[rec] = 'N'
        else:
            normal_res[rec] = 'O'
    return normal_res

def classify_result(test_data, normalized_nec, final_thresh, hr, low_hr, high_hr ):
    class_res = {}
    true_normal = 0.
    false_normal = 0.
    true_other = 0.
    false_other = 0.
    for j in range(0, len(test_data)):
        curr_sample = test_data[j][0]
        curr_label = test_data[j][1]
        if normalized_nec[curr_sample] > final_thresh:
            # predicted 'A'
            class_res[curr_sample] = 'A'
        else:
            # predictied non-'A'
            if (hr[curr_sample] >= low_hr) & (hr[curr_sample] <= high_hr):
                class_res[curr_sample] = 'N'
                if curr_label == 'N':
                    true_normal += 1.
                else:
                    false_normal += 1.
            else:
                class_res[curr_sample] = 'O'
                if curr_label == 'O':
                    true_other += 1.
                else:
                    false_other += 1.

    sensitivity = true_normal / (true_normal + false_other)
    specificity = true_other / (true_other + false_normal)
    return class_res, sensitivity, specificity


class_res, normal_sen, normal_spec = classify_result(test_data=test_data, normalized_nec=normalized_nec,
                                                     final_thresh=final_thresh,
                                                     hr=hr_calc(rr_intervals), low_hr=50, high_hr=110)

