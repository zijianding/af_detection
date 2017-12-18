
nec, normalized_nec = NEC_info(rr_intervals, drr_intervals)

#read partition
import pickle
partition_file = open("/home/dingzj/workspace/ECG/data/record_patition.pickle",'rb')
partition = pickle.load(partition_file)
partition_file.close()

#read labels
import pandas
labels = pandas.read_csv("/home/dingzj/workspace/ECG/data/REFERENCE-v3.csv", header=None)
    labels_dict = {}
    for i in range(0, len(labels[0])):
        labels_dict[labels[0][i]] = labels[1][i]


def roc_curve( partition, normalized_nec ):
    train_sample = partition['train']
    m = len(train_sample)
    gap = 1 / float(m)

    res = np.zeros((m+1,4)) #tp, fp , tn, fn
    sensitivity = np.zeros((m+1))
    specificity = np.zeros((m+1))
    for i in range(0, m+1):
        curr_thresh = i * gap

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

    return res, sensitivity, specificity



result, sensitivity, specificity = roc_curve(partition, normalized_nec)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(1-specificity, sensitivity)
plt.show()


auc = np.trapz(sensitivity, 1-specificity)
