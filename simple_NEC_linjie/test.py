#test performance on validation set
val_data = []
record_set = set(rr_intervals.keys())
for i in range(0, len(partition['val'])):
    curr_sample = partition['val'][i][0]
    if curr_sample in record_set:
        val_data.append(partition['val'][i])

result, sensitivity, specificity = roc_curve(val_data, normalized_nec)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(1-specificity, sensitivity)
plt.show()

auc = np.trapz(sensitivity, 1-specificity)

#test performance on validation set

test_data = []
record_set = set(rr_intervals.keys())
for i in range(0, len(partition['test'])):
    curr_sample = partition['test'][i][0]
    if curr_sample in record_set:
        test_data.append(partition['test'][i])

result, sensitivity, specificity, threshold = roc_curve(test_data, normalized_nec)

final_thresh, sensitivity_level, specificity_level = find_threshold(sensitivity, specificity, threshold, sensi_thresh = 0.9)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(1-specificity, sensitivity)
plt.show()

auc = np.trapz(sensitivity, 1-specificity)


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
    acc = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    precison = tp / (tp+fp)
    return acc, sensitivity, specificity, precison

acc, sens, spec, prec = classify(test_data, normalized_nec, final_thresh)


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

