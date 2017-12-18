#calculate the RR intervals and dRR intervals for each sample

"""
input data: qrs_data, a dict
output data: rr interval, drr interval

"""

import numpy as np
import pandas

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
drr_intervals = {}


for record in qrs_data.keys():
    curr_dat = qrs_data[record]
    x = curr_dat[0]
    qrs_time = np.zeros(len(x))
    for i in range(0,len(x)):
        qrs_time[i] = time2sec(x[i])
    if len(x)-1 > 1:
        curr_rr = np.zeros(len(x) - 1)
        for i in range(1, len(x)):
            curr_rr[i - 1] = qrs_time[i] - qrs_time[i - 1]
        rr_intervals[record] = curr_rr
    else:
        exclude_list.add(record)


for record in rr_intervals.keys():
    new_arr = np.zeros(len(rr_intervals[record])-1)
    for i in range( 1, len(rr_intervals[record]) ):
        new_arr[i-1] = rr_intervals[record][i] - rr_intervals[record][i-1]
    drr_intervals[record] = new_arr



