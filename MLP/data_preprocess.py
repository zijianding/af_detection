import numpy as np
import scipy.io as sio
import wfdb
import pandas


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
 

if __name__ = 'main':
    #read all text records
     path = '/home/dingzj/workspace/ECG/data'
     records = pandas.read_table(path + '/training2017/RECORDS', header=None) #records[0][0], [0][1], a col
     labels = pandas.read_csv(path + '/REFERENCE-v3.csv', header=None) # labels[0][0] and [1][0], a row
     
     samples = read_all_signal(path + '/training2017', records)