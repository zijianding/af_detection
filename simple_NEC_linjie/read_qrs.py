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


def read_signal(file_path, record_name):
	file = file_path + '/' + record_name
	signal = wfdb.rdsamp(recordname = file)
	hea = wfdb.rdheader(file)
	return signal, hea