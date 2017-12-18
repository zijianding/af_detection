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
        hr[rec] = np.mean(rr_intervals[rec])

    return hr


def normal_identify(hr, low=50, high=110):
    normal_res = {}
    for rec in hr.keys():
        if (hr[rec] >= low) & (hr[rec] <= high):
            normal_res[rec] = 'N'
        else:
            normal_res[rec] = 'O'
    return normal_res

