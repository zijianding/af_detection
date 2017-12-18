#calculate the number of non-empty cells

"""
input: rr_interval, drr_interval
output: nec and normalized_nec
"""

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


