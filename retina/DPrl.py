import numpy as np

import pandas as pd



def get_load(id, timewindow, data):
    """Get current load, reqs per second
    args:
        id: the current request id
        timewindow: the time window to estimate current load
        data: the record data of reqs coming timestamp
    returns:
        current load
    """
    req = data.iloc[id]
    req_start = req['start']
    idx = id - 1
    cnt = 1
    while idx >= 0:
        reqi_start = data.iloc[idx]['start']
        if reqi_start > req_start - 6000:
            cnt += 1
            idx -= 1
        else:
            break
    return cnt


def 




