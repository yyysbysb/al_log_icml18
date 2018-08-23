#!/usr/bin/env python3

import math
import tarfile
import glob
import numpy as np

def mean(a):
    return sum(a)/len(a)

def stddev(a):
    if len(a)==1:
        return a[0]-a[0]
    m = mean(a)
    return np.sqrt(sum([(x-m)*(x-m) for x in a])*1.0/(len(a)-1))

def median(a):
    b = sorted(a)
    l = len(a)
    if l%2==0:
        return (b[l//2-1]+b[l//2])/2
    else:
        return b[l//2]

def my_float(s):
    try:
        return float(s)
    except ValueError:
        return 1e100

def my_dict_at(d, k):
    if d is None or k not in d:
        return None
    return d[k]

def backup_code(result_folder):
    with tarfile.open(result_folder+"code.tar", "w") as tar:
        for fn in glob.glob("*.py"):
            print(fn)
            tar.add(fn)
        for fn in glob.glob("AL-bandit/*.py"):
            print(fn)
            tar.add(fn)

