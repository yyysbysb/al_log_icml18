#!/usr/bin/env python3

import numpy as np
import math

def get_weight(model, dat, pred, eta):
    squared_norm = np.inner(dat.x, dat.x)
    exp = dat.w * eta * squared_norm
    if exp<1e-6:
        return 2 * (pred-dat.y) * dat.w * eta
    else:
        return (pred-dat.y) * (1.0 - np.exp(-2.0*exp))/squared_norm

def calc_gap(model, dat, cnt, eta):
    return abs(2*np.inner(model.w, dat.x) / (stepsize(cnt, eta)*np.inner(dat.x, dat.x)))
    
def stepsize(idx, stepsize_para0 = 0.05):
    return math.sqrt(stepsize_para0/(stepsize_para0+idx))

def gd(model, dat, idx, stepsize_para0 = 0.05):
    pred = model.predict(dat.x)
    model.w = model.w - get_weight(model, dat, pred, stepsize(idx, stepsize_para0)) * dat.x
