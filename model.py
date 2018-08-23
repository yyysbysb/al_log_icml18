#!/usr/bin/env python3

import numpy as np

class CModel(object):
    def __init__(self, w=None):
        self.w = w
    
    def predict(self, x):
        ret = np.inner(self.w, x)
        if ret>1: ret = 1
        if ret<-1: ret = -1
        return ret

def evaluate(model, data):
    if data is None or len(data)==0:
        return -1
    else:
        err= sum([1 if model.predict(dp.x)*dp.y<=0 else 0 for dp in data])
        return err/len(data)

class CLearning(object):
    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters.copy()
        self.cnt_labels = 0