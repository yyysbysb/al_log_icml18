#!/usr/bin/env python3

import random
import functools
import os.path
import numpy as np

class CExample(object):
    def __init__(self, x, y, w, z=1):
        self.x = x
        self.y = y
        self.w = w
        self.z = z
    def copy(self):
        return CExample(self.x, self.y, self.w, self.z)


class CDataSet(object):
    def __init__(self, all_data=None, train=None, test=None, log_data=None, online_data=None, r=None):
        self.all_data = []  if all_data is None else [x.copy() for x in all_data]
        self.train_data = [] if train is None else [x.copy() for x in train]
        self.test_data = [] if test is None else [x.copy() for x in test]
        self.log_data = None if log_data is None else [x.copy() for x in log_data]
        self.online_data = None if online_data is None else [x.copy() for x in online_data]

    def load_data(self, filename, handler):
        self.all_data = []        
        with open(filename) as file:
            if handler == data_parser_libsvm:
                self.all_data = data_parser_libsvm([line for line in file])
            else:
                self.all_data = [handler(line.strip().split(',')) for line in file]           
    
    def copy_all(self):
        return CDataSet(self.all_data, self.train_data, self.test_data)
    def copy(self):
        return CDataSet(self.all_data, self.train_data, self.test_data, self.log_data, self.online_data)

    def random_split(self, prop, r=random):
        self.train_data = [x for x in self.all_data]
        r.shuffle(self.train_data)
        cnt = int(len(self.all_data)*prop)
        self.test_data = self.train_data[cnt:]
        self.train_data = self.train_data[:cnt]

    def split_log(self, prop):
        cnt = int(len(self.train_data)*prop)
        self.log_data = self.train_data[:cnt]
        self.online_data = self.train_data[cnt:]

def to_binary_label(dataset, rule):
    return CDataSet([CExample(d.x, rule(d.y), d.w, d.z) for d in dataset.all_data if rule(d.y)!=0])

def normalize(dataset):
    lb = functools.reduce(np.minimum, [e.x for e in dataset.all_data])
    ub = functools.reduce(np.maximum, [e.x for e in dataset.all_data])
    mid = (lb+ub)/2
    diff = np.array([x if x>0 else 1 for x in ub-lb])
    return CDataSet([CExample((e.x-mid)/diff*2, e.y, e.w, e.z) for e in dataset.all_data])

def gen_synthetic_uniform(n, d):
    w = np.random.random(d) - np.random.random(d)
    X = np.random.random((n, d)) - np.random.random((n, d))
    return w, [CExample(x, (1 if np.inner(x,w)>=0 else -1)*(-1 if random.random()<0.1 else 1), 1, 1) for x in X]

def gen_synthetic_bandit(data, Q, r=random):
    prop = [Q(dp.x) for dp in data]
    tmp = [CExample(dp[0].x, dp[0].y, 1.0/dp[1], 1 if r.random()<dp[1] else 0) for dp in zip(data, prop)]
    return [CExample(tmp[i].x, tmp[i].y, tmp[i].w, i+1) for i in range(0, len(tmp)) if tmp[i].z==1]

def data_parser_rear(l):
    features = np.array([float(x) for x in l[:-1]])
    return CExample(features, l[-1], 1, 1)

def data_parser_front(l):
    features = np.array([float(x) for x in l[1:]])
    return CExample(features, l[0], 1, 1)

def data_parser_libsvm(ls):
    split_ls = [l.strip().split() for l in ls]
    num_features = max([max([0] + [int(e.split(":")[0]) for e in l[1:]]) for l in split_ls])
    examples = []
    for l in split_ls:
        f = [0]*num_features
        for e in l[1:]:
            idx, val = e.split(":")
            f[int(idx)-1] = float(val)
        examples.append(CExample(np.array(f), l[0].strip(), 1, 1))
    return examples

DATA_COLLECTION_PATHS = ["../data/", "../../data/", "../../../data/",]

LibsvmBinaryRule = lambda s: 1 if float(s) > 0.5 else -1

DatasetInfo = {"skin": ("skin.txt", data_parser_rear, lambda s: 1 if s=="1" else -1),\
    "magic": ("magic04.data", data_parser_rear, lambda s: 1 if s=="g" else -1),\
    "eeg": ("eeg.data", data_parser_rear, lambda s: 1 if s=="1" else -1),\
    "covtype": ("covtype.data", data_parser_rear, lambda s: 1 if s=="1" else (-1 if s=="2" else 0)),\
    "letter": ("letter.data", data_parser_front, lambda s: 1 if s=="U" else (-1 if s=="P" else 0)),\
    "a9a": ("a9a.txt", data_parser_libsvm, LibsvmBinaryRule),\
    "a5a": ("a5a", data_parser_libsvm, LibsvmBinaryRule),\
    "cod-rna": ("cod-rna.txt", data_parser_libsvm, LibsvmBinaryRule),\
    "german": ("german.numer_scale", data_parser_libsvm, LibsvmBinaryRule),\
    "ijcnn1": ("ijcnn1.tr", data_parser_libsvm, LibsvmBinaryRule),\
    "mushrooms": ("mushrooms.txt", data_parser_libsvm, lambda s: 1 if int(s)==1 else -1),\
    "phishing": ("phishing.txt", data_parser_libsvm, LibsvmBinaryRule),\
    "splice": ("splice.t", data_parser_libsvm, LibsvmBinaryRule),\
    "svmguide1": ("svmguide1.t", data_parser_libsvm, LibsvmBinaryRule),\
    "w7a": ("w7a", data_parser_libsvm, LibsvmBinaryRule),}

def load_data(dataset_name, max_sz = None):
    if dataset_name == "synthetic":
        return CDataSet(gen_synthetic_uniform(6000, 30)[1])
    if dataset_name not in DatasetInfo:
        print("dataset " + dataset_name +" is unknown")
        return None
    dataset_path = None
    info = DatasetInfo[dataset_name]
    for path in DATA_COLLECTION_PATHS:
        #print(path+"/"+dataset_name)
        if os.path.isfile(path+"/"+info[0]):
            dataset_path = path+"/"+info[0]            
            break
    if dataset_path is None:
        print("data file for " + dataset_name +" does not exist")
        return None

    dataset = CDataSet()
    dataset.load_data(dataset_path, info[1])
    dataset = normalize(to_binary_label(dataset, info[2]))
    if max_sz != None:
        random.shuffle(dataset.all_data)
        dataset = CDataSet(dataset.all_data[:max_sz])
    return dataset
