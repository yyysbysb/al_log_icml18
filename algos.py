#!/usr/bin/env python3

import math
import numpy as np
import logger as MLogger
import data as MData
import model as MModel
import opt
import experiments

def batch_train(learning, data, tot_cnt, idx=0):
    assert isinstance(learning, MModel.CLearning)
    if len(data)==0:
        return learning, 0
    sum_loss = 0.0
    if idx==0: 
        learning.model.w = np.zeros(data[0].x.shape)
    opt_update = opt.gd
    data = [MData.CExample(e.x, e.y, e.w, learning.random.randint(idx+1, idx+tot_cnt)) for e in data if e.z>0]
    data = sorted(data, key=lambda e: e.z)

    for example in data:        
        if example.z==0: 
            continue
        else:
            idx += example.w
        if learning.model.predict(example.x)*example.y<=0:
            sum_loss += example.w
        opt_update(learning.model, example, idx, learning.parameters.learning_rate)
    sum_loss = sum([e.w for e in data if e.z>0 and learning.model.predict(e.x)*e.y<=0])
    return learning, sum_loss

def generic_batch_learning(dataset, logger, data_paras, model_paras, digest):
    assert isinstance(dataset, MData.CDataSet)
    assert isinstance(logger, MLogger.CLogger)
    assert isinstance(data_paras, experiments.CDataParameters)
    assert isinstance(model_paras, experiments.CModelParameters)

    model = MModel.CModel()
    model.w = np.zeros(dataset.all_data[0].x.shape)
    learning = MModel.CLearning(model, model_paras)
    learning.random = dataset.random

    learning, sum_loss = batch_train(learning, dataset.log_data, data_paras.cnt_log)
    data_batches = [dataset.log_data]
    cur_online = 0
    cur_sz = model_paras.batch_sz

    logger.on_start(learning, dataset)
    logger.check_and_log(learning, dataset, cur_online)
    while cur_online < len(dataset.online_data)\
    and (model_paras.label_budget is None or learning.cnt_labels<=model_paras.label_budget*2):
        next_batch = [e for e in dataset.online_data[cur_online:cur_online+cur_sz]]
        cur_online += len(next_batch)
        data_batches.append(next_batch)
        cur_dataset, next_batch = digest(learning, data_paras, data_batches, sum_loss)
        data_batches[-1] = next_batch
        learning, sum_loss = batch_train(learning, cur_dataset, data_paras.cnt_log+cur_online)
        cur_sz = int(cur_sz*model_paras.batch_rate)
        logger.check_and_log(learning, dataset, cur_online)

    logger.on_stop(learning, dataset)
    return learning

def passive_digest(learning, data_paras, data_batches, sum_loss):
    learning.cnt_labels += len(data_batches[-1])
    return [e for b in data_batches for e in b], data_batches[-1]
def passive_batch(dataset, logger, data_paras, model_paras):
    return generic_batch_learning(dataset, logger, data_paras, model_paras, passive_digest)

def MIS_transform(batches, data_paras, Qks=None):
    n = [len(b) for b in batches]
    n[0] = data_paras.cnt_log
    if Qks is None:
        s = sum(n[1:])
        return [MData.CExample(e.x, e.y, (n[0]+s)/(n[0]*data_paras.Q0(e.x)+s), e.z) \
            for b in batches for e in b]
    else:
        s = sum(n)
        return [MData.CExample(e.x, e.y, s/sum([nq[0]*nq[1](e.x) for nq in zip(n, Qks)]), e.z) \
            for b in batches for e in b]

def passive_MIS_digest(learning, data_paras, data_batches, sum_loss):
    learning.cnt_labels += len(data_batches[-1])
    return MIS_transform(data_batches, data_paras), data_batches[-1]
def passive_MIS_batch(dataset, logger, data_paras, model_paras):
    return generic_batch_learning(dataset, logger, data_paras, model_paras, passive_MIS_digest)

def get_threshold(sum_loss, c0, t):
    ret = math.sqrt(c0*sum_loss/t/t) + c0*math.log(t)/t
    return ret

def test_dis(sum_loss, c0, cnt, model, cur_data, eta, idx=None):
    if idx is None:
        idx = cnt
    if cnt<=3: return True
    gap = opt.calc_gap(model, cur_data, idx, eta)
    threshold = get_threshold(sum_loss, c0, cnt)
    return gap/cnt<=threshold

def DBAL_IS_digest(learning, data_paras, data_batches, sum_loss):
    log = [e for e in data_batches[0]]
    online = [e for b in data_batches[1:-1] for e in b]
    next_batch = [e.copy() for e in data_batches[-1]]
    n0, n1 = data_paras.cnt_log, len(online)
    eta = learning.parameters.learning_rate
    for e in next_batch:
        if test_dis(sum_loss, learning.parameters.c0/data_paras.xi0, n0+n1, learning.model, e, eta):
            learning.cnt_labels += 1
        else:
            e.y = 1 if learning.model.predict(e.x)>=0 else -1
    return log+online+next_batch, next_batch
def DBAL_IS_batch(dataset, logger, data_paras, model_paras):
    return generic_batch_learning(dataset, logger, data_paras, model_paras, DBAL_IS_digest)

def DBAL_MIS_digest(learning, data_paras, data_batches, sum_loss):
    n1 = sum([len(b) for b in data_batches[1:-1]])
    wmaxk = (data_paras.cnt_log+n1)/(data_paras.cnt_log*data_paras.xi0 + n1)
    tmp_data_paras = data_paras.copy()
    tmp_data_paras.xi0 = 1.0/wmaxk
    ds, next_batch = DBAL_IS_digest(learning, tmp_data_paras, data_batches, sum_loss)
    return MIS_transform(data_batches[:-1]+[next_batch], tmp_data_paras), next_batch
def DBAL_MIS_batch(dataset, logger, data_paras, model_paras):
    return generic_batch_learning(dataset, logger, data_paras, model_paras, DBAL_MIS_digest)

def IDBAL(dataset, logger, data_paras, model_paras):
    assert isinstance(dataset, MData.CDataSet)
    assert isinstance(logger, MLogger.CLogger)
    assert isinstance(data_paras, experiments.CDataParameters)
    assert isinstance(model_paras, experiments.CModelParameters)

    model = MModel.CModel()
    model.w = np.zeros(dataset.all_data[0].x.shape)
    learning = MModel.CLearning(model, model_paras)
    learning.random = dataset.random

    last_tot_cnt = idx = int(data_paras.cnt_log*model_paras.init_log_prop)
    tmp_len = 0
    while tmp_len<len(dataset.log_data) and dataset.log_data[tmp_len].z<last_tot_cnt:
        tmp_len += 1
    learning, sum_loss = batch_train(learning, dataset.log_data[:tmp_len], last_tot_cnt)
    opt_idx = int(sum([e.w for e in dataset.log_data[:tmp_len] if e.z>0]))
    alpha = data_paras.cnt_log * (1.0-model_paras.init_log_prop)/len(dataset.online_data)
    cur_online = 0
    cur_log_cnt = idx
    cur_log_idx = tmp_len
    cur_k = model_paras.batch_sz
    train_data = [MData.CExample(e.x, e.y, 1.0/data_paras.Q0(e.x), e.z) for e in dataset.train_data]

    xi = data_paras.xi0
    wmaxk = 1/xi
    logger.on_start(learning, dataset)
    xis = [xi]
    sum_online_z = 0
    while cur_online < len(dataset.online_data):
        cur_log_batch = []
        while cur_log_idx < len(dataset.log_data) and dataset.log_data[cur_log_idx].z <= cur_log_cnt + int(cur_k * alpha):
            e = dataset.log_data[cur_log_idx]
            cur_log_batch.append(e)
            cur_log_idx += 1
        eta = model_paras.learning_rate

        xi_next = min([1.0/e.w for e in train_data[:int((1+alpha)*cur_k)]\
                if test_dis(sum_loss, model_paras.c0*wmaxk, last_tot_cnt, learning.model, e, eta, opt_idx)]+[1])
        wmaxk_next = (alpha+1)/(alpha*xi_next + 1)
        Qk = lambda x: 1 if data_paras.Q0(x)<=xi+1/alpha else 0

        if len(cur_log_batch)!=0:
            cur_log_cnt = cur_log_batch[-1].z
            cur_log_batch = [MData.CExample(e.x, e.y, (1.0+alpha)/(alpha/e.w+Qk(e.x)), e.z) \
                for e in cur_log_batch]
        
        cur_online_batch = []        
        for tmp in dataset.online_data[cur_online:cur_online+cur_k]:
            cur_data = MData.CExample(tmp.x, tmp.y, (1.0+alpha)/(alpha*data_paras.Q0(tmp.x)+Qk(tmp.x)), 1)
            
            if Qk(cur_data.x)==0:
                cur_data.z = 0
            else:
                sum_online_z += 1
                if test_dis(sum_loss, model_paras.c0*wmaxk, last_tot_cnt, learning.model, cur_data, eta, opt_idx):
                    learning.cnt_labels += 1
                else:
                    cur_data.y = 1 if learning.model.predict(cur_data.x)>=0 else -1
            cur_online_batch.append(cur_data)
        
        last_tot_cnt = int(len(cur_online_batch)*(1+alpha))
        learning, sum_loss = batch_train(learning, cur_log_batch + cur_online_batch, \
            last_tot_cnt, opt_idx)
        opt_idx = int(opt_idx+sum([e.w for e in cur_log_batch if e.z>0])+sum([e.w for e in cur_online_batch if e.z>0]))
        idx += last_tot_cnt
        cur_online += len(cur_online_batch)
        cur_k = int(cur_k*model_paras.batch_rate)
        xi = xi_next
        wmaxk = wmaxk_next
        xis.append(xi)
    logger.log_misc_info("["+",".join([str((sum_online_z+1)/(cur_online+1))]+[str(x) for x in xis])+"]")
    logger.on_stop(learning, dataset)
    return learning
