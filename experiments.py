#!/usr/bin/env python3

import multiprocessing as mp
import time
import random
import math
import enum
import numpy as np
import logger
import data
import model as MModel
import algos as bl
import utils

class CDataParameters(object):
    def __init__(self, prop_train, prop_log, cnt_log, Q0, xi0):
        self.prop_train = prop_train
        self.prop_log = prop_log
        self.cnt_log = cnt_log
        self.Q0 = Q0
        self.xi0 = xi0
    def copy(self):
        return CDataParameters(self.prop_train, self.prop_log, self.cnt_log, self.Q0, self.xi0)

class CModelParameters(object):
    def __init__(self, c0, learning_rate=0.05, init_log_prop=None, \
                batch_sz=None, batch_rate=None, label_budget=None):
        self.c0 = c0
        self.learning_rate = learning_rate
        self.init_log_prop = init_log_prop
        self.batch_sz = batch_sz
        self.batch_rate = batch_rate
        self.label_budget = label_budget
    def copy(self):
        return CModelParameters(self.c0, self.learning_rate, self.init_log_prop, self.batch_sz, \
            self.batch_rate, self.label_budget)

def run_batch(dataset, algo, log, data_paras, model_paras, init_sz, rate):
    if algo != bl.IDBAL:
        log.set_cond_log(init_sz, rate)
        algo(dataset, log, data_paras, model_paras)
    else:
        all_online = dataset.online_data
        dataset.online_data = []
        bl.passive_batch(dataset, log, data_paras, model_paras)
        online_sz = init_sz
        while online_sz <= len(all_online):
            dataset.online_data = all_online[:online_sz]
            learning = algo(dataset, log, data_paras, model_paras)
            if model_paras.label_budget is not None and learning.cnt_labels>model_paras.label_budget*2:
                break
            if online_sz<len(all_online) and online_sz*rate>=len(all_online):
                online_sz = len(all_online)
            else:
                online_sz = int(online_sz * rate)

q0 = 0.005
GDataParasDict = {\
    "_find_logsize": CDataParameters(0.8, 0, -1, None, 1),\
    "Identical": CDataParameters(0.8, 0.5, -1, None, q0),\
    "Uniform": CDataParameters(0.8, 0.5, -1, None, q0),\
    "Uncertainty": CDataParameters(0.8, 0.5, -1, None, q0),\
    "Certainty": CDataParameters(0.8, 0.5, -1, None, q0),\
    }

def gen_dist_policy(dataset, pos_corr_flag):
    model = MModel.CModel()
    d = dataset.train_data[0].x.shape[0]
    model.w = np.zeros(d)
    learn = MModel.CLearning(model, CModelParameters(0, 1))
    learn.random = random.Random()
    l = len(dataset.train_data)//10
    learn, sum_loss = bl.batch_train(learn, dataset.train_data[:l], l)
    w = learn.model.w
    w = w / math.sqrt(np.inner(w, w))
    dst = sorted([np.square(np.inner(e.x, w)) for e in dataset.train_data[-400:]])
    threshold = len(dst)//3
    if pos_corr_flag:
        c = 0.1/(np.square(dst[-threshold])/math.sqrt(d))
        return lambda x: max(q0, min(1,c*(np.power(np.inner(x, w), 4)/math.sqrt(d))))
    else:
        c = math.log(0.1)/((dst[threshold])/d)      
        return lambda x: max(q0, math.exp(c*np.square(np.inner(x, w))/d))

def gen_uniform_policy(dataset):
    d = dataset.all_data[0].x.shape[0]
    w = np.random.random(d)
    dist = [q0, 0.05, 0.5]
    return lambda x: dist[int(np.inner(x,w)*100000)%3]

def gen_policy(dataset, data_paras_name):
    if data_paras_name == "Identical":
        return lambda x:q0
    elif data_paras_name == "Uniform":
        return gen_uniform_policy(dataset)
    elif data_paras_name == "Uncertainty":
        return gen_dist_policy(dataset, False)
    elif data_paras_name == "Certainty":
        return gen_dist_policy(dataset, True)
    elif data_paras_name == "_find_logsize":
        return lambda x:1

def run_once(i, dataset, algo, log, data_paras_name, model_paras, batch_sz, rate, r, extra_paras):
    dataset_cp = dataset.copy()
    dataset_cp.random = r
    data_paras = GDataParasDict[data_paras_name].copy()
    if extra_paras is not None:
        data_paras.prop_log = extra_paras["prop_log"]
    dataset_cp.random_split(data_paras.prop_train, r)
    data_paras.Q0 = gen_policy(dataset_cp, data_paras_name)   
    dataset_cp.split_log(data_paras.prop_log)
    data_paras.cnt_log = len(dataset_cp.log_data)
    dataset_cp.log_data = data.gen_synthetic_bandit(dataset_cp.log_data, data_paras.Q0, r)
    run_batch(dataset_cp, algo, log, data_paras, model_paras, batch_sz, rate)
    return log

def run_multiple(dataset, algo, log, num_iter, data_paras_name, model_paras, batch_sz, rate, \
        num_processes=1, extra_paras=None):
    tmp_loggers = [logger.CLogger(log.info) for i in range(0, num_iter)]
    if num_processes==1:
        for i in range(0, num_iter):    
            run_once(i, dataset, algo, tmp_loggers[i], data_paras_name, model_paras, batch_sz, rate, \
                random.Random(random.random()), extra_paras)            
    else:                 
        with mp.Pool(processes=num_processes) as pool:    
            res = [pool.apply_async(run_once, [i, dataset, algo, tmp_loggers[i], data_paras_name, \
                                               model_paras, batch_sz, rate, random.Random(random.random()), extra_paras])\
                    for i in range(0, num_iter)]
            tmp_loggers = [r.get() for r in res]
    log.init_by_merge(tmp_loggers)

def get_max_cnt_label(dataset, data_paras_name, label_budget=None):
    data_paras = GDataParasDict[data_paras_name]
    len_online = int(len(dataset.all_data)*data_paras.prop_train*(1-data_paras.prop_log))
    if label_budget is None or len_online<label_budget:
        return len_online
    else:
        return label_budget

def tune_batch(info, dataset, algo, num_iter, data_paras_name, model_paras_set, batch_sz, rate, \
        num_processes=1, log_file=None, return_log=False, extra_paras=None):
    print("start tuning for ", info)
    best_auc = 1e10
    loggers = [logger.CLogger(str(m.c0)) for m in model_paras_set]
    for ml in zip(model_paras_set, loggers):
        run_multiple(dataset, algo, ml[1], num_iter, data_paras_name, ml[0], batch_sz, rate, num_processes, extra_paras=extra_paras)
        ml[1].get_stat(max_cnt_label=ml[0].label_budget)
        
        if log_file != None:
            log_info = info + " " + str(ml[0].c0) + " " + str(ml[0].learning_rate)
        else:
            log_info = None
        auc = logger.calc_metric(ml[1], get_max_cnt_label(dataset, data_paras_name, ml[0].label_budget), \
                log_file=log_file, log_info=log_info, extra_tail=utils.my_dict_at(extra_paras, "extra_tail"))
        
        print(ml[0].c0, " ", ml[0].learning_rate, " ", auc)
        if auc<best_auc:
            best_auc = auc
            best_para = ml[0]
            best_log = ml[1]
    #logger.plot_err(loggers)
    if return_log:
        return best_log
    else:
        return best_para

GAlgoDict = {"passive_is": (bl.passive_batch, "Passive"), \
        "passive_mis": (bl.passive_MIS_batch, "passive_mis"),\
        "dbal_is": (bl.DBAL_IS_batch, "DBALw"),\
        "dbal_mis": (bl.DBAL_MIS_batch, "DBALwm"),\
        "idbal_mis": (bl.IDBAL, "IDBAL"),\
    }

def choose_prop_log(dataset, dataset_name, data_paras_name, log_file, debug):
    if debug:
        learning_rates = [0.05]
        num_processes = 1
        num_iter = 1
    else:
        learning_rates = [0.005, 0.02, 0.1, 0.5, 1, 2]
        num_processes = mp.cpu_count()
        num_iter = 4
    batch_sz = int(len(dataset.all_data) * 0.8 / 20)
    paraset = [CModelParameters(0, lr, 0, batch_sz, 1, None) for lr in learning_rates]
    l = tune_batch("%s %s %s"%(data_paras_name, dataset_name, "Choose Prop Log"), \
            dataset, bl.passive_batch, num_iter, "_find_logsize", paraset, batch_sz, 1,\
            num_processes, None, return_log=True)
    
    assert isinstance(l, logger.CLogger) 
    al_start_err = l.err_stat[0][-1] + (l.err_stat[0][1]-l.err_stat[0][-1])*0.5
    al_end_err = l.err_stat[0][-1] + (l.err_stat[0][1]-l.err_stat[0][-1])*0.05
    idx = 1
    while idx<len(l.err_stat[0]):
        if utils.mean(l.err_stat[0][idx-1: idx+2])<=al_start_err and l.err_stat[0][idx]<=al_start_err:
            break
        idx+=1
    if idx==len(l.err_stat[0]): idx -=1
    target_log_cnt = l.label_stat[0][idx]

    while idx<len(l.err_stat[0]):
        if utils.mean(l.err_stat[0][idx-1: idx+2])<=al_end_err and l.err_stat[0][idx]<=al_end_err:
            break
        idx+=1
    if idx==len(l.err_stat[0]): idx -=1
    target_al_cnt = l.label_stat[0][idx]

    r=random.Random(571428)
    dataset_cp = dataset.copy_all()
    dataset_cp.random = r
    dataset_cp.random_split(0.8, r)
    p = gen_policy(dataset_cp, data_paras_name)
    sz, e_cnt = 0, 0
    while sz<len(dataset_cp.train_data)*0.6:
        e_cnt += p(dataset_cp.train_data[sz].x)
        if e_cnt>target_log_cnt:
            break 
        sz+=1
    prop_log = max(0.1, sz/len(dataset_cp.train_data))
    label_budget=max((target_al_cnt-target_log_cnt)//5, 100)
    if log_file is not None:
        log_file.write("prop_log: [(%.2f, %.2f, %.2f, %.2f), (%.3f, %.3f)]\n"\
            %(prop_log, sz, target_log_cnt, label_budget, al_start_err, al_end_err))
    print("prop_log: %.2f\tcnt_log: %2f\teffective_log: %.2f\tbudget: %.2f\terr: %.3f, %.3f"\
            %(prop_log, sz, e_cnt, label_budget, al_start_err, al_end_err))
    return prop_log, label_budget

def run_experiments(dataset, data_paras_name, dataset_name, algos, output_filename, \
            batch_sz, batch_rate,\
            auclogger=None, log_file=None, label_budget=None, debug=False):
    if dataset is None:
        return
    print("[%s] Experiment for %s"%(time.asctime(), output_filename))
        
    c0s = [0.01 * (1<<(2*i)) for i in range(0, 9)]
    learning_rates = [0.0001 * (1<<(2*i)) for i in range(0, 9)] + [3, 10]

    if debug:
        num_process = 1
        tune_iter = 1
        final_iter = 4
        c0s = [1]
        learning_rates = [0.1]
    else:
        num_process = mp.cpu_count()
        tune_iter = 16
        final_iter = 16

    prop_log, _label_budget = choose_prop_log(dataset, dataset_name, data_paras_name, log_file, debug)
    if label_budget==-1:
        label_budget = _label_budget
    extra_paras = {"prop_log": prop_log, "extra_tail": 1, "label_budget": label_budget,\
        "batch_rate": batch_rate}
    if log_file != None:
        log_file.write("EXTRA_PARAS: %s\n"%str(extra_paras))

    para_set_lr = [CModelParameters(0, lr, 1/3, 10, batch_rate, label_budget) for lr in learning_rates]
    para_set_c0lr = [CModelParameters(c0, lr, 1/3, 10, batch_rate, label_budget) for c0 in c0s for lr in learning_rates]  
        
    #tests = [(GAlgoDict[a][0], logger.CLogger(a), None) for a in algos]
    tests = [(GAlgoDict[a][0], logger.CLogger(a), \
              tune_batch("%s %s %s"%(data_paras_name, dataset_name, a), \
                dataset, GAlgoDict[a][0], tune_iter, data_paras_name, para_set_lr, batch_sz, batch_rate,\
                num_process, log_file, extra_paras = extra_paras) \
              if a.find("passive")!=-1 \
              else tune_batch("%s %s %s"%(data_paras_name, dataset_name, a), \
                dataset, GAlgoDict[a][0], tune_iter, data_paras_name, para_set_c0lr, batch_sz, batch_rate,\
                num_process, log_file, extra_paras = extra_paras)) \
            for a in algos]
    #print("Tuned parameters: \n", [t[2].c0 for t in tests])    
    if log_file != None:
        log_file.write("\nBEST:\n")
    for mlp in tests:       
        run_multiple(dataset, mlp[0], mlp[1], final_iter, data_paras_name, mlp[2], batch_sz, batch_rate, num_process, extra_paras = extra_paras)
        mlp[1].get_stat(max_cnt_label=label_budget)

        if log_file != None:
            log_info = "%s %s %s %f %f"%(data_paras_name, dataset_name, mlp[1].info, mlp[2].c0, mlp[2].learning_rate)
        else:
            log_info = None

        auc = logger.calc_metric(mlp[1], get_max_cnt_label(dataset, data_paras_name, label_budget), \
                log_file=log_file, log_info=log_info, extra_tail=extra_paras["extra_tail"])
        
        print(mlp[1].info, ": ", mlp[2].c0, " ", mlp[2].learning_rate, "\t\t", auc)
        if auclogger is not None:
            auclogger.log(data_paras_name, dataset_name, mlp[1].info, auc)

    logger.plot_err([mlp[1] for mlp in tests], False, output_filename)

    if log_file != None:
        log_file.write("===========================================\n\n")
