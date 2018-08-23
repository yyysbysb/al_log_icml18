#!/usr/bin/env python3

import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
import model as MModel
import experiments as MExperiments

def add_to_dict(dic, key, val):
    if key in dic:
        dic[key] += val
    else:
        dic[key] = val

class CLogger(object):
    def __init__(self, info):
        self.err_log = {}
        self.err_stat = None
        self.label_stat = None
        self.orig_err_logs = []
        self.misc_info = []
        self.info = info
        self.conditional_log = None
        self.last_max_cnt_label = None

    def clear(self):
        self.err_log = {}
        self.err_stat = None
        self.label_stat = None
        self.orig_err_logs = []
        self.misc_info = []
        self.conditional_log = None
        self.last_max_cnt_label = None
    
    def init_by_merge(self, logs):
        self.__init__(logs[0].info)
        for l in logs:
            self.orig_err_logs.append(\
                sorted([(e[1][0][0], e[1][0][1]) for e in l.err_log.items()], key=lambda e:e[0]))
            if len(l.misc_info)>0:
                self.misc_info.append(l.misc_info)
            for item in l.err_log.items():
                add_to_dict(self.err_log, item[0], item[1])
    
    def log_misc_info(self, a):
        self.misc_info.append(a)

    def set_cond_log(self, batch_sz, batch_rate):
        self.conditional_log = [batch_sz, batch_rate]

    def check_and_log(self, learning, dataset, sz):
        if self.conditional_log is None: return
        if sz==0 or sz>=self.conditional_log[0]:
            test_err = MModel.evaluate(learning.model, dataset.test_data)
            add_to_dict(self.err_log, sz, [np.array([learning.cnt_labels, test_err])])
        if sz>=self.conditional_log[0]:
            self.conditional_log[0] *= self.conditional_log[1]

    def on_start(self, learning, dataset):
        #test_err = MModel.evaluate(learning.model, dataset.test_data)
        #add_to_dict(self.err_log, 0, [np.array([0, test_err])])
        pass   

    def on_stop(self, learning, dataset):
        test_err = MModel.evaluate(learning.model, dataset.test_data)
        sz = len(dataset.online_data)
        add_to_dict(self.err_log, sz, [np.array([learning.cnt_labels, test_err])])   

    def get_stat(self, max_cnt_label=None):        
        szs = sorted(self.err_log.keys())
        
        ave = [utils.mean(self.err_log[sz]) for sz in szs]
        dev = [utils.stddev(self.err_log[sz]) for sz in szs]
        
        idx = [i for i in range(0, len(ave))]
        idx = sorted(idx, key=lambda i: ave[i][0])
        ave = [ave[i] for i in idx]
        dev = [dev[i] for i in idx]

        length = len(ave) if max_cnt_label is None else len([c for c in ave if c[0]<=max_cnt_label*2])
        self.err_stat = ([c[1] for c in ave[:length]], [c[1] for c in dev[:length]])
        self.label_stat = ([c[0] for c in ave[:length]], [c[0] for c in dev[:length]])
        self.last_max_cnt_label = max_cnt_label

def log_metric(log, ret, log_file, log_info):
    log_file.write("[%s] %s\t%.3f\n"%(time.asctime(), log_info, ret))
    x, xe = log.label_stat
    y, ye = log.err_stat
    result = ["((%.4f, %.4f), (%.4f, %.4f))"%(x[i], xe[i], y[i], ye[i]) for i in range(0, len(x))]
    log_file.write("STAT: [%s]\n"%(" ,".join(result)))
    if log.orig_err_logs is not None:
        for i in range(0, len(log.orig_err_logs)):
            r = log.orig_err_logs[i]
            log_file.write("RUN: [%s]\n"%(",".join(["(%d, %.4f)"%(e[0], e[1]) for e in r])))
            if i<len(log.misc_info):
                misc = log.misc_info[i]
                if len(misc)>0:
                    log_file.write("MISC: " + "\t".join(misc) + "\n")
    log_file.flush()

def calc_metric(log, max_cnt_label, log_file=None, log_info=None, extra_tail = None):    
    if extra_tail is None:
        extra_tail = 0
    label_ave = [x for x in log.label_stat[0] if x<=max_cnt_label]
    length = len(label_ave)    
    err_ave = log.err_stat[0][:length]
    if length<len(log.label_stat[0]):
        p = (max_cnt_label-label_ave[-1])/(log.label_stat[0][length]-label_ave[-1])
        label_ave.append(max_cnt_label)
        err_ave.append(err_ave[-1] + (log.err_stat[0][length]-err_ave[-1])*p)
        length += 1

    label_ave = label_ave + [(max_cnt_label+1)*(1+extra_tail)]
    err_ave = err_ave + [err_ave[-1]]
    ret = 0   
    for i in range(1, length+1):        
        ret += 0.5 * (err_ave[i]+err_ave[i-1]) * (label_ave[i]-label_ave[i-1])

    if log_file != None:
        log_metric(log, ret, log_file, log_info)
    return ret

def plot_err(loggers, logscale=False, filename=None):
    fig, ax = plt.subplots()
    for l in loggers:        
        x, xe = l.label_stat
        y, ye = l.err_stat

        if l.last_max_cnt_label is not None:
            length = len([nl for nl in x if nl<=l.last_max_cnt_label*1.2])
            x, xe = x[:length], xe[:length]
            y, ye = y[:length], ye[:length]

        #print([z for z in zip(x,y)])
        if logscale: 
            ax.set_xscale("log", nonposx='clip')
            x = [ele+1 for ele in x]
            xe = [ele for ele in xe]
            for i in range(0, len(x)):
                if x[i]-xe[i]<1:
                    xe[i] = 0
                if x[i]<1:
                    x[i] = 1        
        ax.errorbar(x, y, xerr=xe, yerr=ye, label=MExperiments.GAlgoDict[l.info][1])
        ax.legend()
        #ax.errorbar(x,y)
        #plt.show()
    plt.xlabel("# of labels")
    plt.ylabel("Test error")
    if filename is None:
        plt.show()
    else:
        if logscale: 
            plt.savefig(filename+"-log.png")
        else:
            plt.savefig(filename+".png")
    plt.close('all')

class CAUCLogger(object):
    def __init__(self, results=None, order=None):
        self.results = results or {}
        self.order = order

    def clear(self):
        self.results = {}

    def log(self, policy_info, dataset_info, algo_name, auc):
        add_to_dict(self.results, policy_info, [(dataset_info, algo_name, auc)])

    def write(self, of_name, sort=False, append=False):
        fflag = "a" if append else "w"
        table_head = "\\begin{table}[tb]\n\\centering\n\\caption{AUC under %s policy}\n\\label{tab:auc-%s}\n\\begin{tabular}{lllll}\n\\toprule\n"
        table_tail = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        with open(of_name, fflag) as f:
            for p in self.results.items():
                f.write(table_head%(p[0], p[0].lower()))
                dic = {}
                for daa in p[1]:
                    add_to_dict(dic, daa[0], [(daa[1], daa[2])])
                if self.order is None:
                    ds = list(dic.keys())
                else:
                    ds = [d for d in self.order if d in dic]
                if sort: ds = sorted(ds)
                algos = [MExperiments.GAlgoDict[aa[0]][1] for aa in dic[ds[0]]]
                if sort: algos = sorted(algos)
                f.write("Dataset & "+(" & ".join(algos))+" \\\\\n\\midrule\n")
                stat = []
                rel_stat = []
                for d in ds:
                    line = dic[d]
                    if sort: line = sorted(line, key = lambda e: e[0])
                    for i in range(0, len(algos)):
                        assert algos[i] == MExperiments.GAlgoDict[line[i][0]][1]
                    f.write("%s & %s \\\\\n" % (d, " & ".join([str.format("%.2f"%e[1]) for e in line])))             
                    if d!="synthetic":
                        #print(d)
                        stat.append(np.array([e[1] for e in line]))
                        rel_stat.append(np.array([(e[1]-line[0][1])/line[0][1] for e in line[1:]]))
                f.write(table_tail+"\n\n")
                if len(stat)>0:
                    f.write(" & ".join([str.format("%.3f"%e) for e in utils.mean(stat)])+"\n")
                    f.write(" & ".join([str.format("%.3f"%e) for e in utils.mean(rel_stat)])+"\n")
                f.write("\n\n")
