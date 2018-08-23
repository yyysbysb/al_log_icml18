#!/usr/bin/env python3

import os
import sys
import numpy as np
import experiments
import data
import logger as MLogger
import utils

def run_experiments(dataset_set, paras_set, algos, max_sz, result_folder, \
        batch_sz, batch_rate, label_budget=None, \
        debug=False):
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    utils.backup_code(result_folder)
    auc_logger = MLogger.CAUCLogger(order=dataset_set)
    auc_logger.write(result_folder+"auc.res", append=False)
    with open(result_folder+"metric.log", "w") as log_file:
        for paras_name in paras_set:
            for dataset_name in dataset_set:
                experiments.run_experiments(data.load_data(dataset_name, max_sz), paras_name, \
                    dataset_name, algos, result_folder+dataset_name+"-"+paras_name, \
                    batch_sz, batch_rate,\
                    auclogger=auc_logger, log_file=log_file, label_budget=label_budget, debug=debug)    
            auc_logger.write(result_folder+"auc.res", append=True)
            auc_logger.clear()

def main():   
    result_path = "../results/"
    print(result_path)

    paras_set = ["Uniform", "Uncertainty", "Certainty", "Identical"]
    uci_set = ["letter", "skin", "magic", "covtype",  ]
    libsvm_set = ["mushrooms", "phishing", "splice", "svmguide1", "a5a", "cod-rna", "german", ]
    dataset_set = ["synthetic"] + uci_set + libsvm_set
    algos = ["passive_is", "dbal_is", "dbal_mis", "idbal_mis"]

    run_experiments(dataset_set, paras_set, algos, 6000, result_path, 10, 2, label_budget=-1)
        
if __name__ == "__main__":
    main()
