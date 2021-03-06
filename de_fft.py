from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
import os
from collections import OrderedDict
from learner_1 import *
from DE import DE
from random import seed
import time
import pandas as pd
import numpy as np


## Global Bounds
cwd = os.getcwd()
data_path = os.path.join(cwd, "data")
data = {"@ivy":     ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],\
        "@lucene":  ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],\
        "@poi":     ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],\
        "@synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],\
        "@velocity":["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"], \
        "@camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"], \
        "@jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"], \
        "@log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"], \
        "@xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"], \
        "@xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
        }
learners_para_dic = OrderedDict([("max_level", 1), ("max_depth", 1),  ("split_method", 1), ("ifan", 1)])
learners_para_bounds=[(4, 5), (3, 10), ("mean", "median", "MDLP", "percentile"), (False, True)]
learners_para_categories=["categorical", "integer", "categorical", "categorical"]
learners=[DT, RF, SVM, NB, KNN, LR]
measures=["Dist2Heaven"] #"Accuracy", "LOC_AUC",
repeats=1


def de_fft(res=''):
    seed(1)
    np.random.seed(1)
    paths = [os.path.join(data_path, file_name) for file_name in data[res]]
    train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
    test_df = pd.read_csv(paths[-1])

    ### getting rid of first 3 columns
    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
    train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
    test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)

    final_dic={}
    temp={}
    for x in measures:
        l=[]
        l1 = []
        start_time = time.time()
        for r in xrange(repeats):
            ## Shuffle

            train_df = train_df.sample(frac=1).reset_index(drop=True)
            test_df = test_df.sample(frac=1).reset_index(drop=True)
            #train_df, test_df = train_df.values, test_df.values


            if x == "Dist2Heaven":
                print("Repeating: %s" % r)
                print(x)
                de = DE(GEN=3, Goal="Min", termination="Early")
                v, pareto = de.solve(fft_process, OrderedDict(learners_para_dic),
                                     learners_para_bounds, learners_para_categories, FFT, x, train_df)
                params = v.ind
                val = fft_eval(FFT, train_df, test_df, x, params.values())
                l.append(val)
                l1.append(v.ind)
            else:
                print(x)
                de = DE(GEN=3, Goal="Max", termination="Early")
                v, pareto = de.solve(fft_process, OrderedDict(learners_para_dic),
                                     learners_para_bounds, learners_para_categories, FFT, x, train_df)
                params = v.ind
                val = fft_eval(FFT, train_df, test_df, x, params.values())
                l.append(val)
                l1.append(v.ind)

        total_time = time.time() - start_time
        temp[x] = [l, l1, total_time]
        final_dic[FFT.__name__] = temp
        print(final_dic)
    with open('dump/' + res + '_early.pickle', 'wb') as handle:
        pickle.dump(final_dic, handle)


if __name__ == '__main__':
    '''
    for dataset in data:
        de_fft(dataset)
    '''
    de_fft("@ivy")