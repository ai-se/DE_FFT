import os
import collections
import pandas as pd

from helpers import load_obj, save_obj
from new_fft_1 import FFT
from SOA import SOA
from plot import plotROC, plotLOC, plot_compare


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
data = {"@ivy":     ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"]}
criterias = ["Accuracy", "Dist2Heaven", "LOC_AUC"] # "Gini", "InfoGain"]

all_data_filepath = os.path.join(data_path, "fixed_NewData_16.pkl")
all_data = {}

'''
if os.path.exists(all_data_filepath):
    all_data = load_obj(all_data_filepath)
else:
    all_data = {}
'''

p_opt_stat = []
cnts = [collections.defaultdict(int) for _ in xrange(len(criterias))]
for name, files in data.iteritems():
    if name not in all_data:
        print '\n'+name
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        paths = [os.path.join(data_path, file_name) for file_name in files]
        train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
        test_df = pd.read_csv(paths[-1])
        train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]
        train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
        test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)
        print "training on:\t" + ', '.join(files[:-1])
        print "testing on: \t" + files[-1]
        all_data[name] = {}
        soa = SOA(train=train_df, test=test_df)
        soa.get_performances()
        all_data[name]["SOA"] = soa
        all_data[name]["FFT"] = []
        for c, criteria in enumerate(criterias):
            print "\n  ................................ " + criteria + " ................................"
            fft = FFT(max_level=4, split_method="median", ifan=False)
            fft.criteria = criteria
            fft.data_name = name
            fft.ignore = {"name", "version", 'name.1', 'prediction'}
            fft.target = "bug"
            fft.train, fft.test = train_df, test_df
            fft.build_trees()               # build and get performance on TEST data
            t_id = fft.find_best_tree()     # find the best tree on TRAIN data
            fft.eval_trees()                # eval all the trees on TEST data
            fft.print_tree(t_id)
            best_structure = fft.structures[fft.best]
            cnts[c][tuple(best_structure)] += 1
            soa.print_soa()
            img_path0 = os.path.join(data_path, 'ROC_' + fft.criteria + "_" + name + ".png")
            plotROC(fft, soa, img_path=img_path0)
            all_data[name]["FFT"] += [fft]

    img_path1 = os.path.join(data_path, 'LOC_dist_5' + name + ".png")
    plotLOC(all_data[name]["FFT"][-2].test, [all_data[name]["FFT"][-2]] + all_data[name]["SOA"].learners,\
                    ["FFT_D2H"] + all_data[name]["SOA"].names,  img_path=img_path1)

    img_path11 = os.path.join(data_path, 'LOC_loc_5' + name + ".png")
    plotLOC(all_data[name]["FFT"][-1].test, [all_data[name]["FFT"][-1]] + all_data[name]["SOA"].learners, \
                  ["FFT_LOC"] + all_data[name]["SOA"].names, img_path=img_path11)

    img_path111 = os.path.join(data_path, 'LOC_comp_5' + name + ".png")
    plotLOC(all_data[name]["FFT"][-1].test, all_data[name]["FFT"][-2:], \
                  ["FFT_D2H", 'FFT_LOC'], img_path=img_path111)


    img_path_del = os.path.join(data_path, 'Delete_' + name + ".png")
    tmp = plotLOC(all_data[name]["FFT"][-1].test, all_data[name]["FFT"][-3:] + all_data[name]["SOA"].learners, \
                  ["FFT_ACC", "FFT_D2H", 'FFT_LOC'] + all_data[name]["SOA"].names, img_path=img_path_del)
    p_opt_stat += [tmp]

    img_path2 = os.path.join(data_path, "FFT_Compare_5" + name + ".png")
    plot_compare(all_data[name]["FFT"][0], all_data[name]["FFT"][1], img_path=img_path2)

p_opt_df = pd.DataFrame(p_opt_stat)
p_opt_path = os.path.join(data_path, "_p_opt_stats5.csv")
p_opt_df.to_csv(p_opt_path)
print 'P_opt statistics saved in: ' + p_opt_path

if not os.path.exists(all_data_filepath):
    save_obj(all_data, all_data_filepath)

cnt_df = pd.DataFrame(cnts)
cnt_path = os.path.join(data_path, "_cnt_stats5.csv")
cnt_df.to_csv(cnt_path)
print 'Cnt statistics saved in: ' + cnt_path

