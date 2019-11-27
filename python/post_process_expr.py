import numpy as np
import pandas as pd
import pickle
import math

params = pd.read_csv("../jobs/SCSG_expr_params.txt", sep=" ", header=None)
params.columns = ['data', 'etaL']
etaL_list = np.unique(np.array(params.etaL))
data_list = np.unique(np.array(params.data))
methods_list = ['SCSG', 'SVRG', 'SARAH', 'Katyusha',
                'SGD0.0', 'SGD1.0', 'GD']

best_res = {}
for data in data_list:
    best_res[data] = {}
    y = np.load('../data/' + data + '_y.npy')
    n = len(y)
    for method in methods_list:
        best_res[data][method] = [[0], [0], [math.inf], [math.inf]]
    for etaL in etaL_list:
        fname = '../results/SCSG-data-' + data + \
            '-etaL-' + str(etaL) + \
            '.p'
        res = pickle.load(open(fname, "rb"))
        for i in range(len(methods_list)):
            npass_record = np.array(res[i][0]) / n
            funcval_record = np.array(res[i][1] / res[i][1][0])
            gradval_record = np.array(res[i][2] / res[i][2][0])
            best_record = best_res[data][methods_list[i]]
            if np.min(funcval_record) < np.min(best_record[2]):
                best_res[data][methods_list[i]] = [
                    [etaL],
                    npass_record,
                    funcval_record,
                    gradval_record]

for data in data_list:
    best_etaconst = {}
    fname = '../data/' + data + '_besteta.p'
    for method in methods_list:
        best_etaconst[method] = best_res[data][method][0][0]
    print(data + str(best_etaconst))
    pickle.dump(best_etaconst, open(fname, "wb"))

for data in data_list:
    fname = '../data/' + data + '_optim.p'
    optim = pickle.load(open(fname, "rb"))
    for method in methods_list:
        best_res[data][method][2] -= optim
        best_res[data][method][2] /= best_res[data][method][2][0]

pickle.dump(best_res, open("../results/SCSG_expr_best_tune.p", "wb"))
