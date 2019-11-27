import argparse
import numpy as np
import pickle
import math
from utils import *
from methods import *
from objective import *

parser = argparse.ArgumentParser(
    description="Numerical experiments for SCSG Adaptivity paper")
parser.add_argument("-f", "--fname", required=False,
                    default=None, help="File name")
parser.add_argument("-d", "--data", required=False, default="mnist",
                    help="Choose dataset")
parser.add_argument("-b", "--batchsize", required=False,
                    type=float, default=1e-4, help="Small batch size / sample size")
parser.add_argument("-N", "--npass", required=False, type=int,
                    default=5000, help="Maximum number of passes")
parser.add_argument("-la", "--lam", required=False, type=float, default=None,
                    help="L2 regularizer lambda")
parser.add_argument("-se", "--seed", required=False, type=int, default=1,
                    help="Random seed")


setting = parser.parse_args()
filename = setting.fname
dataset = setting.data
b = setting.batchsize
npass = setting.npass
lam = setting.lam
seed = setting.seed
if filename is None:
    filename = '../data/' + dataset + '_optim.p'

A = np.load('../data/' + dataset + '_A.npy')
y = np.load('../data/' + dataset + '_y.npy')
n, p = A.shape
if lam is None:
    lam = 1 / n
b = int(np.ceil(n * b))

L, x0, loss = pickle.load(open('../data/' + dataset + '_params.p', "rb"))
best_etaconst = pickle.load(
    open('../data/' + dataset + '_besteta.p', "rb"))['SVRG']
eta = best_etaconst / L

if (loss == "multi_logistic"):
    num_class = int(len(x0) / p + 1)
    sgrad_eval = grad_to_sgrad(multi_logistic_grad, A, y, num_class)
    func_eval = lambda x: multi_logistic_funcval(x, A, y, num_class)
    grad_eval = lambda x: multi_logistic_grad(x, A, y, num_class)

np.random.seed(seed)

print('-----------------  SCSG  -------------------', '\n')
res = scsg(sgrad_eval, eta, x0,
           0.005, 0.001, 1.25,
           func_eval, grad_eval,
           n=n, b=b, use_geom=True,
           max_ngrads=npass * n,
           lam=lam,
           ngrads_per_pass=0.1)

optimum = np.min(res[1]) / res[1][0]
print(optimum)
pickle.dump(optimum, open(filename, "wb"))
