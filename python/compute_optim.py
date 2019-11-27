import argparse
import numpy as np
import pickle
from utils import *
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
                    default=500, help="Maximum number of passes")
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

if (loss == "least_squares"):
    sgrad_eval = grad_to_sgrad(least_squares_grad, A, y)
    func_eval = lambda x: least_squares_funcval(x, A, y)
    grad_eval = lambda x: least_squares_grad(x, A, y)
elif (loss == "multi_logistic"):
    num_class = int(len(x0) / p + 1)
    sgrad_eval = grad_to_sgrad(multi_logistic_grad, A, y, num_class)
    func_eval = lambda x: multi_logistic_funcval(x, A, y, num_class)
    grad_eval = lambda x: multi_logistic_grad(x, A, y, num_class)


def svrg_norecord(sgrad_eval, eta, x0, n,
                  func_eval, grad_eval,
                  m=None, b=1, lam=0,
                  max_ngrads=None):

    if max_ngrads is None:
        max_ngrads = 50 * n

    if m is None:
        m = 2 * n

    funcval0 = func_eval(x0) + lam / 2 * norm2(x0)
    ngrads = 0
    xinit = np.array(x0)
    x = np.array(x0)
    n = int(n)
    k = 0

    while True:
        mu = sgrad_eval(xlist=[xinit], size=n) + lam * x0
        x -= eta * mu
        ngrads += n
        ninner = int(np.ceil(m / b))

        for iter in range(ninner):
            sgrad = sgrad_eval(xlist=[x, xinit], size=b)
            nu = sgrad[0] - sgrad[1] + mu + lam * x
            x -= eta * nu

        ngrads += 2 * b * ninner
        xinit = np.array(x)
        k += 1
        print("Step " + str(k) + " is done")
        if ngrads > max_ngrads:
            break

    funcval = func_eval(x) + lam / 2 * norm2(x)
    funcval = funcval / funcval0

    return funcval

np.random.seed(seed)

print('-----------------  SVRG  -------------------', '\n')
optimum = svrg_norecord(sgrad_eval, eta, x0, n,
                        func_eval, grad_eval,
                        b=b, max_ngrads=npass * n,
                        lam=lam)

print(optimum)
pickle.dump(optimum, open(filename, "wb"))
