import argparse
import numpy as np
import pickle
from utils import *
from methods import *
from objective import *

parser = argparse.ArgumentParser(
    description="Numerical experiments for SCSG Adaptivity paper")
parser.add_argument("-f", "--fname", required=True, help="File name")
parser.add_argument("-d", "--data", required=False, default="mnist",
                    help="Choose dataset: 'mnist', 'covtype', 'blog'")
parser.add_argument("-b", "--batchsize", required=False,
                    type=float, default=1e-4, help="Small batch size / sample size")
parser.add_argument("-N", "--npass", required=False, type=int,
                    default=50, help="Maximum number of passes")
parser.add_argument("-c", "--eta", required=False, type=float, default=1,
                    help="Stepsize eta X smoothness parameter L")
parser.add_argument("-m", "--m0", required=False, type=float,
                    default=5e-3, help="m0 for SCSG")
parser.add_argument("-B", "--B0", required=False, type=float,
                    default=1e-3, help="B0 for SCSG")
parser.add_argument("-a", "--alpha", required=False, type=float, default=1.25,
                    help="alpha for SCSG")
parser.add_argument("-la", "--lam", required=False, type=float, default=None,
                    help="L2 regularizer lambda")
parser.add_argument("-se", "--seed", required=False, type=int, default=1,
                    help="Random seed")


setting = parser.parse_args()
filename = setting.fname
dataset = setting.data
b = setting.batchsize
npass = setting.npass
etaconst = setting.eta
m0 = setting.m0
B0 = setting.B0
alpha = setting.alpha
lam = setting.lam
seed = setting.seed

A = np.load('../data/' + dataset + '_A.npy')
y = np.load('../data/' + dataset + '_y.npy')
n, p = A.shape
if lam is None:
    lam = 1 / n
b = int(np.ceil(n * b))
m0 = int(np.ceil(n * m0))
B0 = int(np.ceil(n * B0))

L, x0, loss = pickle.load(open('../data/' + dataset + '_params.p', "rb"))
eta = etaconst / L

if (loss == "least_squares"):
    sgrad_eval = grad_to_sgrad(least_squares_grad, A, y)
    func_eval = lambda x: least_squares_funcval(x, A, y)
    grad_eval = lambda x: least_squares_grad(x, A, y)
elif (loss == "multi_logistic"):
    num_class = int(len(x0) / p + 1)
    sgrad_eval = grad_to_sgrad(multi_logistic_grad, A, y, num_class)
    func_eval = lambda x: multi_logistic_funcval(x, A, y, num_class)
    grad_eval = lambda x: multi_logistic_grad(x, A, y, num_class)

np.random.seed(seed)

print('-----------------  SVRG  -------------------', '\n')
res_svrg = svrg(sgrad_eval, eta, x0, n,
                func_eval, grad_eval,
                b=b, max_ngrads=npass * n, lam=lam)

print('-----------------  SVRG++  -------------------', '\n')
res_svrgpp = svrgpp(sgrad_eval, eta, x0, n,
                    func_eval, grad_eval,
                    b=b, max_ngrads=npass * n, lam=lam)

print('-----------------  SARAH  -------------------', '\n')
res_sarah = sarah(sgrad_eval, eta, x0, n,
                  func_eval, grad_eval,
                  b=b, max_ngrads=npass * n, lam=lam)

print('-----------------  SARAH++  -------------------', '\n')
res_sarahpp = sarahpp(sgrad_eval, eta, x0, n,
                      func_eval, grad_eval,
                      b=b, max_ngrads=npass * n, lam=lam)

print('----------------  Katyusha_ns  -----------------', '\n')
res_katyusha_ns = katyusha_ns(sgrad_eval, eta, x0, n,
                              func_eval, grad_eval,
                              b=b, max_ngrads=npass * n,
                              lam=lam)

print('-----------------  SGD0.0  -------------------', '\n')
res_sgd1 = sgd(sgrad_eval, eta, x0,
               func_eval, grad_eval,
               decay=0, avg=False,
               n=n, b=b, max_ngrads=npass * n, lam=lam)

print('-----------------  SGD0.5  -------------------', '\n')
res_sgd2 = sgd(sgrad_eval, eta, x0,
               func_eval, grad_eval,
               decay=0.5, avg=True,
               n=n, b=b,
               max_ngrads=npass * n, lam=lam)

print('-----------------  SGD0.66  ------------------', '\n')
res_sgd3 = sgd(sgrad_eval, eta, x0,
               func_eval, grad_eval,
               decay=2 / 3, avg=True,
               n=n, b=b,
               max_ngrads=npass * n, lam=lam)

print('-----------------  SGD1.0  -------------------', '\n')
res_sgd4 = sgd(sgrad_eval, eta, x0,
               func_eval, grad_eval,
               decay=1.0, avg=False,
               n=n, b=b,
               max_ngrads=npass * n, lam=lam)

print('-----------------  GD  -------------------', '\n')
res_gd = gd(sgrad_eval, eta, x0, n,
            func_eval, grad_eval,
            max_ngrads=50 * n, lam=lam)

print('-----------------  SCSG  -------------------', '\n')
res_scsg = scsg(sgrad_eval, eta, x0, m0, B0, alpha,
                func_eval, grad_eval,
                n=n, b=b, use_geom=True,
                max_ngrads=npass * n, lam=lam)

# print('-------------  SCSG no geom  ---------------', '\n')
# res_scsg2 = scsg(sgrad_eval, eta, x0, m0, B0, alpha,
#                  func_eval, grad_eval,
#                  n=n, b=b, use_geom=False,
#                  max_ngrads=npass * n, lam=lam)

res = (res_scsg,
       res_svrg, res_svrgpp,
       res_sarah, res_sarahpp,
       res_katyusha_ns,
       res_sgd1, res_sgd2, res_sgd3, res_sgd4,
       res_gd)
pickle.dump(res, open(filename, "wb"))
