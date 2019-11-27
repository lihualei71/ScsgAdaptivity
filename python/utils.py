import numpy as np


def norm2(x):
    return(np.sum(np.square(x)))


def findmin(grad_eval, func_eval, x0, tol=1e-10, lam=0):

    grad_eval_lam = lambda x: func_eval(x) + lam * x
    func_eval_lam = lambda x: func_eval(x) + lam / 2 * norm2(x)
    res = scipy.optimize.minimize(fun=func_eval, x0=x0, jac=grad_eval, options={
        'disp': True, 'gtol': tol, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': None, 'norm': 2})
    minf = func_eval(res.x)
    return minf


def grad_to_sgrad(grad_eval, A, y, *args):

    n = len(y)

    def sgrad_eval(xlist, size):
        if (size == 0):
            return np.zeros(xlist[0].size)

        size = np.min([size, n])
        subset = np.random.choice(n, size, replace=False)
        A_new = A[subset]
        y_new = y[subset]

        if len(xlist) == 1:
            sgrad = grad_eval(xlist[0], A_new, y_new, *args)
        else:
            sgrad = [grad_eval(x, A_new, y_new, *args) for x in xlist]
        return sgrad

    return sgrad_eval
