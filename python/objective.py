import scipy.optimize
import pdb
from utils import *


def least_squares_grad(x, A, y):

    resid = (np.dot(A, x) - y)
    grad = np.dot(A.transpose(), resid) / len(y)
    return grad


def least_squares_funcval(x, A, y):

    return norm2(np.dot(A, x) - y) / len(y) / 2


def truncate(x, high=1 - 1e-6, low=1e-6):

    return np.minimum(np.maximum(x, low), high)


def multi_logistic_grad(x, A, y, num_class):

    n, p = A.shape
    x_mat = x.reshape(num_class - 1, p)

    linear_comp = np.dot(A, x_mat.T)
    linear_comp_max = np.array([np.max(row) for row in linear_comp])
    linear_comp -= linear_comp_max[:, np.newaxis]
    exp_linear_comp = np.exp(linear_comp)
    exp_linear_comp_rowsum = np.add(
        exp_linear_comp.sum(axis=1), np.exp(-linear_comp_max))
    factor = exp_linear_comp / exp_linear_comp_rowsum[:, np.newaxis]
    factor = np.array([truncate(x) for x in factor])
    factor = np.array([x / sum(x) * min(sum(x), 1 - 1e-6) for x in factor])
    ind_y_not_zero = np.nonzero(y)[0]
    factor[ind_y_not_zero, y[ind_y_not_zero] - 1] -= 1
    sum_grad = np.dot(factor.T, A).reshape(p * (num_class - 1))
    return sum_grad / n


def multi_logistic_funcval(x, A, y, num_class):

    n, p = A.shape
    x_mat = x.reshape(num_class - 1, p)

    linear_comp = np.dot(A, x_mat.T)
    linear_comp_max = np.array([np.max(row) for row in linear_comp])
    linear_comp -= linear_comp_max[:, np.newaxis]
    exp_linear_comp = np.exp(linear_comp)
    exp_linear_comp_rowsum = np.add(
        exp_linear_comp.sum(axis=1), np.exp(-linear_comp_max))
    ind_y_not_zero = np.nonzero(y)[0]
    correct_term = -linear_comp_max
    correct_term[ind_y_not_zero] = linear_comp[
        ind_y_not_zero, y[ind_y_not_zero] - 1]
    val = np.mean(np.log(exp_linear_comp_rowsum) - correct_term)
    return val
