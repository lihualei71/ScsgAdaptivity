import numpy as np
import pandas as pd
import pickle
from utils import *
from objective import *
from sklearn.preprocessing import normalize

# MNIST

# A = np.load("../data/mnist_A_all.npy")
# y = np.load("../data/mnist_y_all.npy")
# Llist = np.array([norm2(a) for a in A])
# ids = np.where(Llist <= np.percentile(Llist, 95))
# A = A[ids]
# y = y[ids]
# np.save("../data/mnist_A.npy", A)
# np.save("../data/mnist_y.npy", y)

A = np.load("../data/mnist_A.npy")
y = np.load("../data/mnist_y.npy")
n, p = A.shape

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p * 9)
loss = "multi_logistic"
pickle.dump((smoothparam, x0, loss), open("../data/mnist_params.p", "wb"))

# Covtype

# df = np.loadtxt("../data/covtype.data", delimiter=",")
# A = df[:, range(54)]
# A = normalize(A, axis=0, norm='l2') * np.sqrt(A.shape[0])
# y = np.array(df[:, 54] - 1, dtype="int8")
# Llist = np.array([norm2(a) for a in A])
# ids = np.where(Llist <= np.percentile(Llist, 95))
# A = A[ids]
# y = y[ids]
# np.save("../data/covtype_A.npy", A)
# np.save("../data/covtype_y.npy", y)

A = np.load("../data/covtype_A.npy")
y = np.load("../data/covtype_y.npy")
n, p = A.shape

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p * 7)
loss = "multi_logistic"
pickle.dump((smoothparam, x0, loss), open("../data/covtype_params.p", "wb"))

# Adult

# df = pd.read_csv("../data/adult.data")
# df = np.array(pd.get_dummies(df))
# A = df[:, range(108)]
# A = normalize(A, axis=0, norm='l2') * np.sqrt(A.shape[0])
# y = df[:, 109]
# Llist = np.array([norm2(a) for a in A])
# ids = np.where(Llist <= np.percentile(Llist, 95))
# A = A[ids]
# y = y[ids]
# np.save("../data/adult_A.npy", A)
# np.save("../data/adult_y.npy", y)

A = np.load("../data/adult_A.npy")
y = np.load("../data/adult_y.npy")
n, p = A.shape

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p * 7)
loss = "multi_logistic"
pickle.dump((smoothparam, x0, loss), open("../data/adult_params.p", "wb"))

# Blog

# df = pd.read_csv('../data/blogData_train.csv')
# A = np.array(df.iloc[:, range(280)])
# A = normalize(A, axis=0, norm='l2') * np.sqrt(A.shape[0])
# y = np.array(df.iloc[:, 280])
# Llist = np.array([norm2(a) for a in A])
# ids = np.where(Llist <= np.percentile(Llist, 95))
# A = A[ids]
# y = y[ids]
# np.save("../data/blog_A.npy", A)
# np.save("../data/blog_y.npy", y)

A = np.load("../data/blog_A.npy")
y = np.load("../data/blog_y.npy")

n = A.shape[0]
p = A.shape[1]

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p)
loss = "least_squares"
pickle.dump((smoothparam, x0, loss), open("../data/blog_params.p", "wb"))

# Energy

# df = pd.read_csv('../data/energydata_complete.csv')
# A = np.array(df.iloc[:, np.arange(25) + 2])
# A = normalize(A, axis=0, norm='l2') * np.sqrt(A.shape[0])
# y = np.array(df.iloc[:, 1])
# Llist = np.array([norm2(a) for a in A])
# ids = np.where(Llist <= np.percentile(Llist, 95))
# A = A[ids]
# y = y[ids]
# np.save("../data/energy_A.npy", A)
# np.save("../data/energy_y.npy", y)

A = np.load("../data/energy_A.npy")
y = np.load("../data/energy_y.npy")

n = A.shape[0]
p = A.shape[1]

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p)
loss = "least_squares"
pickle.dump((smoothparam, x0, loss), open("../data/energy_params.p", "wb"))

# Credit

# df = pd.read_csv('../data/credit.csv')
# df = df.iloc[1:(len(df) - 1), :]
# A = np.array(df.iloc[:, np.arange(23) + 1])
# A = A.astype(float)
# A = normalize(A, axis=0, norm='l2') * np.sqrt(A.shape[0])
# y = np.array(df.iloc[:, 24])
# y = y.astype(int)
# Llist = np.array([norm2(a) for a in A])
# ids = np.where(Llist <= np.percentile(Llist, 95))
# A = A[ids]
# y = y[ids]
# np.save("../data/credit_A.npy", A)
# np.save("../data/credit_y.npy", y)

A = np.load("../data/credit_A.npy")
y = np.load("../data/credit_y.npy")

n = A.shape[0]
p = A.shape[1]

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p)
loss = "multi_logistic"
pickle.dump((smoothparam, x0, loss), open("../data/credit_params.p", "wb"))

# Connect

# df = pd.read_csv('../data/connect-4.data')
# y = pd.get_dummies(df.win)[['win', 'loss']]
# y = np.array(y.win * 2 + y.loss)
# y = y.astype(int)
# df = df.iloc[:, range(42)]
# df = pd.get_dummies(df, drop_first=True)
# A = np.array(df)
# A = A.astype(float)
# A = normalize(A, axis=0, norm='l2') * np.sqrt(A.shape[0])
# Llist = np.array([norm2(a) for a in A])
# ids = np.where(Llist <= np.percentile(Llist, 95))
# A = A[ids]
# y = y[ids]
# np.save("../data/connect_A.npy", A)
# np.save("../data/connect_y.npy", y)

A = np.load("../data/connect_A.npy")
y = np.load("../data/connect_y.npy")

n = A.shape[0]
p = A.shape[1]

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p * 2)
loss = "multi_logistic"
pickle.dump((smoothparam, x0, loss), open("../data/connect_params.p", "wb"))

# Sensor

# df = pd.read_csv('../data/sensor.txt', sep=' ', header=None)
# A = np.array(df.iloc[:, np.arange(48)])
# A = A.astype(float)
# A = normalize(A, axis=0, norm='l2') * np.sqrt(A.shape[0])
# y = np.array(df.iloc[:, 48]) - 1
# y = y.astype(int)
# Llist = np.array([norm2(a) for a in A])
# ids = np.where(Llist <= np.percentile(Llist, 95))
# A = A[ids]
# y = y[ids]
# np.save("../data/sensor_A.npy", A)
# np.save("../data/sensor_y.npy", y)

A = np.load("../data/sensor_A.npy")
y = np.load("../data/sensor_y.npy")

n = A.shape[0]
p = A.shape[1]

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p * 10)
loss = "multi_logistic"
pickle.dump((smoothparam, x0, loss), open("../data/sensor_params.p", "wb"))

# Crowdsource

# df = pd.read_csv('../data/crowdsource.csv')
# y = pd.Categorical(df['class']).codes
# A = np.array(df.iloc[:, np.arange(28) + 1])
# A = normalize(A, axis=0, norm='l2') * np.sqrt(A.shape[0])
# np.save("../data/crowdsource_A.npy", A)
# np.save("../data/crowdsource_y.npy", y)

A = np.load("../data/crowdsource_A.npy")
y = np.load("../data/crowdsource_y.npy")

n = A.shape[0]
p = A.shape[1]

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p * 5)
loss = "multi_logistic"
pickle.dump((smoothparam, x0, loss), open(
    "../data/crowdsource_params.p", "wb"))

# Superconduct

# df = pd.read_csv('../data/superconduct.csv')
# y = np.array(df['critical_temp'])
# A = np.array(df.iloc[:, np.arange(81)])
# A = normalize(A, axis=0, norm='l2') * np.sqrt(A.shape[0])
# Llist = np.array([norm2(a) for a in A])
# ids = np.where(Llist <= np.percentile(Llist, 95))
# A = A[ids]
# y = y[ids]
# np.save("../data/superconduct_A.npy", A)
# np.save("../data/superconduct_y.npy", y)

A = np.load("../data/superconduct_A.npy")
y = np.load("../data/superconduct_y.npy")

n = A.shape[0]
p = A.shape[1]

smoothparam = 2 * np.mean([norm2(a) for a in A])
x0 = np.zeros(p)
loss = "least_squares"
pickle.dump((smoothparam, x0, loss), open(
    "../data/superconduct_params.p", "wb"))
