import matplotlib.pyplot as plt
import pickle


def SCSG_plot(dat, title, fname, target):
    if target == "funcval":
        idx = 2
        ylabel = r'$(f(x) - \hat{f}^{*}) / (f(x_0) - \hat{f}^{*})$'
    elif target == "gradval":
        idx = 3
        ylabel = r'$\|F(x)\|^{2}$'
    plt.plot(dat['SCSG'][1], dat['SCSG'][idx],
             'k-', label='SCSG')
    plt.plot(dat['SVRG'][1], dat['SVRG'][idx],
             'g-', label="SVRG")
    plt.plot(dat['SARAH'][1], dat['SARAH'][idx],
             'r-', label="SARAH")
    plt.plot(dat['Katyusha'][1], dat['Katyusha'][idx],
             'b-', label="Katyusha(ns)")
    plt.plot(dat['SGD0.0'][1], dat['SGD0.0'][idx],
             'c-', label="SGD")
    plt.plot(dat['SGD1.0'][1], dat['SGD1.0'][idx],
             'c--', label="SGD(decay)")
    plt.plot(dat['GD'][1], dat['GD'][idx],
             'y-', label="GD")
    plt.xlim(0, 50)
    plt.xlabel("Number of Effective Passes")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.yscale('log')
    plt.savefig(fname)
    plt.close()

res = pickle.load(open("../results/SCSG_expr_best_tune.p",
                       "rb"))

for data in list(res.keys()):
    funcval_fname = '../figs/' + data + '_funcval.png'
    SCSG_plot(res[data], data, funcval_fname, "funcval")
    gradval_fname = '../figs/' + data + '_gradval.png'
    SCSG_plot(res[data], data, gradval_fname, "gradval")
