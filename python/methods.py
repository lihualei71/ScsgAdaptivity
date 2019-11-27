import numpy as np
import math
import random
from utils import *
import pdb


def scsg(sgrad_eval, eta, x0, m0, B0, alpha,
         func_eval, grad_eval,
         n=math.inf, b=1, lam=0, use_geom=True,
         max_ngrads=None,
         ngrads_per_pass=5):

    if max_ngrads is None:
        max_ngrads = np.min([50 * n, 10**9])

    funcval = func_eval(x0) + lam / 2 * norm2(x0)
    funcval_record = [funcval]
    gradval = norm2(grad_eval(x0) + lam * x0)
    gradval_record = [gradval]
    ngrads_record = [0]
    ngrads = 0
    xinit = np.array(x0)
    x = np.array(x0)
    m = m0
    B = B0
    ngrads_update = int(np.ceil(min(n, max_ngrads) / ngrads_per_pass))
    err = False

    while True:
        try:
            m *= alpha
            B *= (alpha ** 2)
            B = np.min([B, n])
            mu = sgrad_eval(xlist=[xinit], size=int(B)) + lam * x0
            x -= eta * mu
            ngrads += int(B)
            ngrads_record.append(ngrads)
            funcval = func_eval(x) + lam / 2 * norm2(x)
            funcval_record.append(funcval)
            gradval = norm2(grad_eval(x) + lam * x)
            gradval_record.append(gradval)
        except:
            print('Warning: algorithm divergent')
            break

        if math.isnan(funcval) or math.isnan(gradval):
            print('Warning: algorithm divergent')
            break

        if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
            print('Warning: algorithm divergent')
            break

        if use_geom:
            gamma = b / (m + b)
            ninner = np.random.geometric(p=gamma, size=1)[0]
        else:
            ninner = int(np.ceil(m / b))

        count = 1
        ngrads_inc = 0
        for iter in range(ninner):
            try:
                sgrad = sgrad_eval(xlist=[x, xinit], size=b)
                nu = sgrad[0] - sgrad[1] + mu + lam * x
                x -= eta * nu
                ngrads_inc += 2 * b
                if ngrads_inc >= count * ngrads_update:
                    count += 1
                    ngrads_record.append(ngrads + ngrads_inc)
                    funcval = func_eval(x) + lam / 2 * norm2(x)
                    funcval_record.append(funcval)
                    gradval = norm2(grad_eval(x) + lam * x)
                    gradval_record.append(gradval)
            except:
                err = True
                print('Warning: algorithm divergent')
                break

            if math.isnan(funcval) or math.isnan(gradval):
                err = True
                print('Warning: algorithm divergent')
                break

            if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
                err = True
                print('Warning: algorithm divergent')
                break

        if err:
            break

        ngrads += ngrads_inc
        ngrads_record.append(ngrads)
        funcval = func_eval(x) + lam / 2 * norm2(x)
        funcval_record.append(funcval)
        gradval = norm2(grad_eval(x) + lam * x)
        gradval_record.append(gradval)

        xinit = np.array(x)
        print('Number of gradients: {0}, m: {1}, B: {2}, Function value: {3}'.format(
            ngrads, m, B, funcval), '\n')
        if ngrads > max_ngrads:
            break

    return ngrads_record, funcval_record, gradval_record


def svrg(sgrad_eval, eta, x0, n,
         func_eval, grad_eval,
         m=None, b=1, lam=0,
         max_ngrads=None,
         ngrads_per_pass=5):

    if max_ngrads is None:
        max_ngrads = 50 * n

    if m is None:
        m = 2 * n

    funcval = func_eval(x0) + lam / 2 * norm2(x0)
    funcval_record = [funcval]
    gradval = norm2(grad_eval(x0) + lam * x0)
    gradval_record = [gradval]
    ngrads_record = [0]
    ngrads = 0
    xinit = np.array(x0)
    x = np.array(x0)
    n = int(n)
    ngrads_update = int(np.ceil(min(n, max_ngrads) / ngrads_per_pass))
    err = False

    while True:
        try:
            mu = sgrad_eval(xlist=[xinit], size=n) + lam * x0
            x -= eta * mu
            ngrads += n
            ngrads_record.append(ngrads)
            funcval = func_eval(x) + lam / 2 * norm2(x)
            funcval_record.append(funcval)
            gradval = norm2(grad_eval(x) + lam * x)
            gradval_record.append(gradval)
        except:
            print('Warning: algorithm divergent')
            break

        if math.isnan(funcval) or math.isnan(gradval):
            print('Warning: algorithm divergent')
            break

        if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
            print('Warning: algorithm divergent')
            break

        ninner = int(np.ceil(m / b))

        count = 1
        ngrads_inc = 0
        for iter in range(ninner):
            try:
                sgrad = sgrad_eval(xlist=[x, xinit], size=b)
                nu = sgrad[0] - sgrad[1] + mu + lam * x
                x -= eta * nu
                ngrads_inc += 2 * b
                if ngrads_inc >= count * ngrads_update:
                    count += 1
                    ngrads_record.append(ngrads + ngrads_inc)
                    funcval = func_eval(x) + lam / 2 * norm2(x)
                    funcval_record.append(funcval)
                    gradval = norm2(grad_eval(x) + lam * x)
                    gradval_record.append(gradval)
            except:
                err = True
                print('Warning: algorithm divergent')
                break

            if math.isnan(funcval) or math.isnan(gradval):
                err = True
                print('Warning: algorithm divergent')
                break

            if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
                err = True
                print('Warning: algorithm divergent')
                break

        if err:
            break

        ngrads += ngrads_inc
        ngrads_record.append(ngrads)
        funcval = func_eval(x) + lam / 2 * norm2(x)
        funcval_record.append(funcval)
        gradval = norm2(grad_eval(x) + lam * x)
        gradval_record.append(gradval)

        xinit = np.array(x)
        print('Number of gradients: {0}, Function value: {1}'.format(
            ngrads, funcval), '\n')
        if ngrads > max_ngrads:
            break

    return ngrads_record, funcval_record, gradval_record


def svrgpp(sgrad_eval, eta, x0, n,
           func_eval, grad_eval,
           m0=None, b=1, lam=0,
           max_ngrads=None,
           ngrads_per_pass=5):

    if max_ngrads is None:
        max_ngrads = 50 * n

    m = m0
    if m is None:
        m = int(np.ceil(n / 4))

    funcval = func_eval(x0) + lam / 2 * norm2(x0)
    funcval_record = [funcval]
    gradval = norm2(grad_eval(x0) + lam * x0)
    gradval_record = [gradval]
    ngrads_record = [0]
    ngrads = 0
    xinit = np.array(x0)
    x = np.array(x0)
    n = int(n)
    ngrads_update = int(np.ceil(min(n, max_ngrads) / ngrads_per_pass))
    err = False

    while True:
        try:
            mu = sgrad_eval(xlist=[xinit], size=n) + lam * x0
            x -= eta * mu
            ngrads += n
            ngrads_record.append(ngrads)
            funcval = func_eval(x) + lam / 2 * norm2(x)
            funcval_record.append(funcval)
            gradval = norm2(grad_eval(x) + lam * x)
            gradval_record.append(gradval)
        except:
            print('Warning: algorithm divergent')
            break

        if math.isnan(funcval) or math.isnan(gradval):
            print('Warning: algorithm divergent')
            break

        if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
            print('Warning: algorithm divergent')
            break

        ninner = int(np.ceil(m / b))
        count = 1
        ngrads_inc = 0
        for iter in range(ninner):
            try:
                sgrad = sgrad_eval(xlist=[x, xinit], size=b)
                nu = sgrad[0] - sgrad[1] + mu + lam * x
                x -= eta * nu
                ngrads_inc += 2 * b
                if ngrads_inc >= count * ngrads_update:
                    count += 1
                    ngrads_record.append(ngrads + ngrads_inc)
                    funcval = func_eval(x) + lam / 2 * norm2(x)
                    funcval_record.append(funcval)
                    gradval = norm2(grad_eval(x) + lam * x)
                    gradval_record.append(gradval)
            except:
                err = True
                print('Warning: algorithm divergent')
                break

            if math.isnan(funcval) or math.isnan(gradval):
                err = True
                print('Warning: algorithm divergent')
                break

            if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
                err = True
                print('Warning: algorithm divergent')
                break

        if err:
            break

        ngrads += ngrads_inc
        ngrads_record.append(ngrads)
        funcval = func_eval(x) + lam / 2 * norm2(x)
        funcval_record.append(funcval)
        gradval = norm2(grad_eval(x) + lam * x)
        gradval_record.append(gradval)

        xinit = np.array(x)
        m *= 2
        print('Number of gradients: {0}, Function value: {1}'.format(
            ngrads, funcval), '\n')
        if ngrads > max_ngrads:
            break

    return ngrads_record, funcval_record, gradval_record


def sarah(sgrad_eval, eta, x0, n,
          func_eval, grad_eval,
          m=None, b=1, lam=0,
          max_ngrads=None,
          ngrads_per_pass=5):

    if max_ngrads is None:
        max_ngrads = 50 * n

    if m is None:
        m = 2 * n

    funcval = func_eval(x0) + lam / 2 * norm2(x0)
    funcval_record = [funcval]
    gradval = norm2(grad_eval(x0) + lam * x0)
    gradval_record = [gradval]
    ngrads_record = [0]
    ngrads = 0
    xinit = np.array(x0)
    x = np.array(x0)
    n = int(n)
    ngrads_update = int(np.ceil(min(n, max_ngrads) / ngrads_per_pass))
    err = False

    while True:
        try:
            nu = sgrad_eval(xlist=[xinit], size=n) + lam * x0
            xlast = np.array(x)
            x -= eta * nu
            ngrads += n
            ngrads_record.append(ngrads)
            funcval = func_eval(x) + lam / 2 * norm2(x)
            funcval_record.append(funcval)
            gradval = norm2(grad_eval(x) + lam * x)
            gradval_record.append(gradval)
        except:
            print('Warning: algorithm divergent')
            break

        if math.isnan(funcval) or math.isnan(gradval):
            print('Warning: algorithm divergent')
            break

        if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
            print('Warning: algorithm divergent')
            break

        ninner = int(np.ceil(m / b))
        count = 1
        ngrads_inc = 0
        for iter in range(ninner):
            try:
                sgrad = sgrad_eval(xlist=[x, xlast], size=b)
                nu += sgrad[0] - sgrad[1] + lam * (x - xlast)
                xlast = np.array(x)
                x -= eta * nu
                ngrads_inc += 2 * b
                if ngrads_inc >= count * ngrads_update:
                    count += 1
                    ngrads_record.append(ngrads + ngrads_inc)
                    funcval = func_eval(x) + lam / 2 * norm2(x)
                    funcval_record.append(funcval)
                    gradval = norm2(grad_eval(x) + lam * x)
                    gradval_record.append(gradval)
            except:
                err = True
                print('Warning: algorithm divergent')
                break

            if math.isnan(funcval) or math.isnan(gradval):
                err = True
                print('Warning: algorithm divergent')
                break

            if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
                err = True
                print('Warning: algorithm divergent')
                break

        if err:
            break

        ngrads += ngrads_inc
        ngrads_record.append(ngrads)
        funcval = func_eval(x) + lam / 2 * norm2(x)
        funcval_record.append(funcval)
        gradval = norm2(grad_eval(x) + lam * x)
        gradval_record.append(gradval)
        xinit = np.array(x)
        print('Number of gradients: {0}, Function value: {1}'.format(
            ngrads, funcval), '\n')
        if ngrads > max_ngrads:
            break

    return ngrads_record, funcval_record, gradval_record


def sarahpp(sgrad_eval, eta, x0, n,
            func_eval, grad_eval,
            m=None, gamma=0.125, b=1, lam=0,
            max_ngrads=None,
            ngrads_per_pass=5):

    if max_ngrads is None:
        max_ngrads = 50 * n

    if m is None:
        m = 2 * n

    funcval = func_eval(x0) + lam / 2 * norm2(x0)
    funcval_record = [funcval]
    gradval = norm2(grad_eval(x0) + lam * x0)
    gradval_record = [gradval]
    ngrads_record = [0]
    ngrads = 0
    xinit = np.array(x0)
    x = np.array(x0)
    n = int(n)
    ngrads_update = int(np.ceil(min(n, max_ngrads) / ngrads_per_pass))
    err = False

    while True:
        try:
            nu = sgrad_eval(xlist=[xinit], size=n) + lam * x0
            normnu0 = norm2(nu)
            xlast = np.array(x)
            x -= eta * nu
            ngrads += n
            ngrads_record.append(ngrads)
            funcval = func_eval(x) + lam / 2 * norm2(x)
            funcval_record.append(funcval)
            gradval = norm2(grad_eval(x) + lam * x)
            gradval_record.append(gradval)
        except:
            print('Warning: algorithm divergent')
            break

        if math.isnan(funcval) or math.isnan(gradval):
            print('Warning: algorithm divergent')
            break

        if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
            print('Warning: algorithm divergent')
            break

        ninner = int(np.ceil(m / b))
        count = 1
        ngrads_inc = 0
        for iter in range(ninner):
            try:
                if norm2(nu) < gamma * normnu0:
                    break
                sgrad = sgrad_eval(xlist=[x, xlast], size=b)
                nu += sgrad[0] - sgrad[1] + lam * (x - xlast)
                xlast = np.array(x)
                x -= eta * nu
                ngrads_inc += 2 * b
                if ngrads_inc >= count * ngrads_update:
                    count += 1
                    ngrads_record.append(ngrads + ngrads_inc)
                    funcval = func_eval(x) + lam / 2 * norm2(x)
                    funcval_record.append(funcval)
                    gradval = norm2(grad_eval(x) + lam * x)
                    gradval_record.append(gradval)
            except:
                err = True
                print('Warning: algorithm divergent')
                break

            if math.isnan(funcval) or math.isnan(gradval):
                err = True
                print('Warning: algorithm divergent')
                break

            if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
                err = True
                print('Warning: algorithm divergent')
                break

        if err:
            break

        ngrads += ngrads_inc
        ngrads_record.append(ngrads)
        funcval = func_eval(x) + lam / 2 * norm2(x)
        funcval_record.append(funcval)
        gradval = norm2(grad_eval(x) + lam * x)
        gradval_record.append(gradval)

        xinit = np.array(x)
        print('Number of gradients: {0}, Function value: {1}'.format(
            ngrads, funcval), '\n')
        if ngrads > max_ngrads:
            break

    return ngrads_record, funcval_record, gradval_record


def katyusha_ns(sgrad_eval, eta, x0, n,
                func_eval, grad_eval,
                m=None, b=1, tau2=0.5, lam=0,
                max_ngrads=None,
                ngrads_per_pass=5):

    if max_ngrads is None:
        max_ngrads = 50 * n

    if m is None:
        m = 2 * n

    funcval = func_eval(x0) + lam / 2 * norm2(x0)
    funcval_record = [funcval]
    gradval = norm2(grad_eval(x0) + lam * x0)
    gradval_record = [gradval]
    ngrads_record = [0]
    ngrads = 0
    xinit = np.array(x0)
    x = np.array(x0)
    y = np.array(x0)
    z = np.array(x0)
    n = int(n)
    s = 0
    ngrads_update = int(np.ceil(min(n, max_ngrads) / ngrads_per_pass))
    err = False

    while True:
        try:
            mu = sgrad_eval(xlist=[xinit], size=n) + lam * x0
            tau1 = 2 / (s + 4)
            alpha = eta / tau1
            yavg = 0
            xinit = np.array(x)
            ngrads += n
            ngrads_record.append(ngrads)
            funcval_record.append(funcval_record[-1])
            gradval_record.append(gradval_record[-1])
        except:
            print('Warning: algorithm divergent')
            break

        if math.isnan(funcval) or math.isnan(gradval):
            print('Warning: algorithm divergent')
            break

        if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
            print('Warning: algorithm divergent')
            break

        ninner = int(np.ceil(m / b))
        count = 1
        ngrads_inc = 0
        for iter in range(ninner):
            try:
                x = tau1 * z + tau2 * xinit + (1 - tau1 - tau2) * y
                sgrad = sgrad_eval(xlist=[x, xinit], size=b)
                nu = sgrad[0] - sgrad[1] + mu + lam * x
                update = alpha * nu
                z -= update
                y = x - eta * nu
                # y = x - tau1 * update
                yavg = (yavg * iter + y) / (iter + 1)
                ngrads_inc += 2 * b
                if ngrads_inc >= count * ngrads_update:
                    count += 1
                    ngrads_record.append(ngrads + ngrads_inc)
                    funcval = func_eval(yavg) + lam / 2 * norm2(yavg)
                    funcval_record.append(funcval)
                    gradval = norm2(grad_eval(yavg) + lam * yavg)
                    gradval_record.append(gradval)
            except:
                err = True
                print('Warning: algorithm divergent')
                break

            if math.isnan(funcval) or math.isnan(gradval):
                err = True
                print('Warning: algorithm divergent')
                break

            if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
                err = True
                print('Warning: algorithm divergent')
                break

        if err:
            break

        xinit = np.array(yavg)
        ngrads += ngrads_inc
        ngrads_record.append(ngrads)
        funcval = func_eval(yavg) + lam / 2 * norm2(yavg)
        funcval_record.append(funcval)
        gradval = norm2(grad_eval(yavg) + lam * yavg)
        gradval_record.append(gradval)

        s += 1
        print('Number of gradients: {0}, Function value: {1}'.format(
            ngrads, funcval), '\n')
        if ngrads > max_ngrads:
            break

    return ngrads_record, funcval_record, gradval_record


def sgd(sgrad_eval, eta, x0,
        func_eval, grad_eval,
        decay=0.5, avg=False,
        n=math.inf, b=1, lam=0,
        max_ngrads=None,
        ngrads_per_pass=5):

    if max_ngrads is None:
        max_ngrads = np.min([50 * n, 10**9])

    funcval = func_eval(x0) + lam / 2 * norm2(x0)
    funcval_record = [funcval]
    gradval = norm2(grad_eval(x0) + lam * x0)
    gradval_record = [gradval]
    ngrads_record = [0]
    ngrads = 0
    x = np.array(x0)
    xrec = np.array(x0)
    etaroot = eta
    tot_updates = int(np.ceil(max_ngrads / b))
    ngrads_update = int(np.ceil(min(n, max_ngrads) / ngrads_per_pass))
    err = False

    count = 1
    for iter in range(tot_updates):
        try:
            eta = etaroot * np.power(iter + 1, -decay)
            sgrad = sgrad_eval(xlist=[x], size=int(b)) + lam * x
            x -= eta * sgrad
            if avg:
                xrec = (xrec * iter + x) / (iter + 1)
            else:
                xrec = x
            ngrads += int(b)
            if ngrads >= count * ngrads_update:
                count += 1
                ngrads_record.append(ngrads)
                funcval = func_eval(xrec) + lam / 2 * norm2(xrec)
                funcval_record.append(funcval)
                gradval = norm2(grad_eval(xrec) + lam * xrec)
                gradval_record.append(gradval)
                print('Number of gradients: {0}, Function value: {1}'.format(
                    ngrads, funcval), '\n')
        except:
            print('Warning: algorithm divergent')
            break

        if math.isnan(funcval) or math.isnan(gradval):
            print('Warning: algorithm divergent')
            break

        if (funcval > 100 * funcval_record[0]) or (gradval > 100 * gradval_record[0]):
            print('Warning: algorithm divergent')
            break

    return ngrads_record, funcval_record, gradval_record


def gd(sgrad_eval, eta, x0, n,
       func_eval, grad_eval,
       lam=0, max_ngrads=None):

    if max_ngrads is None:
        max_ngrads = np.min([50 * n, 10**9])

    funcval = func_eval(x0) + lam / 2 * norm2(x0)
    funcval_record = [funcval]
    gradval = norm2(grad_eval(x0) + lam * x0)
    gradval_record = [gradval]
    ngrads_record = [0]
    ngrads = 0
    x = np.array(x0)
    tot_updates = int(np.ceil(max_ngrads / n))

    for iter in range(tot_updates):
        sgrad = sgrad_eval(xlist=[x], size=int(n)) + lam * x
        x -= eta * sgrad
        ngrads += int(n)
        ngrads_record.append(ngrads)
        funcval = func_eval(x) + lam / 2 * norm2(x)
        funcval_record.append(funcval)
        gradval = norm2(grad_eval(x) + lam * x)
        gradval_record.append(gradval)
        print('Number of gradients: {0}, Function value: {1}'.format(
            ngrads, funcval), '\n')

    return ngrads_record, funcval_record, gradval_record
