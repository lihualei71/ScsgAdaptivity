import numpy as np
import pandas as pd
import sys
import argparse
from ast import literal_eval


def expand_grid(*args):
    mesh = np.meshgrid(*args)
    mesh = np.array([m.flatten() for m in mesh]).transpose()
    return(pd.DataFrame(mesh))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname",
                        help="Filename", default="params.txt")
    parser.add_argument("-a", "--append",
                        help="Filename", default="")
    setting = parser.parse_known_args()
    params = [literal_eval(par) for par in setting[1]]
    fname = setting[0].fname
    ifappend = bool(setting[0].append)
    df = expand_grid(*params)
    if (ifappend is False):
        np.savetxt(fname, df, fmt="%s")
    else:
        with open(fname, "ab") as f:
            np.savetxt(f, df, fmt="%s")
