#!/usr/bin/python

import os
import sys
import pickle
from multiprocessing import Pool
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, convolve
from scipy.signal.windows import gaussian
from scipy import fftpack
from scipy.optimize import minimize

PS2CM = 1.0/(2.99792e-2)
FS2CM = 1.0/(2.99792e-5)

PS2FS = 1000
def mass_vib(a, b):
    return a*b/(a + b)

MASS_O = 16.0
MASS_H = 1.008
MASS_D = 2.01355
MASS_C = 12.01
MASS_N = 14.01


mass_dict = {
    "OH": mass_vib(MASS_O, MASS_H),
    "OD": mass_vib(MASS_O, MASS_D),
    "CN": mass_vib(MASS_C, MASS_N),
    "COM": 1.0,
}

# k constant 
k_constant_dict = {
    "SPCE_OH": 345000.0,
    "SPCE_OD": 345000.0,
    "AcCN_CN": 798809.28,
    "NaCN_CN": 992662.607877,
    "BROWNIAN_COM": 0.0,
}

def fit_least_square(args):
    # R: nstep, omega: nw, time: nstep
    R, omega, time, q = args
    nw = len(omega)
    nstep = len(time)
    phase = (omega.reshape(nw, 1))*(time.reshape(1, nstep))
    c = np.cos(phase)
    s = np.sin(phase)
    cq = c*q.reshape(1, nstep)
    sq = s*q.reshape(1, nstep)
    cs = np.r_[c, s, cq, sq].T
    def cost_func(params, x, y):
        c1c, c1s, c2c, c2s = params.reshape((4, nw))
        L1 = np.sqrt(c1c**2 + c1s**2) + np.sqrt(c2c**2 + c2s**2)
        dy = y - x.dot(params)
        return np.sum(dy**2) + np.sum(L1)
    def jac(params, x, y):
        c1c, c1s, c2c, c2s = params.reshape((4, nw))
        dy = y - x.dot(params)
        return  - 2.0*x*dy
        
        
    x, rnorm, rank, s = np.linalg.lstsq(cs, R)
    return x.reshape((4, nw))


def main(pklfile, bond_param, bond_atoms, nw, dw):
    xt_raw = None
    vt_raw = None
    t_raw = None

    with open(pklfile, "rb") as f:
        obj = pickle.load(f)
        xt_raw = obj["xt"].T
        vt_raw = obj["vt"].T
        t_raw = obj["t"]
    dt = t_raw[1] - t_raw[0]
    nsample, nstep = xt_raw.shape
    max_step = nstep

    xt_raw = np.c_[xt_raw[:max_step], xt_raw[-max_step:]]
    vt_raw = np.c_[vt_raw[:max_step], vt_raw[-max_step:]]
    dt = dt*PS2FS
    vt_raw = vt_raw/PS2FS

    mass = mass_dict[bond_atoms]
    k = k_constant_dict[bond_param + "_" + bond_atoms]/PS2FS**2

    xt_raw = xt_raw - np.mean(xt_raw, axis=1).reshape(-1, 1)

    Nsum = nsample
    # w = np.arange(max_step)*dw
    w = fftpack.fftfreq(n=max_step, d=dt)
    q = xt_raw[:Nsum, :max_step]
    p = vt_raw[:Nsum, :max_step]*mass
    F = - q*k
    dp = np.gradient(p, dt, axis=1)
    t = np.arange(max_step)*dt/PS2FS

    R = dp - F

    w_fit = np.arange(nw)*dw

    args = [(R[i, :], w_fit, t, q[i, :]) for i in range(Nsum)]

    with Pool(processes=8) as pool:
        ret = pool.map(fit_least_square, args)
        AB = np.array(ret)
        A_all = AB[:, 0, :]
        B_all = AB[:, 1, :]
        C_all = AB[:, 2, :]
        D_all = AB[:, 3, :]

    Amp_LL = np.sqrt(A_all**2 + B_all**2)
    Amp_NL = np.sqrt(C_all**2 + D_all**2)
    wcm = w_fit*PS2CM/2/np.pi

    plt.plot(wcm, np.mean(Amp_LL, axis=0))
    plt.show()
    plt.plot(wcm, np.mean(Amp_NL, axis=0))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True)
    parser.add_argument("-a", "--atoms", type=str, choices=["OH", "OD", "CN", "COM"], help="Atoms of the system", required=True)
    parser.add_argument("-p", "--parameter", type=str, choices=["SPCE", "TIP4P", "NaCN", "AcCN", "BROWNIAN"], help="Parameter name of MD", required=True)
    parser.add_argument("-n", "--nomega", type=int, default=200)
    parser.add_argument("-d", "--domega", type=float, default=5.0)
    args = parser.parse_args()

    pklfile = args.filename
    bond_atoms = args.atoms
    bond_param = args.parameter
    dw = args.domega
    nw = args.nomega

    if not os.path.exists(pklfile):
        print(pklfile, " does not exist.")
        exit()
    main(pklfile, bond_param, bond_atoms, nw, dw)
