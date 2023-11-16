#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt

dirname = sys.argv[1]
dw_ps = 1.0
filename = dirname + "/last_bamp_s.log"
d_s = np.loadtxt(filename)
filename = dirname + "/last_bamp_a.log"
d_a = np.loadtxt(filename)
filename = dirname + "/last_bamp_t.log"
d_t = np.loadtxt(filename)

nbatch, N = d_a.shape
dw = dw_ps/(2.0*np.pi*2.99792e-2)
w = np.arange(N)*dw
d2_a = d_a**2
d2_s = d_s**2
d2_t = d_t**2
d2m_a = np.mean(d2_a, axis=0)
d2m_s = np.mean(d2_s, axis=0)
d2m_t = np.mean(d2_t, axis=0)

w_ps = np.arange(1, N)*dw_ps
corr = w_ps
Jw_a = d2m_a[1:]*corr
Jw_s = d2m_s[1:]*corr
Jw_t = d2m_t[1:]*corr
w = w[1:]

np.savetxt("Jomega.log", np.c_[w, Jw_a, Jw_s, Jw_t])


