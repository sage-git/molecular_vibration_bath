#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

h_const = 6.62607004e-34 # m*m*kg/s
kB = 1.38064852e-23 # m*m*kg/s/s/K
temperature = 3e5
beta = 1.0/temperature/kB # s*s/m/m/kg
hbeta = h_const*beta # s
hbeta_ps = hbeta*1e12 # ps
m_bath = 1e0
y_scaling = 10000

dw_ps = 1.0
filename = "last_bamp_s.log"
d_s = np.loadtxt(filename)
filename = "last_bamp_a.log"
d_a = np.loadtxt(filename)
filename = "last_bamp_t.log"
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
plt.plot(w, d2m_a, label="bath of antisymmetry")
plt.plot(w, d2m_s, label="bath of symmetry")
plt.plot(w, d2m_t, label="bath of bending")
plt.xlabel("wavenumber/cm$^{-1}$")
plt.legend()
plt.show()

w_ps = np.arange(1, N)*dw_ps
corr = w_ps
Jw_a = d2m_a[1:]*corr
Jw_s = d2m_s[1:]*corr
Jw_t = d2m_t[1:]*corr
ratio = max(np.max(Jw_a), np.max(Jw_s))/np.max(Jw_t)
ratio = (ratio // 100 )*100
ratio = 1 if ratio < 100 else ratio
ratio = 100 if 100 <= ratio < 500 else ratio
ratio = 500 if 500 <= ratio < 1000 else ratio
Jw_t = Jw_t*ratio
w = w[1:]
plt.plot(w, Jw_a/y_scaling, label="antisym. stretch")
plt.plot(w, Jw_s/y_scaling, label="sym. stretch")
plt.plot(w, Jw_t/y_scaling, label="bending x {}".format(ratio), color="black")
plt.xlabel("frequency/cm$^{-1}$")
plt.legend()
plt.show()
