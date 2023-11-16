#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

h_const = 6.62607004e-34 # m*m*kg/s
kB = 1.38064852e-23 # m*m*kg/s/s/K
temperature = 3e5
beta = 1.0/temperature/kB # s*s/m/m/kg
hbeta = h_const*beta # s
hbeta_ps = hbeta*1e12 # ps
m_bath = 1e0

max_freq = 5000
ratio = 100

dw_ps = 1.0
filename = "Jomega_spce.log"
J_spce = np.loadtxt(filename)
filename = "Jomega_tip4p.log"
J_tip4p = np.loadtxt(filename)

y_scaling = 10000
J_spce = J_spce/y_scaling
J_tip4p = J_tip4p/y_scaling

N, _ = J_spce.shape
dw = dw_ps/(2.0*np.pi*2.99792e-2)
w = np.arange(N)*dw
w = w[w < max_freq]
N_max = len(w)

rcParams['figure.figsize'] = 5, 8

fig = plt.figure()
plt.subplots_adjust(hspace=0.001)

ax1 = plt.subplot(211)
ax1.plot(w, J_spce[:N_max, 1], label="antisym. stretch")
ax1.plot(w, J_spce[:N_max, 2], label="sym. stretch")
ax1.plot(w, J_spce[:N_max, 3]*ratio, label="bending $\\times$ {}".format(ratio), color="black")
#y_max = 52000/y_scaling
#ax1.set_ylim([-2000/y_scaling, y_max])
y_max = np.max(J_spce[:, 1:])*0.95
text_x = np.max(w)*0.0
text_y = y_max*0.90
if False:
    ax1in = ax1.inset_axes([4000, y_max*0.25, 1000, y_max*0.5], transform=ax1.transData) 
    ax1in.set_xlim([2800, 3500])
    ax1in.plot(w, J_spce[:N_max, 1])
    ax1in.plot(w, J_spce[:N_max, 2])

plt.text(text_x, text_y, "(a) SPC/E flexible")
plt.legend(bbox_to_anchor=(0.45, 1.2), loc='upper center', ncol=2)

ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(w, J_tip4p[:N_max, 1], label="antisym. stretch")
ax2.plot(w, J_tip4p[:N_max, 2], label="sym. stretch")
ax2.plot(w, J_tip4p[:N_max, 3]*ratio, label="bending $\\times$ {}".format(ratio), color="black")
text_y = np.max(J_tip4p[:, 1:])*0.95
plt.text(text_x, text_y, "(b) TIP4P")

xticks = ax1.get_xticklabels()
plt.setp(xticks, visible=False)

plt.xlabel("frequency/cm$^{-1}$")
plt.ylabel("$J(\\omega)$/arbitrary unit")
plt.show()
