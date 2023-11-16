#!/usr/bin/python

from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ideal_traj_maker import IdealTrajMaker, J
from system_bath import system_bath
from init_params import SystemParameters

def main():
    params = SystemParameters()

    sys_mass = params.mass_system
    sys_k = params.k_system
    nstep = params.n_step
    dt = params.delta_t
    bath_mass = params.mass_bath
    nomega = params.nomega
    domega = params.domega
    gamma = params.gamma
    jp = params.J_param
    lr = params.learning_rate
    x0 = 0.0
    x_amp = params.x_amp
    v_amp = params.v_amp
    tmp0 = params.bath_temperature
    eq_step = params.equil
    Jomega = lambda w: jp*J(w, bath_mass, gamma)

    traj_maker = IdealTrajMaker(sys_mass, sys_k, x0, dt, nstep, bath_mass, nomega, domega, Jomega)
    # traj_maker = IdealTrajMaker(sys_mass, sys_k, x0, dt, nstep, bath_mass, nomega, domega, lambda w: 10.0*(w > 300))
    # traj_maker = IdealTrajMaker(sys_mass, sys_k, x0, dt, nstep, bath_mass, nomega, domega, lambda w: jp)

    fig, ax = plt.subplots(2, 1)
    fig.suptitle(params.arg_summary)
    line = None

    sum_C = np.zeros(nstep*2 - 1)
    dw = 2*np.pi/(dt*(nstep*2 - 1))
    W = np.arange(nstep)*dw

    ax[1].plot(W, Jomega(W))
    count = 0
    while True:
        x0 = (np.random.rand() - 0.5)*x_amp
        v0 = (np.random.rand() - 0.5)*v_amp
        ret = traj_maker(x0, v0, equil_step=eq_step, bath_temperature=tmp0, log_bath=False)
        x = ret["x"]

        C = np.correlate(x, x, mode="full")
        C = np.roll(C, len(C)//2)
        sum_C += C
        count += 1

        FC = np.fft.rfft(sum_C/count)
        rFC = np.real(FC[:nstep])

        if count == 1:
            line, = ax[0].plot(W, rFC)
            continue
        line.set_data(W, rFC)
        plt.pause(0.0001)


if __name__ == "__main__":
    main()
