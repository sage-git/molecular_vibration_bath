#!/usr/bin/python

from math import sqrt, log
import random
import numpy as np

from Jomega_lib import J, complex_J


kB = 1.0

def sys_potential(x, k, math_library=np):
    # return k*(x**4 - x**2)
    return 0.5*k*x**2
    # return  - 160*math_library.cos(k*x)
    # return  math_library.cosh(k*x)

def sys_gradient(x, k, math_library=np):
    return k*x
    # return 2*k*(2*x**3 - x) 
    # return k*math_library.sinh(k*x)
    # return  160*k*math_library.sin(x)

class IdealTrajMaker(object):
    def __init__(self, mass, spring_const, x_equil, dt, nstep,
                 mass_bath, nomega, domega, thermal_bath_func=J):
        self.m = mass
        self.k = spring_const
        self.xeq = x_equil
        self.nw = nomega
        self.dw = domega
        self.dt = dt
        self.n = nstep
        self.xb = np.zeros(nomega)
        self.vb = np.zeros(nomega)
        self.w = (np.arange(nomega) + 1)*domega
        self.mb = np.ones(nomega)*mass_bath
        self.cb = thermal_bath_func(self.w)
    
    def temperature_kin(self, vb):
        return 0.5*np.mean(self.mb*vb*vb)/kB
    
    def temperature_vil(self, x, xb):
        dUdxb = self.mb*self.w*self.w*xb
        return 0.5*np.mean(dUdxb*xb)/kB
    
    def _set_bath_coord(self, x0, temperature, smooth_len=5):
        T = temperature
        if T <= 0.0:
            xb = np.zeros(self.nw)
            vb = np.zeros(self.nw)
            return xb, vb

        xb = np.random.rand(self.nw) - 0.5
        vb = np.random.rand(self.nw) - 0.5
        if smooth_len > 1:
            xb = np.convolve(xb, np.ones(smooth_len), mode="same")
            vb = np.convolve(vb, np.ones(smooth_len), mode="same")
        
        vb = self._set_bath_temperature_velocity(vb, temperature)
        xb = self._set_bath_temperature_virial(x0, xb, temperature)
        return xb, vb

    def _set_bath_temperature_velocity(self, vb, temperature):
        T_v = self.temperature_kin(vb)
        rT_v = temperature/T_v
        vb = np.sqrt(rT_v)*vb
        return vb
    
    def _set_bath_temperature_virial(self, x0, xb, temperature):
        T_c = self.temperature_vil(x0, xb)
        # solve a^2 mw^2x^2 - aqcx = DVN with a
        rT_c = temperature/T_c
        alpha = sqrt(rT_c)
        xb = xb*alpha
        return xb

    def __call__(self, x0, v0, bath_temperature=0.0, equil_step=None, log_H=False, log_bath=False, log_T=True):
        nequil = random.randint(1, self.n) if equil_step is None else equil_step
        xt = np.zeros(self.n)
        vt = np.zeros(self.n)
        Ht = np.zeros(self.n) if log_H else None
        xbt = np.zeros((self.n, self.nw)) if log_bath else None
        vbt = np.zeros((self.n, self.nw)) if log_bath else None
        Tt = np.zeros(self.n) if log_T else None
        x = x0
        v = v0
        xb, vb = self._set_bath_coord(x0, bath_temperature)
        for _ in range(nequil):
            x, v, xb, vb = self._step(x, v, xb, vb)
            if bath_temperature > 0.0:
                vb = self._set_bath_temperature_velocity(vb, bath_temperature)
        for i in range(self.n):
            xt[i] = x
            vt[i] = v
            if log_bath:
                xbt[i, :] = xb
                vbt[i, :] = vb
            if log_H:
                Ht[i] = self.hamiltonian(x, v, xb, vb)
            if log_T:
                Tt[i] = self.temperature_kin(vb)
            x, v, xb, vb = self._step(x, v, xb, vb)
        xt[-1] = x
        vt[-1] = v
        ret = {"x": xt, "v": vt}
        if log_T:
            Tt[-1] = self.temperature_kin(vb)
            ret["T"] = Tt
        if log_H:
            Ht[-1] = self.hamiltonian(x, v, xb, vb)
            ret["H"] = Ht
        if log_bath:
            xbt[-1, :] = xb
            vbt[-1, :] = vb
            ret["xb"] = xbt
            ret["vb"] = vbt
        return ret
    
    def _step(self, x, v, xb, vb):
        dt = self.dt

        xn = x + 0.5*dt*v
        xbn = xb + 0.5*dt*vb

        dx = xn - self.xeq
        dUdx = sys_gradient(dx, self.k) 
        dIdx = np.sum(self.cb*xbn)
        dIdxb = self.cb*dx
        dUdxb = self.mb*self.w*self.w*xbn

        vbn = vb - dt*(dUdxb)/self.mb
        vn = v - dt*(dUdx + dIdx)/self.m
        
        xn = xn + 0.5*dt*vn
        xbn = xbn + 0.5*dt*vbn

        return xn, vn, xbn, vbn
    
    def hamiltonian(self, x, v, xb, vb):
        dx = x - self.xeq
        H_sys = 0.5*self.m*(v**2) + sys_potential(dx, self.k)
        H_bath = 0.5*np.sum(self.mb*vb**2) + 0.5*np.sum(self.mb*self.w**2*xb**2)
        E_int = np.sum(self.cb*dx*xb)
        return H_sys + H_bath + E_int

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from init_params import SystemParameters

    params = SystemParameters()

    mass_sys = params.mass_system
    k = params.k_system
    nstep = params.n_step
    dt = params.delta_t
    mass_bath = params.mass_bath
    nomega = params.nomega
    domega = params.domega
    gamma = params.gamma
    jp = params.J_param
    tmp0 = params.bath_temperature
    eq_step = params.equil

    x0 = 0.0
    itm = IdealTrajMaker(mass_sys, k, x0, dt, nstep, mass_bath, nomega, domega, lambda w: 0.0)
    ret = itm(0.0, 1.0/mass_sys, equil_step=0, log_H=True)
    xs = ret["x"]
    vs = ret["v"]
    hs = ret["H"]
    C = np.correlate(xs, xs, mode="full")
    C = np.roll(C, len(C)//2)
    FCs = np.fft.rfft(C)

    # itm = IdealTrajMaker(1e-3, 2e2, 0.0, dt, nstep, 500, 1e0, lambda w: 10.0*(w > 200.0))
    # itm = IdealTrajMaker(mass_sys, k, x0, dt, nstep, mass_bath, nomega, domega, lambda w: jp*J(w, mass_bath, gamma))
    itm = IdealTrajMaker(mass_sys, k, x0, dt, nstep, mass_bath, nomega, domega, lambda w: jp*complex_J(w, gamma=gamma))
    # itm = IdealTrajMaker(mass_sys, k, x0, dt, nstep, mass_bath, nomega, domega, lambda w: 5.0)

    ret = itm(0.0, 1.0/mass_sys, bath_temperature=tmp0, equil_step=eq_step, log_H=True, log_bath=True)
    x = ret["x"]
    v = ret["v"]
    h = ret["H"]
    xb = ret["xb"]
    tmp = ret["T"]
    t = np.arange(nstep)*dt

    C = np.correlate(x, x, mode="full")
    C = np.roll(C, len(C)//2)
    FC = np.fft.rfft(C)

    plt.figure(1)
    plt.suptitle(params.arg_summary)
    ax1 = plt.subplot(4, 2, 1)
    plt.plot(t, x, label="x(t) w bath")
    plt.plot(t, xs, label="wo bath")
    plt.legend()
    plt.subplot(4, 2, 3, sharex=ax1)
    plt.plot(t, v, label="v(t) w bath")
    plt.plot(t, vs, label="wo bath")
    plt.legend()
    plt.subplot(4, 2, 5, sharex=ax1)
    plt.plot(t, h, label="H(t) w bath")
    # plt.plot(t, hs, label="wo bath")
    plt.legend()
    plt.subplot(4, 2, 7, sharex=ax1)
    plt.plot(t, tmp, label="bath temperature")
    plt.legend()
    # plt.plot(t, 0.5*k*xs**2)
    # plt.plot(t, 0.5*mass_sys*vs**2)
    # plt.plot(t, 0.5*k*xs**2 + 0.5*mass_sys*vs**2)
    ax2 = plt.subplot(4, 2, 2)
    plt.plot(itm.w, itm.cb, label=r'$c(\omega)$')
    plt.legend()
    plt.subplot(4, 2, 4, sharex=ax2)
    plt.plot(itm.w, xb[0, :], label=r'$x_b(0)$') 
    plt.legend()
    plt.subplot(4, 2, 6, sharex=ax2)
    plt.plot(itm.w, xb[-1, :], label=r'$x_b(d_t n_{step})$') 
    plt.legend()
    dw = 2*np.pi/(dt*len(C))
    W = np.arange(len(FC))*dw
    iWmax = len(W[W < np.max(itm.w)])
    plt.subplot(4, 2, 8)
    plt.plot(W[:iWmax], np.log(np.real(FC[:iWmax])), label=r'F.T. of $C(t)$ with bath') 
    plt.plot(W[:iWmax], np.log(np.real(FCs[:iWmax])), label=r'F.T. of $C(t)$ no bath') 
    plt.legend()
    plt.show()
    