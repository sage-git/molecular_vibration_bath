#!/usr/bin/python

import tensorflow as tf
import numpy as np

kB = 1.0e10
x0_spce = 0.1
x0_tip4p = 0.09572
t0_spce = 109.47
t0_tip4p = 104.52
kr_spce = 345000.0
kr_tip4p = 502416.0
kt_spce = 383.0
kt_tip4p = 628.02
mo = 15.9994
mh = 1.008
md = 2.01355

gsa0 = 100.0
gta0 = 10.0
gts0 = 300.0
gssa0 = 200000.0
gsaa0 = 200000.0
gtta0 = 100.0
gtaa0 = 1000.0
gtts0 = 500.0
gtss0 = 1000.0

poly2vs = {
  "D": 444.18825237,
  "a": 24.16198688,
  "c0": 888.2734862,
  "beta" : 260.07948667,
  "c2" : 210.25813654,
  "r0" : 0.09578399999715363,
  "t0" : 1.8240086945979024,
  "kt" : 400.691354,
  "kr" : 494669.373,
}

class system_bath(object):
    def __init__(self, water_param, deuterium, nbatch, mass_bath, nomega, domega, gamma, float_precision,
                 substep=1, init_c_value=10.0, bath_temp=0.1):
        self.kt0 = kt_spce
        self.kr0 = kr_spce
        self.x0 = x0_spce
        self.t0 = t0_spce
        self.use_poli2vs = False
        if water_param.lower() == "tip4p":
            self.kt0 = kt_tip4p
            self.kr0 = kr_tip4p
            self.x0 = x0_tip4p
            self.t0 = t0_tip4p
        if water_param.lower() == "poli2vs":
            self.kt0 = poly2vs["kt"]
            self.kr0 = poly2vs["kr"]
            self.x0 = poly2vs["r0"]
            self.t0 = poly2vs["t0"]
            self.use_poli2vs = True
        self.mo = mo
        self.mh = md if deuterium else mh
        self.r_m_oh = 1.0/self.mo + 1.0/self.mh
        self.m_oh = 1.0/self.r_m_oh
        self.kt = 1.0
        self.kr = 1.0

        self.fp = float_precision
        self.no = nomega
        self.nd = 9
        self.nbatch = nbatch
        w = (np.arange(nomega) + 1)*domega
        c_0 = init_c_value#*gamma**2/((w - w0)**2 + gamma**2)
        b_0 = np.exp( - w*w/bath_temp/kB)
        self.x0 = tf.get_variable("equil_x0", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=self.x0, dtype=self.fp)) 
        self.t0 = tf.get_variable("equil_t0", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=self.t0*np.pi/180.0, dtype=self.fp)) 
        self.kr = tf.get_variable("spring_r_bend", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=1.0, dtype=self.fp)) 
        self.kt = tf.get_variable("spring_k_bend", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=1.0, dtype=self.fp)) 

        self.m = tf.get_variable("bath_mass", shape=[nomega], dtype=self.fp, initializer=tf.initializers.constant(value=mass_bath*w, dtype=self.fp)) 
        self.c = tf.get_variable("system_coord_factor", shape=[nomega], dtype=self.fp, initializer=tf.initializers.constant(value=b_0, dtype=self.fp)) 

        self.omega = tf.get_variable("bath_frequency", dtype=self.fp, shape=[1, nomega], initializer=tf.initializers.constant(value=w, dtype=self.fp))
        self.bamp_t = tf.get_variable("bath_amplitude_theta", dtype=self.fp, shape=[nbatch, nomega], initializer=tf.initializers.constant(value=c_0, dtype=self.fp))
        self.bamp_s = tf.get_variable("bath_amplitude_sym", dtype=self.fp, shape=[nbatch, nomega], initializer=tf.initializers.constant(value=c_0, dtype=self.fp))
        self.bamp_a = tf.get_variable("bath_amplitude_assym", dtype=self.fp, shape=[nbatch, nomega], initializer=tf.initializers.constant(value=c_0, dtype=self.fp))
        self.x_sys = tf.placeholder(self.fp, shape=[nbatch, None, self.nd], name="system_coordinate")
        self.v_sys = tf.placeholder(self.fp, shape=[nbatch, None, self.nd], name="system_velocity")
        self.gsa = tf.get_variable("coupling_asym_sym", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=0.0, dtype=self.fp)) 
        self.gssa = tf.get_variable("coupling_ssa", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=0.0, dtype=self.fp)) 
        self.gsaa = tf.get_variable("coupling_saa", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=0.0, dtype=self.fp)) 
        self.gts = tf.get_variable("coupling_bend_sym", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=0.0, dtype=self.fp)) 
        self.gtts = tf.get_variable("coupling_tts", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=0.0, dtype=self.fp)) 
        self.gtss = tf.get_variable("coupling_tss", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=0.0, dtype=self.fp)) 
        self.gta = tf.get_variable("coupling_bend_asym", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=0.0, dtype=self.fp)) 
        self.gtta = tf.get_variable("coupling_tta", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=0.0, dtype=self.fp)) 
        self.gtaa = tf.get_variable("coupling_taa", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=0.0, dtype=self.fp)) 
        self.VSL_s = tf.get_variable("VSL/VLL_sym", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=1.0, dtype=self.fp)) 
        self.VSL_a = tf.get_variable("VSL/VLL_asym", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=1.0, dtype=self.fp)) 
        self.VSL_t = tf.get_variable("VSL/VLL_bend", shape=[], dtype=self.fp, initializer=tf.initializers.constant(value=1.0, dtype=self.fp)) 
        self.dt = 0
        self.nstep = 0
        self.b_temp = bath_temp
        self.subtic = substep
        self.minibatch_start = tf.placeholder(tf.int32, shape=[], name="minibatch_start")
        self.minibatch_end = tf.placeholder(tf.int32, shape=[], name="minibatch_end")
        self.norm_weight = tf.placeholder(self.fp, name="L_norm_weight")
        self.use_VSL = True
        self.use_couterterm = False
        self.use_couple = True
        self.use_sqcouple = True

    def V_interact(self, q, VSLLL):
        if self.use_VSL:
            return q + 0.5*VSLLL*q*q
        return q

    def V_interact_d(self, q, VSLLL):
        if self.use_VSL:
            return 1.0 + VSLLL*q
        return 1

    def force_sys(self, xO, xH1, xH2, xb_s, xb_a, xb_t, sumc_s, sumc_a, sumc_t):
        """
        xO, xH1, xH2: shape=(nbatch, timestep, dim)
        xb_s, xb_a, xb_t: shape=(nbatch, timestep, nomega)
        """
        dx1 = xH1 - xO
        dx2 = xH2 - xO
        vomega = tf.cross(dx1, dx2)
        vertv1 = tf.cross(vomega, dx1)
        vertv2 = tf.cross(dx2, vomega)
        n1 = vertv1/tf.norm(vertv1, axis=2, keepdims=True)
        n2 = vertv2/tf.norm(vertv2, axis=2, keepdims=True)
        dr1 = tf.norm(dx1, axis=2, keepdims=True)
        dr2 = tf.norm(dx2, axis=2, keepdims=True)
        u1 = dx1/dr1
        u2 = dx2/dr2
        cos_t = tf.reduce_sum(u1*u2, axis=2, keepdims=True)
        theta = tf.acos(cos_t)

        Fs1 = self.kr0*self.kr*(self.x0 - dr1)
        Fs2 = self.kr0*self.kr*(self.x0 - dr2)
        Fb = - self.kt0*self.kt*(self.t0 - theta)
        if self.use_poli2vs:
            D = poly2vs["D"]
            a = self.kr*poly2vs["a"]
            ddr1 = dr1 - self.x0
            ddr2 = dr2 - self.x0
            Fs1 = 2*a*D*(tf.exp(-2*a*ddr1) - tf.exp(-a*ddr1))
            Fs2 = 2*a*D*(tf.exp(-2*a*ddr2) - tf.exp(-a*ddr2))
#            Fb = self.kt0*self.kt*(cos_t - tf.cos(self.t0))*(tf.sin(theta) - tf.cos(self.t0))
        x_sym = 0.5*(dr1 + dr2) - self.x0
        x_asym = 0.5*(dr1 - dr2)
        x_bend = theta - self.t0

        Vint_s = self.V_interact_d(x_sym, self.VSL_s)
        Vint_a = self.V_interact_d(x_asym, self.VSL_a)
        Vint_t = self.V_interact_d(x_bend, self.VSL_t)

        f_sym  = Vint_s*tf.reduce_sum(xb_s, axis=2, keepdims=True)
        f_asym = Vint_a*tf.reduce_sum(xb_a, axis=2, keepdims=True)
        f_bend = Vint_t*tf.reduce_sum(xb_t, axis=2, keepdims=True)
        # the 3rd axis changes omega -> dim

        if self.use_couterterm:
            Vint_s = 2.0*Vint_s*self.V_interact(x_sym, self.VSL_s)
            Vint_a = 2.0*Vint_a*self.V_interact(x_asym, self.VSL_a)
            Vint_t = 2.0*Vint_t*self.V_interact(x_bend, self.VSL_t)
            f_sym =  f_sym  - sumc_s*Vint_s*np.pi*self.b_temp
            f_asym = f_asym - sumc_a*Vint_a*np.pi*self.b_temp
            f_bend = f_bend - sumc_t*Vint_t*np.pi*self.b_temp
        
        if self.use_couple: 
            gsa = self.gsa*gsa0
            gts = self.gts*gts0
            gta = self.gta*gta0
            f_sym  += gsa*x_asym + gts*x_bend 
            f_asym += gsa*x_sym  + gta*x_bend 
            f_bend += gts*x_sym  + gta*x_asym
        if self.use_sqcouple:
            gssa = self.gssa*gssa0
            gsaa = self.gsaa*gsaa0
            gtts = self.gtts*gtts0
            gtss = self.gtss*gtss0
            gtta = self.gtta*gtta0
            gtaa = self.gtaa*gtaa0
            f_sym  += 2.0*gssa*x_sym*x_asym + gsaa*x_asym*x_asym \
                    + 2.0*gtss*x_sym*x_bend + gtts*x_bend*x_bend
            f_asym += 2.0*gsaa*x_sym*x_asym + gssa*x_sym*x_sym \
                    + 2.0*gtaa*x_asym*x_bend + gtta*x_bend*x_bend
            f_bend += 2.0*gtts*x_bend*x_sym + gtts*x_sym*x_sym \
                    + 2.0*gtta*x_asym*x_bend + gtaa*x_asym*x_asym

        Fs1 = Fs1 + (f_sym + f_asym)
        Fs2 = Fs2 + (f_sym - f_asym)
        Fb = Fb + f_bend

        FH1 = u1*Fs1 + Fb*n1/dr1
        FH2 = u2*Fs2 + Fb*n2/dr2
        FO = - FH1 - FH2

        return FO, FH1, FH2
    
    def loss_function(self, dt, nstep, nstep_evolve=1, L1_regularize=False, L2_loss=False):
        sib = self.minibatch_start
        eib = self.minibatch_end
        self.dt = dt
        self.nstep = nstep

        nstep_calc = nstep - nstep_evolve
        xi = self.x_sys[sib:eib, :nstep_calc, :]
        vi = self.v_sys[sib:eib, :nstep_calc, :]
        w = self.omega
        T0 = tf.range(nstep_calc, dtype=self.fp)*dt
        T0 = tf.reshape(T0, (nstep_calc, 1))
        bp_s = tf.random.uniform((eib - sib, 1, self.no), minval=0, maxval=np.pi, dtype=self.fp)
        bp_a = tf.random.uniform((eib - sib, 1, self.no), minval=0, maxval=np.pi, dtype=self.fp)
        bp_t = tf.random.uniform((eib - sib, 1, self.no), minval=0, maxval=np.pi, dtype=self.fp)
        ba_s = tf.reshape(self.bamp_s[sib:eib, :], (-1, 1, self.no))
        ba_a = tf.reshape(self.bamp_a[sib:eib, :], (-1, 1, self.no))
        ba_t = tf.reshape(self.bamp_t[sib:eib, :], (-1, 1, self.no))
        sumc_s = tf.reduce_sum(ba_s**2, axis=2, keepdims=True)
        sumc_a = tf.reduce_sum(ba_a**2, axis=2, keepdims=True)
        sumc_t = tf.reduce_sum(ba_t**2, axis=2, keepdims=True)
        for subi in range(self.subtic*nstep_evolve):
            t = T0 + subi*self.dt/self.subtic
            t = tf.reshape(t, (1, nstep_calc, 1))
            xi = xi + vi*self.dt/self.subtic
            xb_s = ba_s*tf.sin(t*w + bp_s)
            xb_a = ba_a*tf.sin(t*w + bp_a)
            xb_t = ba_t*tf.sin(t*w + bp_t)
            xO = xi[:, :, :3]
            xH1 = xi[:, :, 3:6]
            xH2 = xi[:, :, 6:9]
            FxO, FxH1, FxH2 = self.force_sys(xO, xH1, xH2, xb_s, xb_a, xb_t, sumc_s, sumc_a, sumc_t)
            axO = FxO/self.mo
            axH1 = FxH1/self.mh
            axH2 = FxH2/self.mh
            Ai = tf.concat([axO, axH1, axH2], 2)
            vi = vi + Ai*self.dt/self.subtic

        dx = self.x_sys[sib:eib, nstep_evolve:nstep, :] - xi
        dv = self.v_sys[sib:eib, nstep_evolve:nstep, :] - vi
        self.x_loss = tf.reduce_mean(dx*dx)
        self.v_loss = tf.reduce_mean(dv*dv)
        L = self.x_loss + self.v_loss
        self.L1_a = self.L1_s = self.L1_t = tf.constant(0.0)
        self.L2_a = self.L2_s = self.L2_t = tf.constant(0.0)

        if L1_regularize:
            self.L1_a = tf.reduce_mean(tf.abs(self.bamp_a))
            self.L1_s = tf.reduce_mean(tf.abs(self.bamp_s))
            self.L1_t = tf.reduce_mean(tf.abs(self.bamp_t))
            L = L + self.norm_weight*(self.L1_a + self.L1_s + self.L1_t)
        if L2_loss:
            self.L2_a = tf.reduce_mean(self.bamp_a**2)
            self.L2_s = tf.reduce_mean(self.bamp_s**2)
            self.L2_t = tf.reduce_mean(self.bamp_t**2)
            L = L + self.norm_weight*(self.L2_a + self.L2_s + self.L2_t)
        paddings = tf.constant([[0, 0], [0, nstep_evolve], [0, 0]])
        return L, tf.pad(dx, paddings), tf.pad(dv, paddings)
    
if __name__ == "__main__":
    sbtest = system_bath(10, 0.001, 1000, 1.0, 200, 5.0, 100, tf.float64)
    L = sbtest.loss_function(0.001, 10, 1000)
    print(L)
    H, X = sbtest.hamiltonian_log(2, 100)
    print(H, X)
