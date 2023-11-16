#!/usr/bin/python

import os
import pickle
from collections import deque
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from system_bath import system_bath
from system_bath import gsa0, gta0, gts0, gssa0, gtta0, gtss0, gsaa0, gtaa0, gtts0
from init_params import SystemParameters
from get_execute_env import get_current_env

class Plotter(object):
    def __init__(self, session, range=100):
        self.sess = session
        self.iter_idx = deque([], range)
        self.L_log = deque([], range)
        self.lines = []
        self.range = range
        self.eval_xval = []
        self.xlog_gt = []
        self.fig, self.ax = plt.subplots(2, 3)
    
    def set_val(self, H, L, xlog):
        self.H = H
        self.L = L
        self.xlog = xlog

    def set_trained(self, ba, bp, c):
        self.ba = ba
        self.bp = bp
        self.c = c
    
    def set_xaxis_val(self, x_val):
        self.omega = x_val
    
    def set_xlog_gt(self, xlog_gt):
        self.xlog_gt = xlog_gt
    
    def set_title(self, title):
        self.fig.suptitle(title)
        
    def __call__(self, niter, feed_dict):
        w, H, L, ba, c, bp, xlog = self.sess.run([self.omega, self.H, self.L, self.ba, self.c, self.bp, self.xlog], feed_dict=feed_dict)
        w = np.abs(w[0, :])

        nstep = len(H)
        T = np.arange(nstep) * 1e-3

        self.L_log.append(L)
        self.iter_idx.append(niter)

        if len(self.lines) == 0:
            line, = self.ax[0, 0].plot(T, ba)
            self.lines.append(line)
            line, = self.ax[0, 1].plot(w, bp, linestyle="None", marker='o')
            self.lines.append(line)
            line, = self.ax[0, 2].plot(w, c)
            self.lines.append(line)
            line, = self.ax[1, 0].plot(T, H)
            self.lines.append(line)
            line, = self.ax[1, 1].plot(self.iter_idx, self.L_log)
            self.lines.append(line)
            line, = self.ax[1, 2].plot(T, self.xlog_gt)
            self.lines.append(line)
            plt.pause(0.0001)
            return

        self.lines[0].set_data(T, ba)
        self.ax[0, 0].set_ylim((ba.min(), ba.max()))
        self.lines[1].set_data(w, bp)
        self.ax[0, 1].set_xlim((0, w.max()))
        self.ax[0, 1].set_ylim((bp.min(), bp.max()))
        self.lines[2].set_data(w, c)
        self.ax[0, 2].set_xlim((0, w.max()))
        self.ax[0, 2].set_ylim((c.min(), c.max()))
        self.lines[3].set_data(T, H)
        self.lines[4].set_data(self.iter_idx, self.L_log)
        self.ax[1, 0].set_ylim((np.min(H), np.max(H)))
        self.ax[1, 1].set_ylim((np.min(self.L_log), np.max(self.L_log)))
        self.ax[1, 1].set_xlim((self.iter_idx[0], self.iter_idx[-1]))
        plt.pause(0.0001)
        np.savetxt("last_c.log", np.c_[w, c])


def main():
    with open("last_environment.log", "w") as f:
        f.write(get_current_env())

    params = SystemParameters()

    nstep = params.n_step

    bath_mass = params.mass_bath
    nomega = params.nomega
    domega = params.domega
    gamma = params.gamma
    lr = params.learning_rate
    seg_denom = params.segment_denom
    tmp0 = params.bath_temperature
    nsubstep = params.substep
    nbatch = params.nsample
    nminibatch = params.nbatch
    xt_raw = None
    vt_raw = None
    t_raw = None
    resume = params.resume_training

    traj_file = params.traj_file
    dt = 0.001
    step_all = nstep
    if os.path.exists(traj_file):
        with open(traj_file, "rb") as f:
            obj = pickle.load(f)
            xt_raw = obj["xt"]
            vt_raw = obj["vt"]
            t_raw = obj["t"]
            dt = t_raw[1] - t_raw[0]
            step_all = xt_raw.shape[0]
            nbatch = min(xt_raw.shape[1], nbatch)
    else:
        print("Error on load the trajectory file")
        return
    xO = xt_raw[:, :, :3]
    xH1 = xt_raw[:, :, 3:6]
    xH2 = xt_raw[:, :, 6:9]
    dx1 = xH1 - xO
    dx2 = xH2 - xO
    dr1 = np.linalg.norm(dx1, axis=2)
    dr2 = np.linalg.norm(dx2, axis=2)
    avg_x0 = np.mean(np.r_[dr1, dr2])
    theta = np.arccos(np.sum(dx1*dx2, axis=2)/dr1/dr2)
    avg_t0 = np.mean(theta)
    
    sys = system_bath(params.pmodel, False, nbatch, bath_mass, nomega, domega, gamma, tf.float32, substep=nsubstep,init_c_value=params.c_init, bath_temp=tmp0)
    # sys.x0 = avg_x0
    # sys.t0 = avg_t0
    Loss, dx_pred, dv_pred = sys.loss_function(dt, nstep, nstep_evolve=1)
    use_VSL = sys.use_VSL
    use_couple = sys.use_couple
    use_counter = sys.use_couterterm
    use_sqcouple = sys.use_sqcouple

    x0, t0 = sys.x0, sys.t0
    kr, kt = sys.kr, sys.kt
    gta, gts, gsa = sys.gta, sys.gts, sys.gsa
    gtts, gtss = sys.gtts, sys.gtss
    gtta, gtaa = sys.gtta, sys.gtaa
    gssa, gsaa = sys.gssa, sys.gsaa
    VSL_s, VSL_a, VSL_t = sys.VSL_s, sys.VSL_a, sys.VSL_t
    x_ph, v_ph = sys.x_sys, sys.v_sys
    bamp_s, bamp_a, bamp_t = sys.bamp_s, sys.bamp_a, sys.bamp_t
    b_w = sys.omega
    latent_vals = [bamp_s, bamp_a, bamp_t]
    train_vals = latent_vals + [kr, kt]
    if use_couple:
        train_vals.extend([gta, gts, gsa])
    if use_sqcouple:
        train_vals.extend([gtts, gtss, gtta, gtaa, gssa, gsaa,  ])
    if use_VSL:
        train_vals.extend([VSL_a, VSL_s, VSL_t])
    norm_loss = sys.norm_weight*(sys.L2_a + sys.L2_s + sys.L2_t)
    ib_start = sys.minibatch_start
    ib_end = sys.minibatch_end
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    opt_m = optimizer.minimize(tf.reduce_mean(Loss), var_list=train_vals)
    opt_test = optimizer.minimize(tf.reduce_mean(Loss), var_list=latent_vals)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    nst0 = nstep
    xtraj_raw = xt_raw[nst0:nst0+nstep, :nbatch, :].transpose((1, 0, 2))
    vtraj_raw = vt_raw[nst0:nst0+nstep, :nbatch, :].transpose((1, 0, 2))
    pot_log = "last_potential_params.log"
    couple_log = "last_coupling_params.log"
    interact_log = "last_interact_params.log"
    bampa_log = "last_bamp_a.log"
    bamps_log = "last_bamp_s.log"
    bampt_log = "last_bamp_t.log"
    loss_log = "last_loss.log"
    ckpt_path = "last_train.ckpt"
    gij_param = np.array([gts0, gtts0, gtss0,
                          gta0, gtta0, gtaa0,
                          gsa0, gssa0, gsaa0])

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        niter = 0 
        if resume:
            with open(loss_log, "r") as f:
                logline = f.readlines()[-1]
                niter = int(logline.split()[0])
        sess.run(init)
        sess.run([x0.assign(avg_x0), t0.assign(avg_t0)])
        plot = Plotter(sess)
        plot.set_val(tf.reduce_mean(dv_pred, axis=(0, 2)), tf.reduce_mean(Loss), tf.zeros(nstep))
        plot.set_trained(tf.reduce_mean(dx_pred, axis=(0, 2)), tf.random.uniform((nomega, )), tf.reduce_mean(sys.bamp_a, axis=0))
        # plot.set_xaxis_val((np.arange(nomega) + 1)*domega)
        plot.set_xaxis_val(sys.omega/(2.0*np.pi*2.99792e-2))
        plot.set_title(params.arg_summary)
        plot.set_xlog_gt(xtraj_raw[0, :nstep, 0])

        noise_amp = 0e-4
        with open(loss_log, 'w') as f:
            f.write("# niter  train   test")
        with open("last_dxv_mse.log", 'w') as f:
            f.write("")
        with open("last_loss_components.log", 'w') as f:
            f.write("# niter  loss  L_dx  L_dv  L_l2a  L_l2s  L_l2t\n")
        if resume:
            ckpt = tf.train.get_checkpoint_state("./")
            saver.restore(sess, ckpt.model_checkpoint_path)
        #else:
        #    os.makedirs(ckpt_path)

        if not (resume and os.path.exists(pot_log)):
            with open(pot_log, 'w') as f:
                f.write("# niter  x0  kr  t0  kt\n")

        if not (resume and os.path.exists(couple_log)):
            with open(couple_log, 'w') as f:
                f.write("# niter  gts  gtts  gtss  gta  gtta  gtaa  gsa  gssa  gsaa\n")
        if not (resume and os.path.exists(interact_log)):
            with open(interact_log, 'w') as f:
                f.write("# niter  VSL_t  VSL_s  VSL_a\n")

        norm_weight_decay = 0.99 
        norm_weight_decay = 0.99 
        while True:
            niter += 1
            vtraj = vtraj_raw + np.random.normal(scale=noise_amp)
            xtraj = xtraj_raw + np.random.normal(scale=noise_amp)
            feed_dict = {
                x_ph: xtraj.reshape(nbatch, nstep, 9),
                v_ph: vtraj.reshape(nbatch, nstep, 9),
                ib_start: 0, ib_end: nbatch,
                sys.norm_weight: norm_weight_decay**(niter - 1)
            }
           
            ibs = 0
            ibe = nminibatch
            loss_list = []
            test_loss = []
            dx_list = []
            dv_list = []
          
            L_dx = 0.0
            L_dv = 0.0
            L_norm = 0.0
            while ibs < nbatch:
                feed_dict[ib_start] = ibs
                feed_dict[ib_end] = ibe
                # test_batch = (ibe == nbatch) and (ibs > 0)
                test_batch = False
                vals = [None, Loss, dx_pred, dv_pred, sys.x_loss, sys.v_loss, norm_loss]
                vals[0] = opt_test if test_batch else opt_m 
                sess_ret = sess.run(vals, feed_dict=feed_dict)
                _, loss_n, dx_np, dv_np, L_dxi, L_dvi, L_ni = sess_ret
                if test_batch:
                    test_loss.append(loss_n)
                else:
                    loss_list.append(loss_n)
                    dx_list.append(dx_np)
                    dv_list.append(dv_np)
                    L_dx = L_dx + L_dxi
                    L_dv = L_dv + L_dvi
                    L_norm = L_norm + L_ni
                ibs = ibe
                ibe = min(ibe + nminibatch, nbatch)
            loss_n = np.mean(loss_list)
            loss_nt = np.mean(test_loss)
            dx_np = np.concatenate(dx_list, axis=0)
            dv_np = np.concatenate(dv_list, axis=0)
            with open("last_loss_components.log", "a") as f:
                f.write("{} {} {} {} {}\n".format(niter, loss_n, L_dx, L_dv, L_norm))

            if (niter - 1)%10 == 0:
                plot(niter, feed_dict)
                params_np = sess.run([x0, kr, t0, kt])
                bamp_np = sess.run([bamp_a, bamp_s, bamp_t])
                gij_np = sess.run([gts, gtts, gtss, gta, gtta, gtaa, gsa, gssa, gsaa])
                gij_np = gij_param*np.array(gij_np)
                VSL_np = sess.run([VSL_t, VSL_s, VSL_a])
                np.savetxt(bampa_log, bamp_np[0])
                np.savetxt(bamps_log, bamp_np[1])
                np.savetxt(bampt_log, bamp_np[2])
                dx_mse = np.mean(dx_np[:, :-1, :]**2)
                dv_mse = np.mean(dv_np[:, :-1, :]**2)
                  
                with open(loss_log, 'a') as f:
                    f.write("{} {} {}\n".format(niter, loss_n, loss_nt))
                with open("last_dxv_mse.log", "a") as f:
                    f.write("{} {} {}\n".format(niter, dx_mse, dv_mse))
                with open(pot_log, "a") as f:
                    f.write("{} {} {} {} {}\n".format(niter, *params_np))
                with open(couple_log, "a") as f:
                    f.write("{} {} {} {} {} {} {} {} {} {}\n".format(niter, *gij_np))
                with open(interact_log, "a") as f:
                    f.write("{} {} {} {}\n".format(niter, *VSL_np))
                saver.save(sess, "./" + ckpt_path)

if __name__ == "__main__":
    main()
