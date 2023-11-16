#!/usr/bin/python

import sys
import pickle
import numpy as np

MAX_REMOVE_STEP = 100

def make_trajectory(pklfile):
    xt_raw = None
    vt_raw = None
    t_raw = None
    with open(pklfile, "rb") as f:
        obj = pickle.load(f)
        xt_raw = obj["xt"]
        vt_raw = obj["vt"]
        t_raw = obj["t"]
    nstep_raw, nsample_raw = xt_raw.shape
    dt = t_raw[1] - t_raw[0]
    nstep = nstep_raw - MAX_REMOVE_STEP
    nsample = nsample_raw
    x_samples = np.zeros((nstep, nsample))
    v_samples = np.zeros((nstep, nsample))
    for i in range(nsample_raw):
        start_t = 1
        while True:
            if vt_raw[start_t - 1, i]*vt_raw[start_t, i] <= 0.0:
                start_t += 1
                break
            start_t += 1
        end_t = start_t + nstep
        x_samples[:, i] = xt_raw[start_t:end_t, i]
        v_samples[:, i] = vt_raw[start_t:end_t, i]
    t = np.arange(nstep)*dt
    with open("traj_shift.pkl", "wb") as f:
        pickle.dump({"xt": x_samples, "vt": v_samples, "t": t}, f)


if __name__ == "__main__":
    if len(sys.argv)== 2:
        make_trajectory(sys.argv[1])