#!/usr/bin/python

import pickle
import numpy as np
from tip4p_traj_loader import OHTrajExtractor

dt = 0.0005

def main():
    coord_f = "sample_cor.xvg"
    veloc_f = "sample_vel.xvg"
    ohte = OHTrajExtractor(coord_f, veloc_f)
    t = np.arange(ohte.nstep)*dt
    xt = ohte.xtraj[:, :]
    vt = ohte.vtraj[:, :]
    with open("traj.pkl", "wb") as f:
        pickle.dump({"xt": xt, "vt": vt, "t": t}, f)


if __name__ == "__main__":
    main()