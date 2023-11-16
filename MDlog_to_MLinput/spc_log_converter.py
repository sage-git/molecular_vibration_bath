#!/usr/bin/python

import pickle
import numpy as np
from spc_traj_loader import AllCoordExtractor

dt = 0.001

def main():
    coord_f = "spce_cor.xvg"
    veloc_f = "spce_vel.xvg"
    ohte = AllCoordExtractor(coord_f, veloc_f)
    t = np.arange(ohte.nstep)*dt
    xt = ohte.xtraj[:, :]
    vt = ohte.vtraj[:, :]
    with open("traj.pkl", "wb") as f:
        pickle.dump({"xt": xt, "vt": vt, "t": t}, f)


if __name__ == "__main__":
    main()
