#!/usr/bin/python

from __future__ import division
import numpy as np

# recommended output command for Gromacs traj:
# gmx trjconv -pbc mol -o traj_nopbc.trr
# gmx traj -f traj_nopbc.trr -ox -ov -fp -nojump

natom_mol = len(["O", "H", "H", "M"])
x0 = 0.09572
k0 = 502416.0

def xvg_file_reader(filename):
    ret = []
    with open(filename, "r") as f:
        for L in f:
            if L.startswith("@"):
                continue
            if L.startswith("#"):
                continue
            cols = L.strip().split()
            ret.append(cols[1:])
    nstep = len(ret)
    return np.array(ret, dtype=float).reshape(nstep, -1, 3)

class groFileLoader(object):
    def __init__(self, coord_file, veloc_file):
        self.c = xvg_file_reader(coord_file)
        self.v = xvg_file_reader(veloc_file)
        self.natom = self.c.shape[1]
        self.nstep = self.c.shape[0]
        self.nmol = self.c.shape[1]//natom_mol
    
    def _get_snippet(self, array, start, end, imol):
        iatom1 = imol*natom_mol
        if iatom1 < 0 or iatom1 >= self.natom - 3:
            raise IndexError
        if start < 0 or start >= self.nstep:
            raise IndexError
        if start >= end:
            raise IndexError
        if end <= 0 or end > self.nstep:
            raise IndexError
        ret = array[start:end, iatom1:iatom1 + 3, :]
        return ret
    
    def get_xcoord_snippet(self, start, end, imol):
        return self._get_snippet(self.c, start, end, imol)

    def get_vcoord_snippet(self, start, end, imol):
        return self._get_snippet(self.v, start, end, imol)
    

class OHTrajExtractor(object):
    def __init__(self, coord_file, veloc_file):
        c_h2o = xvg_file_reader(coord_file)
        v_h2o = xvg_file_reader(veloc_file)
        self.natom = c_h2o.shape[1]
        self.nstep = c_h2o.shape[0]
        self.nmol = self.natom//natom_mol
        self.nsys = self.nmol*2
        self.xtraj = np.zeros((self.nstep, self.nsys))
        self.vtraj = np.zeros((self.nstep, self.nsys))
        self._load(c_h2o, v_h2o)

    def _load(self, c, v):
        dr1 = c[:, 1::natom_mol, :] - c[:, 0::natom_mol, :]
        dr2 = c[:, 2::natom_mol, :] - c[:, 0::natom_mol, :]
        self.xtraj[:, 0::2] = np.linalg.norm(dr1, axis=2)
        self.xtraj[:, 1::2] = np.linalg.norm(dr2, axis=2)
        dv1 = v[:, 1::natom_mol, :] - v[:, 0::natom_mol, :]
        dv2 = v[:, 2::natom_mol, :] - v[:, 0::natom_mol, :]
        self.vtraj[:, 0::2] = np.sum(dr1*dv1, axis=2)
        self.vtraj[:, 1::2] = np.sum(dr2*dv2, axis=2)
        self.vtraj = self.vtraj/self.xtraj
        self.xtraj = self.xtraj - x0

if __name__ == "__main__":
    coord_f = "sample_cor.xvg"
    veloc_f = "sample_vel.xvg"
    loader = groFileLoader(coord_f, veloc_f)
    print(loader.get_xcoord_snippet(0, 10, 1))
#     print(loader.get_vcoord_snippet(0, 100, 3))
