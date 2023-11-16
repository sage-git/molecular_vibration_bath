#!/usr/bin/python

from __future__ import division
import numpy as np

# recommended output command for Gromacs traj:
# gmx trjconv -pbc mol -o traj_nopbc.trr
# gmx traj -f traj_nopbc.trr -ox -ov -fp -nojump

natom_mol = len(["O", "H", "H"])
x0 = 0.1
k0 = 345000

d2o = False

mo = 15.9994
mh = 1.008
md = 2.01355
mh1 = md if d2o else mh
mh2 = md if d2o else mh

mwat = mh1 + mh2 + mo

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

class AllCoordExtractor(object):
    def __init__(self, coord_file, veloc_file):
        c_h2o = xvg_file_reader(coord_file)
        v_h2o = xvg_file_reader(veloc_file)
        self.natom = c_h2o.shape[1]
        self.nstep = min(c_h2o.shape[0], 2000)
        self.nmol = self.natom//natom_mol
        self.nsys = 9
        self.xtraj = np.zeros((self.nstep, self.nmol, self.nsys))
        self.vtraj = np.zeros((self.nstep, self.nmol, self.nsys))
        self._load(c_h2o[:self.nstep, :, :], v_h2o[:self.nstep, :, :])

    def _load(self, c, v):
        xO = c[:, 0::natom_mol, :]
        xH1 = c[:, 1::natom_mol, :]
        xH2 = c[:, 2::natom_mol, :]
        cm = (mo*xO + mh1*xH1 + mh2*xH2)/mwat
        xO = xO - cm
        xH1 = xH1 - cm
        xH2 = xH2 - cm

        vO = v[:, 0::natom_mol, :]
        vH1 = v[:, 1::natom_mol, :]
        vH2 = v[:, 2::natom_mol, :]
        vm = (mo*vO + mh1*vH1 + mh2*vH2)/mwat
        vO = vO - vm
        vH1 = vH1 - vm
        vH2 = vH2 - vm

        dxOH1 = xH1 - xO
        dxOH2 = xH2 - xO
        nvec = np.cross(dxOH1, dxOH2)
        nvec = nvec/np.linalg.norm(nvec, axis=2)[:, :, np.newaxis]
        nz = np.array([0., 0., 1.0]).reshape((1, 1, 3))
        cost = nvec[:, :, 2:3] # np.sum(cost*nz, axis=2)

        rotv = np.cross(nvec, nz)
        sint = np.linalg.norm(rotv, axis=2)[:, :, np.newaxis]
        rotv = rotv/sint

        def rodriguies_rot(v):
            k = rotv
            v_rot = v*cost
            v_rot += np.cross(k, v)*sint 
            v_rot += k*np.sum(k*v, axis=2)[:, :, np.newaxis]*(1 - cost)
            return v_rot

        xO = rodriguies_rot(xO)
        xH1 = rodriguies_rot(xH1)
        xH2 = rodriguies_rot(xH2)

        vO = rodriguies_rot(vO)
        vH1 = rodriguies_rot(vH1)
        vH2 = rodriguies_rot(vH2)

        self.xtraj[:, :, 0:3] = xO[:, :, :]
        self.xtraj[:, :, 3:6] = xH1[:, :, :]
        self.xtraj[:, :, 6:9] = xH2[:, :, :]

        self.vtraj[:, :, 0:2] = vO[:, :, :2]
        self.vtraj[:, :, 3:5] = vH1[:, :, :2]
        self.vtraj[:, :, 6:8] = vH2[:, :, :2]

class OTrajExtractor(object):
    def __init__(self, coord_file, veloc_file):
        c_h2o = xvg_file_reader(coord_file)
        v_h2o = xvg_file_reader(veloc_file)
        self.natom = c_h2o.shape[1]
        # self.nstep = c_h2o.shape[0]
        self.nstep = 2000
        self.nmol = self.natom//natom_mol
        self.nsys = (self.nmol - 1)*self.nmol//2
        self.xtraj = np.zeros((self.nstep, self.nsys))
        self.vtraj = np.zeros((self.nstep, self.nsys))
        self._load(c_h2o, v_h2o)

    def _load(self, c, v):
        ipair = 0
        for imol in range(self.nmol - 1):
            for jmol in range(imol + 1, self.nmol):
                r1 = c[:self.nstep, imol*natom_mol, :]
                r2 = c[:self.nstep, jmol*natom_mol, :]
                v1 = v[:self.nstep, imol*natom_mol, :]
                v2 = v[:self.nstep, jmol*natom_mol, :]
                dr = r1 - r2
                dv = v1 - v2
                self.xtraj[:, ipair] = np.linalg.norm(dr, axis=1)
                self.vtraj[:, ipair] = np.sum(dr*dv, axis=1)
                ipair += 1
        self.vtraj = self.vtraj/self.xtraj
        self.xtraj = self.xtraj - self.xtraj[0, :]

if __name__ == "__main__":
    coord_f = "spce_cor.xvg"
    veloc_f = "spce_vel.xvg"
    loader = groFileLoader(coord_f, veloc_f)
    print(loader.get_xcoord_snippet(0, 10, 1))
#     print(loader.get_vcoord_snippet(0, 100, 3))
