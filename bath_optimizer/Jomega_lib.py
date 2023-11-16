#!/usr/bin/python

import numpy as np

def J(omega, mass=1.0, gamma=1.0):
    return mass*omega*gamma*gamma/(omega*omega + gamma*gamma)

def complex_J(omega, gamma=100.0, w0s=[30, 100, 300, 600], c0s=[0.5, 2, 36, 100]):
    w2 = omega**2
    gw2 = gamma*w2
    ret = 1/(gw2 + gamma**2)
    for w0, c0 in zip(w0s, c0s):
        ret += c0/(gw2 + (w2 - w0**2)**2)
    return gamma*gamma*omega*ret