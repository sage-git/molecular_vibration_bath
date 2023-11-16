#!/usr/bin/python

from collections import deque
import matplotlib.pyplot as plt
import numpy as np

class Plotter(object):
    def __init__(self, ncol, nrow, nrowsess=None, nrange=100):
        self.sess = sess
        self.range = nrange
        self.nrow = nrow
        self.ncol = ncol
        self.fig, self.ax = plt.subplot(self.nrow, self.ncol)
        self.lines = []
    
    def set_session(self, sess):
        self.sess = sess
    
    def set_val(self, key, x_val, y_val, icol, irow):
    
    def __call__(self, niter, feed_dict):
        if len(self.lines) == 0:
            line, = self
        