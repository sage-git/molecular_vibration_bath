#!/usr/bin/python

import tensorflow as tf

class InitStatePredictor(object):
    def __init__(self, nomega, domega, nstep, float_precision):
        self.nw = nomega
        self.dw = domega
        self.fp = float_precision
        self.nfeature = 1

        self.nh = self.nw
        self.rcell_list = [tf.nn.rnn_cell.GRUCell(self.nh, dtype=self.fp) for _ in range(2)]
        self.rnnlayer = tf.nn.rnn_cell.MultiRNNCell(self.rcell_list)
        self.N = nstep
        self.xtraj = tf.placeholder(self.fp, shape=[self.N, self.nfeature])
        self.vtraj = tf.placeholder(self.fp, shape=[self.N, self.nfeature])
        self.ftraj = tf.placeholder(self.fp, shape=[self.N, self.nfeature])
        self.W_out = tf.get_variable("W_output", shape=[self.nh, self.nw*2], dtype=self.fp)
        self.b_out = tf.get_variable("b_output", shape=[self.nw*2], dtype=self.fp)
    
    def __call__(self):
        """
        output: shape=(nbatch, 2*nomega)
        2*nomega: xbath and vbath
        """
        nbatch = 1
        input_data = tf.reshape(tf.stack([self.xtraj, self.vtraj, self.ftraj]), (nbatch, self.N, -1))
        input_data = tf.reverse(input_data, [1])
        initial_state = self.rnnlayer.zero_state(nbatch, self.fp)
        outputs, state = tf.nn.dynamic_rnn(self.rnnlayer, input_data, initial_state=initial_state, dtype=self.fp)
        ret = tf.tanh(tf.matmul(outputs[:, -1, :], self.W_out) + self.b_out)
        return ret
    
    def train_vals(self):
        ret = [self.W_out, self.b_out]
        for rcell in self.rcell_list:
            ret.extend(rcell.trainable_variables)
        return ret

def main():
    isp = InitStatePredictor(100, 0.1, 50, tf.float64)
    output = isp()
    print(output)
    print(isp.train_vals())

    

if __name__ == "__main__":
    main()