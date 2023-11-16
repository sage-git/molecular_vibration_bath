#!/usr/bin/python
import sys
import argparse

class SystemParameters(object):
    def __init__(self):
        parser = argparse.ArgumentParser("Harmonic bath oscillators",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        training_group = parser.add_argument_group("training")
        training_group.add_argument("-ss", "--nsample", metavar="N", type=int, default=1, help="sample size")
        training_group.add_argument("-nb", "--nbatch", metavar="n", type=int, default=1, help="Minibatch size")
        training_group.add_argument("-c", "--c_init", metavar="c0", type=float, default=1e-1, help="Initial guess value for c")
        training_group.add_argument("-lr", "--learning_rate", metavar="alpha", type=float, default=0.01, help="Learning rate for the optimizser")
        training_group.add_argument("-sd", "--segment_denom", metavar="D", type=int, default=2, help="Denominator of Nstep to segments")
        training_group.add_argument("-R", "--resume", action="store_true", help="Resume training from last log")

        system_group = parser.add_argument_group("system")
        system_group.add_argument("-N", "--nstep", metavar="N",  type=int, default=1000, help="Number of time steps")
        system_group.add_argument("-k", "--k_sys", metavar="k", type=float, default=1e2, help="Spring constant of the system")
        system_group.add_argument("-T", "--temperature", metavar="T", type=float, default=1e0, help="Bath temperature")
        system_group.add_argument("-M", "--m_sys", metavar="M", type=float, default=1e-3, help="Mass of the system")
        system_group.add_argument("-dt", "--delta_t", metavar="dt", type=float, default=1e-3, help="Time step")
        system_group.add_argument("-dw", "--delta_omega", metavar="dw", type=float, default=3e0, help="Frequency unit of bath oscillators")
        system_group.add_argument("-n", "--n_omega", metavar="n", type=int, default=300, help="Nnumber of bath oscillators")
        system_group.add_argument("-m", "--m_bath", metavar="m", type=float, default=1e0, help="Mass of the bath oscillators")
        system_group.add_argument("-ns", "--nsubstep", metavar="n", type=int, default=1, help="Number of steps between each trajectory log")

        sampling_group = parser.add_argument_group("sampling")
        sampling_group.add_argument("-e", "--equilibrium_step", metavar="S", type=int, default=0, help="Preparation time step for bath equilibrium")
        sampling_group.add_argument("-g", "--gamma", metavar="g", type=float, default=3e2, help="Gamma parameter of J(omega)")
        sampling_group.add_argument("-p", "--Jparam", metavar="c0", type=float, default=1e0, help="Multiply parameter for J(omega)")
        sampling_group.add_argument("-xa", "--x_amp", metavar="x0", type=float, default=1e0, help="Amplitude parameter for x(0)")
        sampling_group.add_argument("-va", "--v_amp", metavar="v0", type=float, default=1e0, help="Amplitude parameter for v(0)")
        sampling_group.add_argument("-pm", "--pot_model", metavar="s", type=str, default="sample", help="potential model for sample")
        sampling_group.add_argument("-tm", "--target_mode", metavar="target", type=str, default="sample", help="target mode")
        sampling_group.add_argument("-f", "--traj_file", metavar="f.pkl", type=str, default="sample", help="trajectory file")

        self.args = parser.parse_args()
        self.pmodel = self.args.pot_model.lower()
        self.tmode = self.args.target_mode

    @property
    def resume_training(self):
        return self.args.resume
    
    @property
    def segment_denom(self):
        return self.args.segment_denom
    
    @property
    def traj_file(self):
        return self.args.traj_file
    
    @property
    def mass_system(self):
        if self.tmode == "OH":
            return 16.0*1.0/(16.0 + 1.0)*self.args.m_sys
        if self.tmode == "OD":
            return 16.0*2.01355/(16.0 + 2.01355)*self.args.m_sys
        if self.tmode == "CN":
            return 12.01*14.01/(12.01 + 14.01)*self.args.m_sys
        return self.args.m_sys
    
    @property
    def mass_bath(self):
        return self.args.m_bath
    
    @property
    def k_system(self):
        if self.tmode == "brownian":
            self.args.k_sys = 0.0
        if self.pmodel == "tip4p":
            if self.tmode == "OH" or self.tmode == "OD":
                return 502416.0*self.args.m_sys
        if self.pmodel == "spce":
            if self.tmode == "OH" or self.tmode == "OD":
                return 345000.0*self.args.m_sys
        if self.pmodel == "accn_sol":
            if self.tmode == "CN":
                return 798809.28*self.args.m_sys
        if self.pmodel == "nacn_sol":
            if self.tmode == "CN":
                return 992662.607877*self.args.m_sys
        return self.args.k_sys
    
    @property
    def n_step(self):
        return self.args.nstep
    
    @property
    def nomega(self):
        return self.args.n_omega
    
    @property
    def domega(self):
        return self.args.delta_omega
    
    @property
    def bath_temperature(self):
        return self.args.temperature

    @property
    def nbatch(self):
        return min(self.args.nbatch, self.args.nsample)

    @property
    def nsample(self):
        return self.args.nsample
    
    @property
    def delta_t(self):
        return self.args.delta_t
        
    @property
    def gamma(self):
        return self.args.gamma
    
    @property
    def J_param(self):
        return self.args.Jparam
    
    @property
    def c_init(self):
        return self.args.c_init
    
    @property
    def equil(self):
        return self.args.equilibrium_step

    @property
    def learning_rate(self):
        return self.args.learning_rate
    
    @property
    def x_amp(self):
        return self.args.x_amp

    @property
    def v_amp(self):
        return self.args.v_amp
    
    @property
    def substep(self):
        return self.args.nsubstep
    
    @property
    def arg_summary(self):
        ret = r"{}".format(self.args)
        ret = ret.replace("Namespace", "")
        ret = ret.replace("delta_omega", '$d_\omega$')
        ret = ret.replace("n_omega", '$n_\omega$')
        ret = ret.replace("gamma", '$\gamma$')
        ret = ret.replace("temperature", '$T$')
        ret = ret.replace("delta_t", '$d_t$')
        ret = ret.replace("c_init", '$c_0$')
        ret = ret.replace("m_bath", '\n$m_b$')
        ret = ret.replace("m_sys", "M")
        ret = ret.replace("k_sys", "k")
        ret = ret.replace("x_amp", '$x(0)$')
        ret = ret.replace("v_amp", '$v(0)$')
        ret = ret.replace('learning_rate', 'lr')
        ret = ret.replace('equilibrium_step', '$N_{equil}$')
        return ret

if __name__ == "__main__":
    p = SystemParameters()
    print(p.arg_summary)
