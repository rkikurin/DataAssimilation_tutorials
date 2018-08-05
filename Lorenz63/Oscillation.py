import numpy as np

class Oscillation:
    def __init__(self,x, v, noise,Pf=None, dt=0.01,k=0.5,mass=1.0,dump=0.3):
        self.x = x
        self.v = v
        self.noise = noise
        self.Pf = Pf
        self.dt = dt
        self.k = k
        self.mass = mass
        self.dump = dump
    def time_integration(self):
        self.xt = self.x + self.dt * self.v
        self.vt = - (self.k * self.dt / self.mass) * self.x + (1.0 - self.dump * self.dt / self.mass) * self.v
        return self.xt,self.vt
    def observation_noise(self):
        self.xo = self.xt + np.random.normal(0,self.noise)
        self.vo = self.vt #+ np.random.normal(0,self.noise)
        return self.xo, self.vo
    def transient_matrix(self):
        _M = np.array([[1.0,self.dt],
                       [- self.k * self.dt /self.mass,1.0 - self.dump * self.dt / self.mass]])
        _Ptmp = np.dot(_M, self.Pf)
        self.Pf = np.dot(_Ptmp, _M.T)
        return self.Pf
