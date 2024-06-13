# coding: utf-8

import numpy as np

class Lorenz63:
    def __init__(self,x, y, z, noise,Pf=None, dt=0.001,s=10.0,r=28.0,b=8.0/3.0):
        self.x = x
        self.y = y
        self.z = z
        self.noise = noise
        self.Pf = Pf
        self.dt = dt
        self.s = s
        self.r = r
        self.b = b
    def time_integration(self):
        xs = self.x - self.s * (self.x - self.y) * self.dt
        ys = self.y + (- self.y - self.x * self.z + self.r * self.x) * self.dt
        zs = self.z + (self.x * self.y - self.b * self.z) * self.dt       
        return xs,ys,zs
    def observation_noise(self):
        self.xo = self.x + np.random.normal(0,self.noise)
        self.yo = self.y #+ np.random.normal(0,self.noise)
        self.zo = self.z #+ np.random.normal(0,self.noise)
        return self.xo, self.yo, self.zo
    def transient_matrix(self):
        _M = np.array([[1.0 - self.dt * self.s, self.dt * self.s, 0.0],
                       [self.dt * (self.r -self.z),1.0 - self.dt ,-self.dt * self.x],
                       [self.dt * self.y, self.dt * self.x, 1.0 - self.dt * self.b]])
        _Ptmp = np.dot(_M, self.Pf)
        self.Pf = np.dot(_Ptmp, _M.T)
        return self.Pf
