import numpy as np

class KalmanFilter:
    def __init__(self,x, y, z, noise,Pf, H, xobs,yobs,zobs):
        self.x = x
        self.y = y
        self.z = z
        self.noise = noise
        self.Pf = Pf
        self.H = H
        self.xobs = xobs
        self.yobs = yobs
        self.zobs = zobs

    def fit(self):
        #State vector
        _simstate = np.array([self.x, self.y, self.z])
        _obsstate = np.array([self.xobs, self.yobs, self.zobs])

        #Kalman gain
        _inv = np.dot(np.dot(self.H, self.Pf), self.H.T)
        for i in range(_inv.shape[0]):
            _inv[i,i] = _inv[i,i] + self.noise
        _inv = np.linalg.inv(_inv)
        _Kg = np.dot(np.dot(self.Pf, self.H.T), _inv)

        #Innovation
        _innov = np.dot(self.H, _obsstate) - np.dot(self.H, _simstate)

        #Update
        _simstate = _simstate + np.dot(_Kg, _innov)
        _Pa = self.Pf - np.dot(np.dot(_Kg, self.H), self.Pf)

        return _simstate[0], _simstate[1], _simstate[2], _Pa
