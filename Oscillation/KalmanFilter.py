import numpy as np

class KalmanGain:
    def __init__(self,x, v, noise,Pf, H, xobs,vobs):
        self.x = x
        self.v = v
        self.noise = noise
        self.Pf = Pf
        self.H = H
        self.xobs = xobs
        self.vobs = vobs

    def fit(self):
        #State vector
        _simstate = np.array([self.x, self.v])
        _obsstate = np.array([self.xobs, self.vobs])

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

        return _simstate[0], _simstate[1], _Pa
