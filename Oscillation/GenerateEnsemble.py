import numpy as np

class GenerateEnsemble:
    def __init__(self,xsim,nens,noise):
        self.xsim = xsim
        self.nens = nens
        self.noise = noise

    def fit(self):
        
        ensemble = []
        for ens in range(self.nens):
            data = np.array([self.xsim[0] +  np.random.normal(0,self.noise),
                            self.xsim[1] +  np.random.normal(0,self.noise)])
            if(ens==0):
                ensemble = data
            else:
                ensemble = np.vstack((ensemble, data))
        return ensemble