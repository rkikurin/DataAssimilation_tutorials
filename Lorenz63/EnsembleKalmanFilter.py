import numpy as np

class EnsembleKalmanFilter:
    def __init__(self,ensemble, H, obs, noise):
        self.ensemble = ensemble
        self.H = H
        self.obs = obs
        self.noise = noise

    def fit(self):
        #Average
        EnsembleAverage = []
        for i in range(self.ensemble.shape[1]):
            EnsembleAverage.append(self.ensemble[:,i].mean())
       
        #Perturb
        for i in range(self.ensemble.shape[1]):
            Prtb = []
            for ens in range(self.ensemble.shape[0]):
                Prtb.append(self.ensemble[ens,i]-EnsembleAverage[i])
            if (i==0):
                EnsemblePrtb = Prtb
            else:
                EnsemblePrtb = np.vstack((EnsemblePrtb,Prtb))

        #covariance
        covariance = np.dot(EnsemblePrtb/(self.ensemble.shape[0]-1),EnsemblePrtb.T)

        #Inverse matrix
        inv = np.dot(self.H, covariance)
        inv2 = np.dot(inv,self.H.T)

        if(self.H.shape[0] != 1):
            for i in range(self.H.shape[0]):
                inv2[i,i]=inv2[i,i] + self.noise * self.noise
            invKG = np.linalg.inv(inv2)
        else:
            inv2 = inv2 + self.noise * self.noise
            invKG = 1.0/inv2
        invBef = np.dot(covariance,self.H.T)
        KG = np.dot(invBef,invKG)

        #Innovation
        EnsInnov = []
        for ens in range(self.ensemble.shape[0]):
            Innov = (np.dot(self.H, self.obs) - np.dot(self.H, self.ensemble[ens, :]) + np.random.normal(0,self.noise))
            EnsInnov.append(Innov)
        EnsInnov = np.array(EnsInnov)
        EnsInnov = EnsInnov.reshape(1,len(EnsInnov)) 
        Update = np.dot(KG,EnsInnov)
        return Update.T
