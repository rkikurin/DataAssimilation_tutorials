import numpy as np

class ParticleFilter:
    def __init__(self,ensemble, H, obs, noise):
        self.ensemble = ensemble
        self.H = H
        self.obs = obs
        self.noise = noise

    def fit(self):
        
        #PF likelihood
        pf_stddev = np.zeros((self.H.shape[0], self.H.shape[0]))
        for i in range(self.H.shape[0]):
            pf_stddev[i,i] = 1.0/self.noise
        
        #determinant
        determinant = (self.noise**self.H.shape[0])*(2.0*np.pi)**3.0
        
        likelihood = np.zeros(self.ensemble.shape[0])
        for ens in range(self.ensemble.shape[0]):
            pf_likelihood = np.zeros((self.H.shape[0], 1))
            
            for i in range(self.H.shape[0]):
                pf_likelihood[i,0] = self.obs[i] - self.ensemble[ens,i]
            
            lh = 0.0
            for i in range(self.H.shape[0]):
                lh = lh + pf_likelihood[i,0] * pf_stddev[i,i] * pf_likelihood[i,0]
            
            likelihood[ens] = np.log(determinant) - 0.5 * lh
        
        #PF weight
        maxlilkelihood = np.max(likelihood)
        
        allweights = 0.0
        weights = np.zeros(self.ensemble.shape[0])
        for ens in range(self.ensemble.shape[0]):
            weights[ens] = np.exp(likelihood[ens] - maxlilkelihood)
            allweights = allweights + weights[ens]
        
        for ens in range(self.ensemble.shape[0]):
            weights[ens] = weights[ens] / allweights
        
        #PF resampling
        zeta = np.zeros(self.ensemble.shape[0])
        eeta = np.zeros(self.ensemble.shape[0])
        isrc = np.zeros(self.ensemble.shape[0])
        nsample = np.zeros(self.ensemble.shape[0])
        izero = np.zeros(self.ensemble.shape[0])
        
        zeta[0] = weights[0]
        for ens in range(1, self.ensemble.shape[0]-1):
            zeta[ens] = weights[ens] + zeta[ens-1]
        zeta[-1] = 1.0
        
        for ens in range(self.ensemble.shape[0]):
            eeta[ens] = (ens - 0.5 + 1.0)/(self.ensemble.shape[0])

        nzero = 0
        aa = 0
        
        for ens in range(self.ensemble.shape[0]):
            if(aa < self.ensemble.shape[0]):
                while (eeta[aa]<zeta[ens]):
                    nsample[ens] = nsample[ens] + 1
                    aa = aa + 1
                    if(aa>self.ensemble.shape[0]-1):
                        break

            if(nsample[ens]==0):
                nzero = nzero + 1
                izero[nzero] = ens

        for ens in range(self.ensemble.shape[0]):
            isrc[ens] = ens
        
        aa = 0
        for ens in range(self.ensemble.shape[0]):
            while (nsample[ens]>1):
                isrc[np.int(izero[aa])] = ens
                aa = aa + 1
                nsample[ens] = nsample[ens] - 1
        
        np.random.seed(seed=0)  
        for ens in range(self.ensemble.shape[0]):
            for i in range(self.ensemble.shape[1]):
                self.ensemble[ens,i] = self.ensemble[np.int(isrc[ens]),i] + np.random.normal(0,0.001)        #perturbation
        
        return self.ensemble