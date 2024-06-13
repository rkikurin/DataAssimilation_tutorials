import numpy as np

class SquareRootFilter:
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
        
        #Observation Matrix
        Observe = []
        for i in range(self.ensemble.shape[0]):
            if (i==0):
                Observe = np.reshape(self.obs,(self.obs.shape[0],1))
            else:
                Observe = np.hstack((Observe,np.reshape(self.obs,(self.obs.shape[0],1))))
        
        #Perturbation Matrix
        # EnPrtb = emsemble.T - EnMean
        # Prtb = EnPrtb / sqrt(m-1)
        EnsembleMean = []
        for i in range(self.ensemble.shape[0]):
            if (i==0):
                EnsembleMean = np.reshape(EnsembleAverage,(self.ensemble.shape[1],1))
            else:
                EnsembleMean = np.hstack((EnsembleMean,np.reshape(EnsembleAverage,(self.ensemble.shape[1],1))))
        EnsemblePrtb = self.ensemble.T - EnsembleMean
        Prtb = EnsemblePrtb / np.sqrt(self.ensemble.shape[0]-1)
        
        #Weighting matrix
        # Prtb w w.T Prtb.T = ( E - KH ) Prtb Prtb.T
        # K = Prtb hpr.T Inverse[ hpr hpr.T + Inverse(rer) ]
        #   = Prtb Inverse[ E + hpr.T rer hpr ] hpr.T rer
        # hpr = H Prtb
        # => Prtb w w.T Prtb.T = Prtb [ E - Inverse[ E + hpr.T rer hpr ] hpr.T rer hpr ] Prtb.T
        # => w w.T = E - Inverse[ E + hpr.T rer hpr ] hpr.T rer hpr
        #          = Inverse[ E + hpr.T rer hpr ]
        #          = a.T binv binv.T a
        # => w = a.T binv
        # ==========================================
        # hpr = H Prtb
        hpr = np.dot(self.H, Prtb)
        rer = np.zeros((self.H.shape[0], self.H.shape[0]))
        # rer : Diagonal matrix : Inverse of observation covariance matrix
        #     : diag[ Inverse(rer) ] = noise * noise
        for i in range(self.H.shape[0]):
            rer[i,i] = 1.0 / (self.noise * self.noise)
        # aba = a.T binv binv.T a
        aba = np.dot(np.dot(hpr.T,rer),hpr)
        for i in range(aba.shape[0]):
            aba[i,i] = aba[i,i] + 1.0
        # Calculation eigen values and vectors
        # Recomend to use linalg.eigh, instead of linalg.eig
        # a   : eigen vectors
        # eig : eigen values
        eig, a = np.linalg.eigh(aba)
        # Set eigen value matrix => binv binv.T
        binv = np.zeros((aba.shape[0], aba.shape[0]))
        for i in range(aba.shape[0]):
            binv[i,i] = 1.0 / np.sqrt(eig[i])
        # Weight matrix : w = a binv a.T
        w = np.dot(np.dot(a,binv),a.T)

        #Kalman gain matrix
        # KG = Prtb Inverse[ E + hpr.T rer hpr ] hpr.T rer
        #    = Prtb ( w w.T ) hpr.T rer
        # ==========================================
        # aba2 = w w.T
        aba2 = np.dot(w,w.T)
        KG = np.dot(np.dot(np.dot(Prtb,aba2),hpr.T),rer)
        
        #Innovation
        Ensemble = np.dot(self.H,Observe) - np.dot(self.H,EnsembleMean)
        Ensemble = EnsembleMean + np.dot(KG,Ensemble) + np.sqrt(self.ensemble.shape[0]-1)*np.dot(Prtb,w)
        return Ensemble.T
