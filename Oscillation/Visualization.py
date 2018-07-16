import numpy as np
import matplotlib.pyplot as plt

class Visualization():
    def __init__(self,obs, true, sim, da, name,xlim):
        self.obs = obs
        self.true = true
        self.sim = sim
        self.da = da
        self.name = name
        self.xlim = xlim

    def fit(self):
        fig = plt.figure(figsize=(12, 3)) # figureオブジェクト作成
        #plt.subplot(1,2,1)
        plt.plot(self.obs,color='red',label='Obs.')
        plt.plot(self.true,color='green',label='True')
        plt.plot(self.sim,color='blue',label='Sim.')
        plt.plot(self.da,color='purple',label='DA')
        plt.xlabel("Time Step",size=12)
        plt.ylabel(self.name,size=12)
        plt.xlim(0,self.xlim)
        plt.legend()
        plt.show()
    def rmse(self):
        _rmse_da= 0.0
        for i in range(len(self.da)):
            _rmse_da = _rmse_da + (self.true[i] - self.da[i])**2
        _rmse_da = np.sqrt(_rmse_da/len(self.da))
        _rmse_sim= 0.0
        for i in range(len(self.sim)):
            _rmse_sim = _rmse_sim + (self.true[i] - self.sim[i])**2
        _rmse_sim = np.sqrt(_rmse_sim/len(self.da))
        print ("RMSE of Simulation = ", _rmse_sim)
        print ("RMSE of DA result = ", _rmse_da)
