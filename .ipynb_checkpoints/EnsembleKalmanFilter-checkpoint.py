#!/usr/bin/env python
# coding:UTF-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import signal, interpolate
from scipy import optimize
import random
import copy
import sys

class EnsembleKalmanFilter:
    def __init__(self,param_all,ens_no,sim,obs_data):
        self.param_all = param_all
        self.ens_no = ens_no

        self.sim = sim
        self.obs_data = obs_data

    def generation_ensemble(self):
        counter = 0
        param_store = np.zeros(self.param_all.shape[1])
        for ens in range(self.ens_no):
            param_store[0] = self.param_all[counter,0] #+ np.random.normal(0,0.2)
            param_store[1] = self.param_all[counter,1] #+ np.random.normal(0,0.1)
            param_store[2] = self.param_all[counter,2] #+ np.random.normal(0,0.1)
            param_store[3] = self.param_all[counter,3] #+ np.random.normal(0,0.1)
            param_store[4] = self.param_all[counter,4] #+ np.random.normal(0,0.1)
            param_store[5] = self.param_all[counter,5] #+ np.random.normal(0,0.1)
            param_store[6] = self.param_all[counter,6] #+ np.random.normal(0,0.01)
            param_store[7] = self.param_all[counter,7] #+ np.random.normal(0,0.01)
            param_store[8] = self.param_all[counter,8] #+ np.random.normal(0,0.01)
            param_store[9] = self.param_all[counter,9] #+ np.random.normal(0,0.01)
            param_store[10] = self.param_all[counter,10] #+ np.random.normal(0,0.01)
            param_store[11] = self.param_all[counter,11] #+ np.random.normal(0,0.01)
            param_store[12] = self.param_all[counter,12] #+ np.random.normal(0,0.01)
            param_store[13] = self.param_all[counter,13] #+ np.random.normal(0,0.01)
            param_store[14] = self.param_all[counter,14] #+ np.random.normal(0,0.01)
            param_store[15] = self.param_all[counter,15] #+ np.random.normal(0,0.01)
            param_store[16] = self.param_all[counter,16] #+ np.random.normal(0,0.01)
            param_store[17] = self.param_all[counter,17] #+ np.random.normal(0,0.01)
            param_store[18] = self.param_all[counter,18] #+ np.random.normal(0,0.01)
            param_store[19] = self.param_all[counter,19] #+ np.random.normal(0,0.01)
            param_store[20] = self.param_all[counter,20] #+ np.random.normal(0,0.01)
            param_store[21] = self.param_all[counter,21] #+ np.random.normal(0,0.01)
            param_store[22] = self.param_all[counter,22] #+ np.random.normal(0,0.01)

            counter = counter + 1
            if(counter==self.param_all.shape[0]-1):
                counter = 0

            if(ens==0):
                param_ensemble = param_store
            else:
                param_ensemble = np.vstack((param_ensemble,param_store))
        return param_ensemble

    def da(self):
        #Average
        EnsembleAverage = []
        for i in range(self.sim.shape[0]):
            EnsembleAverage.append(sum(self.sim[i,:])/float(self.sim.shape[1]))

        #Perturb
        for i in range(self.sim.shape[0]):
            Prtb = []
            for ens in range(self.sim.shape[1]):
                Prtb.append(self.sim[i,ens]-EnsembleAverage[i])
            if (i==0):
                EnsemblePrtb = Prtb
            else:
                EnsemblePrtb = np.vstack((EnsemblePrtb,Prtb))

        #covariance
        covariance = np.dot(EnsemblePrtb/(self.sim.shape[0]-1),EnsemblePrtb.T)

        #obsmatrix
        obssum = 0
        for i in range(len(self.obs_data)):
            if(self.obs_data[i]>0):
                obssum += 1

        counter = 0
        for i in range(len(self.obs_data)):
            obsHtmp = np.zeros(self.sim.shape[0])
            if(self.obs_data[i]>0):
                obsHtmp[i]=1.0
                if(counter == 0):
                    obsH = obsHtmp
                else:
                    obsH = np.vstack((obsH,obsHtmp))
                counter += 1
        obsH = obsH.T

        #Inverse matrix
        inv = np.dot(obsH.T,covariance)
        inv2 = np.dot(inv,obsH)

        obsvar = 1.0
        if(obssum != 1):
            for i in range(obssum):
                inv2[i,i]=inv2[i,i] + obsvar
            invKG = np.linalg.inv(inv2)
        else:
            inv2 = inv2 + obsvar
            invKG = 1.0/inv2
        invBef = np.dot(covariance,obsH)
        KG = np.dot(invBef,invKG)

        #Innovation
        EnsInnov = []
        counter = 0
        for i in range(len(self.obs_data)):
            Innov = []
            if(self.obs_data[i]>0):
                for ens in range(self.sim.shape[1]):
                    Innov.append(self.obs_data[i]-self.sim[i,ens]+ np.random.normal(0,1.0))

                if (counter==0):
                    EnsInnov = np.array(Innov)
                else:
                    EnsInnov = np.vstack((EnsInnov,Innov))
                counter += 1
        if(obssum!=1):
            Update = np.dot(KG,EnsInnov)
        else:
            Update = np.zeros([self.sim.shape[0],self.sim.shape[1]])
            for i in range(Update.shape[0]):
                for j in range(Update.shape[1]):
                    Update[i,j] = KG[i]*EnsInnov[j]
        return Update

    def likelihood(self):
        likelihood = []
        for ens in range(self.ens_no):
            diff = 0
            for i in range(len(self.obs_data)):
                diff = diff +(self.obs_data[i]-self.sim[i,ens])**2
            likelihood.append(diff)

        minvalue = 999999999
        for ens in range(len(likelihood)):
            if(minvalue>likelihood[ens]):
                minvalue=likelihood[ens]
                mindata=ens
        return mindata
