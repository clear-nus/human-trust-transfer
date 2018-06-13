#!/usr/bin/env python
# -*- coding: utf-8 -*-

# our imports
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter

import numpy as np
from numpy.linalg import norm

import csv

from scipy.special import gamma
from numpy import pi, sign, fabs

from sklearn.manifold import TSNE
import sklearn.metrics as metrics

import spacy
from spacy.language import Language

import time
import sys

import pickle

from trustmodels import *


#from matplotlib import cm
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


# some globals
usecuda = False
usecuda = usecuda and torch.cuda.is_available()
dtype = torch.FloatTensor
if usecuda:
    dtype = torch.cuda.FloatTensor


class NeuralTrustNet(torch.nn.Module):
    def __init__(self, modelname, nfeats, nperf, taskrepsize=5, perfrepsize=2, numGRUlayers=1, Zinit=None):
        super(NeuralTrustNet, self).__init__()
        repsize = taskrepsize + perfrepsize
        H = 15
        self.zrep = torch.nn.Sequential(
            torch.nn.Linear(nfeats, H),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H),
            torch.nn.Tanh(),
            torch.nn.Linear(H, taskrepsize)
        )
            
        if Zinit is not None:
            self.zrep.weight = Parameter(dtype(np.array(Zinit)))
            #print(self.zrep.weight.shape)

        self.alpha = Parameter(dtype(0.1*np.ones(1)))
        self.beta = Parameter(dtype(0.1*np.ones(1)))
        self.sqrt2 = np.sqrt(2)
        
        self.sigma = Parameter(dtype(np.zeros(1)))
        self.perfrep = nn.Linear(nperf, perfrepsize)
        self.rnn = nn.GRU(repsize, taskrepsize, numGRUlayers)
        # self.h0 = Variable(torch.zeros(numGRUlayers, ntotalpred, taskrepsize), requires_grad=False) #initial hidden layer
        self.obslin = torch.nn.Linear(1, 1)
        self.taskrepsize = taskrepsize
        self.modelname = modelname


    def forward(self, inptasksobs, inptasksperf, inptaskspred):

        # latent representation for tasks that the user observed
        zk = self.zrep(inptasksobs)
        zperf = self.perfrep(inptasksperf)
        zkperf = torch.cat((zk, zperf), 2)
        
        # latent representation for tasks to predict
        zkpred = self.zrep(inptaskspred)
        output, hn = self.rnn(zkperf)

        # tasks to predict
        ntotalpred = output.shape[1]
        
        pz = torch.bmm(
            output[-1].view(ntotalpred, 1, self.taskrepsize),
            zkpred.view(ntotalpred, self.taskrepsize, 1)
        )
        
        ztrust = 1.0 / (1.0 + torch.exp(-pz/torch.exp(self.sigma)))
        obstrust = torch.clamp(ztrust, 1e-2, 0.99)  # 6*ztrust + 1
        
        return obstrust

    def getTaskEmbeddings(self, ntasks, reptype="1hot", feats=None):
        taskreps = None
        if reptype == "1hot":
            alltasks1hot = np.zeros((1, ntasks, ntasks))
            for i in range(ntasks):
                alltasks1hot[0, i, i] = 1

            inpalltasks = Variable(dtype(alltasks1hot), requires_grad=False)

            taskreps = self.zrep(inpalltasks).data[0]
        elif reptype == "wordfeat":
            inpalltasks = Variable(dtype(feats), requires_grad=False)
            taskreps = self.zrep(inpalltasks)
        elif reptype == "tsne":
            inpalltasks = Variable(dtype(feats), requires_grad=False)
            taskreps = self.zrep(inpalltasks)
        else:
            print("Wrong!")
        return taskreps



class GPTrustTransfer(torch.nn.Module):
    def __init__(self, modelname,
                 inpsize,
                 reptype,
                 obsseqlen,
                 taskrepsize=2,
                 verbose=False,
                 A=None,
                 phiinit=None,
                 usepriormean=False,
                 usepriorpoints=False
                 ):

        super(GPTrustTransfer, self).__init__()

        # for debugging
        self.modelname = modelname
        self.returnerror = False
        self.reptype = reptype
        self.verbose = verbose
        # set the kernel function
        self.PROJKERNEL = 0
        self.ARDKERNEL = 1
        self.FAKERNEL = 2
        self.SEKERNEL = 3
        
        # if we use prior observations
        self.usepriorpoints = usepriorpoints
        
        # prior mean function
        self.usepriormean = usepriormean
        self.by = Parameter(dtype(np.eye(1))) # this is used for a constant 
        if usepriormean:
            Ay = np.random.randn(1, taskrepsize)*0.5 #0.5 works
            self.Ay = Parameter(dtype(np.array(Ay)))

        
        # self.kerneltype = self.PROJKERNEL
        # self.kerneltype = self.SEKERNEL
        # self.kerneltype = self.ARDKERNEL
        self.kerneltype = self.FAKERNEL

        if self.kerneltype == self.PROJKERNEL:
            self.kfunc = self.projkernel
        elif self.kerneltype == self.ARDKERNEL:
            self.kfunc = self.ardkernel
        elif self.kerneltype == self.FAKERNEL:
            # we do the projections on the outside. 
            self.kfunc = self.sekernel
        elif self.kerneltype == self.SEKERNEL:
            self.kfunc = self.sekernel

        self.obsseqlen = obsseqlen

        # set the kernel function parameters
        self.taskrepsize = taskrepsize
        
        if A is None:
            A = np.random.randn(taskrepsize,inpsize)*0.5
            #print(A)
            # A = np.random.randn(inpsize,taskrepsize) #np.ones((taskrepsize,inpsize))

        self.A = Parameter(dtype(np.array(A)))
        # self.A = Variable(dtype(np.array(A)), requires_grad=False)
        # self.s = Parameter(dtype(np.eye(1)))
        self.s = Variable(dtype(np.eye(1)), requires_grad=False)


        self.sigm = torch.nn.Sigmoid()

        self.noisevar = Parameter(dtype(np.eye(1)))  # *1e-1
        if phiinit is None:
            phiinit = 1.0  # 2.0 #-5.0
        phi = None
        if self.kerneltype == self.FAKERNEL or self.kerneltype == self.PROJKERNEL:
            phi = np.ones(taskrepsize) * phiinit  #
        elif self.kerneltype == self.ARDKERNEL:
            phi = np.ones(inpsize) * phiinit
        elif self.kerneltype == self.SEKERNEL:
            phi = np.array([1.0]) * phiinit  

        self.phi = Variable(dtype(phi), requires_grad=False)

        self.kparams = {'A': self.A, 's': self.s, 'phi': self.phi, 'noisevar': self.noisevar}

        self.sqrt2 = np.sqrt(2)
        self.minvar = Variable(dtype([1e-6]), requires_grad=False)

        self.obslin = torch.nn.Linear(1, 1)
        if True:
            self.obslin.weight = Parameter(dtype(np.ones((1, 1)) * 6.0))
            self.obslin.bias = Parameter(dtype(np.ones((1, 1))))
        #    self.obslin.requires_grad = False

        priorinit = 1.0 / np.sqrt(inpsize)  # (1.0/inpsize)#
        if self.kerneltype == self.FAKERNEL:
            self.priorsucc = Parameter(dtype(np.random.randn(1, taskrepsize))) * priorinit
            self.priorfail = Parameter(dtype(np.random.randn(1, taskrepsize))) * priorinit
        else:
            self.priorsucc = Parameter(dtype(np.random.randn(1, inpsize))) * priorinit
            self.priorfail = Parameter(dtype(np.random.randn(1, inpsize))) * priorinit

        priorweight = 1.0
        self.succ = Variable(dtype(np.eye(1) * priorweight), requires_grad=True)
        self.fail = Variable(dtype(np.eye(1) * -priorweight), requires_grad=True)

        self.fullsucc = Variable(dtype(np.eye(1)), requires_grad=False)
        self.fullfail = Variable(dtype(np.eye(1) * -1.0), requires_grad=False)

        self.inpsize = inpsize

        # constants     
        self.one = Variable(dtype(np.eye(1)), requires_grad=False)
        self.zero = Variable(dtype(np.eye(1)*0), requires_grad=False)

        self.useAlimit = False
        self.limiter = torch.nn.Tanh()
        self.Alimit = 1.0

        
    def getPriorMean(self,x):
        if self.usepriormean:
            return torch.dot(self.Ay, x) + self.by
        elif self.usepriorpoints:
            # zero mean function
            return Variable(dtype([0.0]))
        else:
            #constant mean
            return self.by 
        
    def forward(self, inptasksobs, inptasksperf, inptaskspred):
        N = inptasksobs.shape[1]
        predtrust = Variable(dtype(np.zeros((N, 1))), requires_grad=False)
        errors = Variable(dtype(np.zeros((N, 1))), requires_grad=False)
        if usecuda:
            predtrust = predtrust.cuda()

        Alp = self.A
        if self.useAlimit:
            Alp = self.limiter(self.A) * self.Alimit

        for i in range(N):
            # print(i)
            alpha, C, Q, bvs = None, None, None, None

            if self.kerneltype == self.FAKERNEL:
                if self.reptype == "1hot":
                    x = torch.matmul(Alp, inptasksobs[0, i, :], ).view(1, self.taskrepsize)
                else:
                    x = torch.matmul(Alp, inptasksobs[0, i, :]).view(1, self.taskrepsize)
            else:
                x = inptasksobs[0, i, :].view(1, self.inpsize)
                
                
            if (x == 0).all():
                if self.usepriorpoints:
                    alpha, C, bvs = self.GPupdate(self.priorsucc, self.succ, self.kfunc, self.kparams, alpha, C, bvs, rawx=None)
                    alpha, C, bvs = self.GPupdate(self.priorfail, self.fail, self.kfunc, self.kparams, alpha, C, bvs, rawx=None)
                noop = None
            else:
                # first update "priors"
                if self.usepriorpoints:
                    alpha, C, bvs = self.GPupdate(self.priorsucc, self.succ, self.kfunc, self.kparams, alpha, C, bvs, rawx=None)
                    alpha, C, bvs = self.GPupdate(self.priorfail, self.fail, self.kfunc, self.kparams, alpha, C, bvs, rawx=None)

                # then update observed sequence
                for t in range(self.obsseqlen):
                    if self.kerneltype == self.FAKERNEL:
                        if self.reptype == "1hot":
                            x = torch.matmul(Alp, inptasksobs[t, i, :]).view(1, self.taskrepsize)
                        else:
                            x = torch.matmul(Alp, inptasksobs[t, i, :]).view(1, self.taskrepsize)
                    else:
                        x = inptasksobs[t, i, :].view(1, self.inpsize)

                        #estdiff = self.getDifficulty(x)
                    if (inptasksperf[t, i, 0] == 1).all():
                        y = self.fullfail
                    else:
                        y = self.fullsucc

                    alpha, C, bvs = self.GPupdate(x, y, self.kfunc, self.kparams, alpha, C, bvs, rawx=inptasksobs[t, i, :])

            # perform prediction
            if self.kerneltype == self.FAKERNEL:
                if self.reptype == "1hot":
                    x = torch.matmul(Alp, inptaskspred[i, :]).view(1, self.taskrepsize)
                else:
                    x = torch.matmul(Alp, inptaskspred[i, :]).view(1, self.taskrepsize)
            else:
                x = inptaskspred[i, :].view(1, self.inpsize)
            ypred, psuccess, error = self.GPpredict(x, self.kfunc, self.kparams, alpha, C, bvs, rawx = inptaskspred[i, :])
            predtrust[i] = psuccess
            errors[i] = error

        obstrust = torch.clamp(predtrust, 1e-2, 0.99)
        if self.returnerror:
            return obstrust, errors
        return obstrust


    def ardkernel(self, x1, x2, kparams):

        s = kparams['s']
        phi = kparams['phi']
        d = (x1 - x2)
        # print(d)
        # k = d
        # phi = torch.clamp(phi, -10, 0.01)
        k = torch.dot(d, 1.0 / torch.exp(phi))
        # print(k)
        k = s * s * torch.exp(-torch.dot(k.view(-1), k.view(-1)))
        return k

    def sekernel(self, x1, x2, kparams):

        s = kparams['s']
        phi = kparams['phi']
        # d = torch.div((x1-x2), torch.pow(phi, 2.0))
        # d = (x1-x2)
        # phi = torch.clamp(phi, -10, 0.01)
        d = torch.div((x1 - x2), torch.exp(phi))
        # print(d)
        k = s * s * torch.exp(-torch.dot(d.view(-1), d.view(-1)))
        return k

    def projkernel(self, x1, x2, kparams):
        A = kparams['A']
        s = kparams['s']
        noisevar = kparams['noisevar']
        phi = kparams['phi']
        d = (x1 - x2)
        # print(d)
        k = torch.matmul(d, torch.t(A))
        # print(k)
        k = s * s * torch.exp(-torch.dot(k.view(-1), k.view(-1)))
        return k

    def getKernelMatrix(self, X, kfunc, kparams, X2=None):
        # noisevar = kparams['noisevar']
        if X2 is None:
            n = X.shape[0]
            K = Variable(dtype(np.zeros((n, n))))

            for i in range(n):
                for j in np.arange(i, n):
                    K[i, j] = kfunc(X[i], X[j], kparams)
                    if i != j:
                        K[j, i] = kfunc(X[i], X[j], kparams)
                    else:
                        K[i, j] = K[i, j] + 1e-6
            return K
        else:
            n1 = X.shape[0]
            n2 = X2.shape[0]
            K = Variable(dtype(np.zeros((n1, n2))))
            # print(X2)
            for i in range(n1):
                for j in np.arange(n2):
                    K[i, j] = kfunc(X[i], X2[j], kparams)
            return K

        
    # update method = 'c' for classification, 'r' for regression
    def GPupdate(self, x, y, kfunc, kparams, alpha=None, C=None, bvs=None, update_method='c', rawx=None):

        kstar = kfunc(x, x, kparams)
        #noisevar = kparams['noisevar']
        
        noise = torch.exp(self.noisevar) + 0.01  # for numerical stability
        mx = self.getPriorMean(x)

        if bvs is None:
            # first ever update
            alpha = (y - mx) / kstar
            # print('alpha', alpha)
            C = Variable(dtype(np.zeros((1, 1))))
            if usecuda:
                C = C.cuda()
            C[0] = -1 / (kstar + noise)
            bvs = x
        else:
            # subsequent updates (projected process approximation)
            nbvs = bvs.shape[0]
            k = self.getKernelMatrix(bvs, kfunc, kparams, X2=x)
            m = torch.dot(k.view(-1), alpha.view(-1)) - mx
            Ck = torch.matmul(C, k)
            if self.verbose:
                print('Ck', Ck)
            s2 = kstar + torch.matmul(torch.t(k), Ck) + noise

            if (s2 < self.minvar).all():
                print("==== WARNING! =====")
                print('m', m, 's2', s2, 'k', k, 'Ck', Ck, 'alpha', alpha)
                s2[0] = self.minvar[0]

            sx = torch.sqrt(s2)
            z0 = m / sx

            z = y * z0

            Erfz = (torch.erf(z / self.sqrt2) + 1) / 2
            # print('Erfz', Erfz)
            regl = 1.0  # dampener: in case.
            constl = regl * 2.0 / np.sqrt(2 * np.pi)
            dErfz = torch.exp(-torch.pow(z, 2.0) / 2.0) * constl
            dErfz2 = dErfz * (-z) * regl

            if update_method == 'c':
                rclp = 1.0  # clamp value for numerical stability
                q = (y / sx) * (dErfz / Erfz)
                q = torch.clamp(q, -rclp, rclp)
                r = (1.0 / s2) * ((dErfz2 / Erfz) - torch.pow((dErfz / Erfz), 2.0));
                r = torch.clamp(r, -rclp, rclp)
                # print('r', r)
            else:
                # regression updates
                r = -1.0 / (s2);
                q = -r * (y - m);

            if (r != r).any() or (q != q).any():
                return (alpha, C, bvs)

            # grow and update alpha and C
            s = torch.cat((Ck.view(-1), self.one.view(-1))).view(1,-1)
            alpha = torch.cat((alpha.view(-1), self.zero.view(-1)))

            nbvs += 1
            bvs = torch.cat((bvs, x))

            zerocol = Variable(dtype(np.zeros((nbvs - 1, 1))), requires_grad=False)
            zerorow = Variable(dtype(np.zeros((1, nbvs))), requires_grad=False)

            if usecuda:
                zerocol = zerocol.cuda()
                zerorow = zerorow.cuda()

            C = torch.cat((C, zerocol), 1)
            C = torch.cat((C, zerorow))
            C = C + r * torch.matmul(s.t() , s)

            alpha = alpha + s * q

        return (alpha, C, bvs)

    def GPpredict(self, x, kfunc, kparams, alpha, C, bvs, rawx = None):

        kstar = kfunc(x, x, kparams)

        mx = self.getPriorMean(x)
        noise = torch.exp(self.noisevar) + 0.01

        s2 = 0.0
        if bvs is None:
            m = -mx
            s2 = kstar + noise
        else:
            k = self.getKernelMatrix(bvs, kfunc, kparams, X2=x)
            m = torch.dot(k.view(-1), alpha.view(-1)) - mx
            Ck = torch.matmul(C, k)
            s2 = kstar + torch.matmul(torch.t(k), Ck) + noise

        error = 0
        if (s2 < self.minvar).all():
            error = 1
            s2[0] = self.minvar[0]
        sx = torch.sqrt(s2)
        
        z = (m)/(sx)
        predErfz = (torch.erf(z / self.sqrt2) + 1.0) / 2.0
        if (predErfz > 0.5).all():
            ypred = 1.0
        else:
            ypred = -1.0
            
        return (ypred, predErfz, error)

    def getTaskEmbeddings(self, ntasks, reptype="1hot", feats=None):
        Alp = self.A
        if self.useAlimit:
            Alp = self.limiter(self.A) * self.Alimit
        taskreps = None
        if self.reptype == "1hot":
            alltasks1hot = np.zeros((1, ntasks, ntasks))
            for i in range(ntasks):
                alltasks1hot[0, i, i] = 1

            inpalltasks = Variable(dtype(alltasks1hot), requires_grad=False)
            taskreps = torch.matmul(inpalltasks, self.A).data[0]
        elif self.reptype == "wordfeat" or self.reptype == "tsne":
            inpalltasks = Variable(dtype(feats), requires_grad=False)
            if self.kerneltype == self.FAKERNEL:
                taskreps = torch.matmul(inpalltasks, torch.t(Alp))
            else:
                taskreps = torch.matmul(inpalltasks, torch.t(Alp))
        else:
            print("Wrong!")
        return taskreps

#Baseline trust models
class BaselineTrustModel(torch.nn.Module):
    def __init__(self, modelname, inpsize, obsseqlen, consttrust = False, verbose=False):
        super(BaselineTrustModel, self).__init__()
        
        # based off the OPTIMO paper by [Xu and Dudek 2015]
        self.trust0 = Parameter(dtype(np.random.rand(1,1))) # initial trust
        self.sigma0 = Parameter(dtype(np.abs(np.random.rand(1,1) + 1.0))) # initial uncertainty over trust
        self.wb = Parameter(dtype(np.random.rand(1,1)))
        self.wtp = Parameter(dtype(np.random.rand(1,1)))
        self.sigma_t = Parameter(dtype(np.abs(np.random.rand(1,1) + 1.0)))
        self.sigma_n = Parameter(dtype(np.abs(np.random.rand(1,1) + 1.0))) 
        #self.sigma2_n = torch.pow(self.sigma_n,2.0)
        
        self.consttrust = consttrust
        self.sqrt2 = np.sqrt(2)
        self.inpsize = inpsize
        self.obsseqlen = obsseqlen
        
        self.modelname = modelname
        
        self.succ = Variable(dtype(np.eye(1)), requires_grad=False)
        self.fail = Variable(dtype(np.eye(1)*-1.0), requires_grad=False)
        
    def trustUpdate(self, trust, sigma2, perf):
        trust = trust + self.wb + self.wtp*perf #no (perf_t - perf_t-1) since we only observe all success or all failures
        sigma2 = sigma2 + torch.pow(self.sigma_t,2.0)
        return (trust, sigma2)
    
    def trustPredict(self, trust, sigma2):
        #z = trust/torch.sqrt(self.sigma2_n + sigma2)
        z = trust/torch.sqrt(sigma2)
        predErfz = 1.0/(1.0 + torch.exp(-z))
        #predErfz = (torch.erf(z/self.sqrt2) + 1.0)/2.0
        return predErfz
        
    def forward(self, inptasksobs, inptasksperf, inptaskspred):
        N = inptasksobs.shape[1]
        predtrust = Variable(dtype(np.zeros((N,1))), requires_grad=False)

        for i in range(N): 
            trust = self.trust0
            sigma2 = torch.pow(self.sigma0, 2.0) 
            if self.consttrust:
                predtrust[i] = self.trustPredict(trust, sigma2)
            else:
                x = inptasksobs[0,i,:].view(1,self.inpsize)
            
                if not (x == 0).all():
                    # we have some observations
                    for t in range(self.obsseqlen):
                        x = inptasksobs[t,i,:].view(1,self.inpsize)
                        if (inptasksperf[t,i,0] == 1).all():
                            y = self.fail
                        else:
                            y = self.succ

                        trust, sigma2 = self.trustUpdate(trust, sigma2, y)
                predtrust[i] = self.trustPredict(trust, sigma2)
            
            
        obstrust = torch.clamp(predtrust, 1e-2, 0.99)
        return obstrust
    
    
    
def initModel(modeltype, modelname, parameters):
    if modeltype == "neural":
        perfrepsize = parameters["perfrepsize"]
        numGRUlayers = parameters["numGRUlayers"]
        nperf = parameters["nperf"]
        nfeats = parameters["nfeats"]
        taskrepsize = parameters["taskrepsize"]
        Ainit=parameters["Ainit"]
        model = NeuralTrustNet(modelname, nfeats, nperf, taskrepsize=taskrepsize,
                               perfrepsize=perfrepsize, numGRUlayers=numGRUlayers,
                               Zinit = Ainit
                               )
    elif modeltype == "gp":
        inputsize = parameters["inputsize"]
        taskrepsize = parameters["taskrepsize"]
        phiinit = parameters["phiinit"]
        Ainit = parameters["Ainit"]
        verbose = parameters["verbose"]
        reptype = parameters["reptype"]
        obsseqlen = parameters["obsseqlen"]
        usepriormean = parameters["usepriormean"]
        usepriorpoints = parameters["usepriorpoints"]
        model = GPTrustTransfer(modelname, inputsize,
                                reptype=reptype,
                                obsseqlen=obsseqlen,
                                taskrepsize=taskrepsize,
                                A=Ainit,
                                phiinit=phiinit,
                                verbose=verbose,
                                usepriormean=usepriormean,
                                usepriorpoints=usepriorpoints
                                )
    elif modeltype == "lineargaussian":
        inputsize = parameters["inputsize"]
        obsseqlen = parameters["obsseqlen"]
        model = BaselineTrustModel(modelname, inputsize, obsseqlen, consttrust=False)
    elif modeltype == "constant":
        inputsize = parameters["inputsize"]
        obsseqlen = parameters["obsseqlen"]
        model = BaselineTrustModel(modelname, inputsize, obsseqlen, consttrust=True)
    else:
        raise ValueError("No such model")
    return model

