#!/usr/bin/env python
# -*- coding: utf-8 -*-

# our imports
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter

import csv
import os 

import numpy as np
from numpy.linalg import norm
from numpy import pi, sign, fabs


from scipy.special import gamma
import scipy.stats as stats
from sklearn.manifold import TSNE
import sklearn.metrics as metrics
from sklearn.decomposition import PCA

import spacy
from spacy.language import Language

import time
import sys

import pickle

from trustmodels import *

import matplotlib.pyplot as plt

# some globals
usecuda = False # change this to True if you want cuda
usecuda = usecuda and torch.cuda.is_available()
dtype = torch.FloatTensor
if usecuda:
    dtype = torch.cuda.FloatTensor

include_prepreds = True
npcafeats = 4
pca = PCA(n_components=npcafeats)
dirname = os.path.realpath('..')


def loadData(dom="household", bound01=True):
    # constants
    
    data_filenames = {
        'driving': os.path.join(dirname, 'data', 'trust_transfer_driving_cleaned.csv'), 
        'household': os.path.join(dirname, 'data', 'trust_transfer_household_cleaned.csv'),
    }

    # the task labels match those in the paper. 
    task_labels = {
        'driving': ['0-0', 'C-5', 'C-2', 'C-4', 'C-1', 'C-6', 'C-3', 'D-6', 'D-1', 'D-4', 'D-3', 'D-5', 'D-2'],
        'household': ['0-0', 'A-5', 'A-3', 'A-6', 'A-1', 'A-4', 'A-2', 'B-4', 'B-2', 'B-5', 'B-1', 'B-6', 'B-3'],
    }

    labels = task_labels[dom]
    nc = 2  # number of classes
    neg = 0  # negative class
    pos = 1  # positive class

    # task parameters
    ntasks = 12
    m = 2  # latent dimensionality

    # observations
    maxobs = 2  # maximum number of observations per person

    maxnpred = 3
    J = 7
    N = 0

    with open(data_filenames[dom]) as csvfile:

        reader = csv.DictReader(csvfile)

        i = 0

        pretasks_pred = []
        tasks_pred = []

        pretrust_scores = []
        trust_scores = []

        SSFF = []

        nobs = []
        tasks_obs = []
        tasks_obs_perf = []
        nprepred = []
        npred = []

        for row in reader:
            # these are the predicted tasks (before and observations)
            pretasks_pred += [[int(row['C_ID']), int(row['D_ID']), int(row['E_ID'])]]
            tasks_pred += [[int(row['C_ID']), int(row['D_ID']), int(row['E_ID'])]]  

            # these are the scores before observations
            pretrust_scores += [[int(row['C1_rating']), int(row['D1_rating']), int(row['E1_rating'])]]
            trust_scores += [
                [int(row['C2_rating']), int(row['D2_rating']), int(row['E2_rating'])]]  

            # this tracks if the robot suceeds or fails
            SSFF += row['B_SF']

            if row['B_SF'] == '1':  # success scenario (robot succeeds)
                # nobs += [[0,2]]
                tasks_obs += [[int(row['A_ID']), int(row['B_ID'])]]
                tasks_obs_perf += [[1, 1]]
            else:  # failure (robot fails)
                # nobs += [[2,0]]
                tasks_obs += [[int(row['A_ID']), int(row['B_ID'])]]
                tasks_obs_perf += [[0, 0]]

            nprepred += [3]
            npred += [3]
            N += 1
            i += 1

    # N=32 # number of participants
    tasks_obs_perf = np.array(tasks_obs_perf)
    tasks_obs = np.array(tasks_obs)
    tasks_pred = np.array(tasks_pred)
    trust_scores = np.array(trust_scores)

    pretasks_pred = np.array(pretasks_pred)
    pretrust_scores = np.array(pretrust_scores)
    nparts = N  # number of participants
    print("Num participants: %d" % (nparts))
    
    def sigmoid(x):
          return 1 / (1 + np.exp(-x))
    
    if bound01: 
        pretrust_scores = sigmoid(pretrust_scores-3.5)
        trust_scores = sigmoid(trust_scores-3.5)#(trust_scores - 1) / 6

    data = {
        "tasks_obs_perf": tasks_obs_perf,
        "tasks_obs": tasks_obs,
        "tasks_pred": tasks_pred,
        "trust_scores": trust_scores,
        "pretasks_pred": pretasks_pred,
        "pretrust_scores": pretrust_scores,
        "nparts": nparts,
        "labels": labels,
    }

    return data, nparts


def recreateWordVectors(vectors_loc="wordfeats/glove.6B/glove.6B.50d.txt", save_loc="wordfeats"):

    lang = "en"
    if lang is None:
        nlp = Language()
    else:
        nlp = spacy.blank(lang)
    with open(vectors_loc, 'rb') as file_:
        header = file_.readline()
        # nr_row, nr_dim = header.split()
        nr_dim = 50
        nlp.vocab.reset_vectors(width=int(nr_dim))
        for line in file_:
            line = line.rstrip().decode('utf8')
            pieces = line.rsplit(' ', int(nr_dim))
            word = pieces[0]
            vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')
            nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab

    nlp.to_disk(save_loc)
    return


def loadWordFeatures(dom, loc="wordfeats", loadpickle=False, savepickle=False):
    if loadpickle:
        # we load saved word features from our pickle file
        # the word features were generated as below
        with open(os.path.join(dirname, 'data', 'wordfeatures.pkl'),'rb') as f:
            featdict = pickle.load(f)
        return featdict[dom]


    nlp = Language().from_disk(loc)

    taskwords = {
        'household': 
        [
            [' '],
            ['Pick and Place glass'],
            ['Pick and Place plastic can'],
            ['Pick and Place lemon'],
            ['Pick and Place plastic bottle'],
            ['Pick and Place apple'],
            ['Pick and Place plastic cup'],
            ['Navigate while avoiding moving people'],
            ['Navigate to the main room door'],
            ['Navigate while following a person'],
            ['Navigate to the dining table'],
            ['Navigate while avoiding obstacles'],
            ['Navigate to the living room']
        ],
        'driving':
        [
            [' '],
            ['Parking backwards cars and people around, misaligned'],
            ['Parking backwards empty lot, misaligned'],
            ['Parking backwards cars and people around, aligned'],
            ['Parking forwards empty lot, aligned'],
            ['Parking forwards cars and people around, misaligned'],
            ['Parking forwards empty lot, misaligned'],
            ['Navigating lane merge with other moving vehicles'],
            ['Navigating lane merge on a clear road'],
            ['Navigating traffic-circle with other moving vehicles'],
            ['Navigating traffic-circle on a clear road'],
            ['Navigating T-junction with other moving vehicles'],
            ['Navigating T-junction on a clear road'],
        ]
    }

    featdict = {}
    for d,task_word_list in taskwords.items():
        wordfeatures = []
        for i in range(len(task_word_list)):
            print(task_word_list[i][0])
            wordfeatures.append(nlp(task_word_list[i][0]).vector)

        wordfeatures = np.array(wordfeatures)
        featdict[d] = wordfeatures

    wordfeatures = featdict[dom]

    # save the data
    if savepickle:
        with open(os.path.join(dirname, 'data', 'wordfeatures.pkl'),'wb') as f:
            pickle.dump(featdict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return wordfeatures


def getInputRep(taskid, nfeats, reptype="1hot", feats=None):
    taskfeat = []
    if reptype == "1hot":
        taskfeat = np.zeros((nfeats))
        taskfeat[taskid] = 1
    elif reptype == "wordfeat":
        taskfeat = np.zeros((1, feats.shape[1]))
        taskfeat[:] = feats[taskid, :]
    elif reptype == "taskid":
        taskfeat = np.zeros((1, 1))
        taskfeat[:] = taskid
    elif reptype == "tsne":
        taskfeat = np.zeros((1, feats.shape[1]))
        taskfeat[:] = feats[taskid, :]
    elif reptype == "pca":
        taskfeat = np.zeros((1, npcafeats))
        taskfeat[:] = feats[taskid, :]
    else:
        print("ERROR!")
        raise NameError('no such reptype')
    return taskfeat


# transforms raw data into dataset usable by our models
def createDataset(data, reptype, allfeatures):
    # create dataset suitable for model
    nperf = 2  # number of performance outcomes (e.g., 2 - success, failure)
    obsseqlen = 2  # length of observation sequence
    predseqlen = 3  # length of prediction sequence

    tasks_obs_perf = data["tasks_obs_perf"]
    tasks_obs = data["tasks_obs"]
    tasks_pred = data["tasks_pred"]
    trust_scores = data["trust_scores"]
    pretasks_pred = data["pretasks_pred"]
    pretrust_scores = data["pretrust_scores"]
    nparts = data["nparts"]

    if reptype == "1hot":
        nfeats = 13
    elif reptype == "taskid":
        nfeats = 1
    elif reptype == "wordfeat":
        taskfeatures = allfeatures["wordfeat"]
        nfeats = taskfeatures.shape[1]
    elif reptype == "tsne":
        nfeats = 3
        taskfeatures = allfeatures["tsne"]
    elif reptype == "pca":
        nfeats = npcafeats
        taskfeatures = allfeatures["pca"]
    else:
        raise ValueError("no such reptype")
    ntasks = taskfeatures.shape[0]
    print("num features:", nfeats)
    print(taskfeatures.shape)
    # create 1-hot representation for tasks observed
    N = nparts
    tasksobsfeats = np.zeros((obsseqlen, nparts, nfeats))
    tasksobsids = np.zeros((obsseqlen, nparts, 1))
    for i in range(N):
        for t in range(obsseqlen):
            tasksobsids[t, i, :] = tasks_obs[i, t]
            tasksobsfeats[t, i, :] = getInputRep(tasks_obs[i, t], nfeats, reptype=reptype, feats=taskfeatures)
    tasksobsfeats = np.tile(tasksobsfeats, [1, predseqlen, 1])
    tasksobsids = np.tile(tasksobsids, [1, predseqlen, 1])

    tasksobsperf = np.zeros((obsseqlen, nparts, nperf))
    for i in range(N):
        for t in range(obsseqlen):
            tasksobsperf[t, i, tasks_obs_perf[i, t]] = 1
    tasksobsperf = np.tile(tasksobsperf, [1, predseqlen, 1])

    # create 1-hot representation for tasks to predict
    ntotalpred = int(np.prod(tasks_pred.shape))
    tasks_pred_T = tasks_pred.transpose().reshape([ntotalpred, 1])
    taskspredfeats = np.zeros((ntotalpred, nfeats))
    for t in range(ntotalpred):
        taskspredfeats[t, :] = getInputRep(tasks_pred_T[t][0], nfeats, reptype=reptype, feats=taskfeatures)

    trust_scores_T = trust_scores.transpose().reshape([ntotalpred, 1])
    trustpred = np.zeros(ntotalpred)
    for t in range(ntotalpred):
        trustpred[t] = trust_scores_T[t][0]

    taskpredids = tasks_pred_T
    taskpredtrust = trust_scores_T
    if include_prepreds:
        pretasksobsids = np.zeros((obsseqlen, N, 1))
        pretasksobsfeats = np.zeros((obsseqlen, N, nfeats))

        pretasksobsids = np.tile(pretasksobsids, [1, predseqlen, 1])
        pretasksobsfeats = np.tile(pretasksobsfeats, [1, predseqlen, 1])

        pretasksobsperf = np.zeros((obsseqlen, N, nperf))
        pretasksobsperf = np.tile(pretasksobsperf, [1, predseqlen, 1])

        # create 1-hot representation for pre-observation tasks to predict
        ntotalprepred = int(np.prod(pretasks_pred.shape))
        pretasks_pred_T = pretasks_pred.transpose().reshape([ntotalprepred, 1])
        pretaskspredfeats = np.zeros((ntotalprepred, nfeats))
        for t in range(ntotalprepred):
            pretaskspredfeats[t, :] = getInputRep(pretasks_pred_T[t][0], nfeats, reptype=reptype, feats=taskfeatures)

        pretrust_scores_T = pretrust_scores.transpose().reshape([ntotalprepred, 1])
        pretrustpred = np.zeros(ntotalprepred)
        for t in range(ntotalprepred):
            pretrustpred[t] = pretrust_scores_T[t][0]

        # create merged dataset
        tasksobsfeats = np.column_stack([pretasksobsfeats, tasksobsfeats])
        tasksobsperf = np.column_stack([pretasksobsperf, tasksobsperf])
        taskspredfeats = np.concatenate([pretaskspredfeats, taskspredfeats])
        trustpred = np.concatenate([pretrustpred, trustpred])

        tasksobsids = np.column_stack([pretasksobsids, tasksobsids])
        taskpredids = np.concatenate([pretasks_pred_T, tasks_pred_T])
        taskpredtrust = np.concatenate([pretrust_scores_T, trust_scores_T])

    # ok, I got too lazy to create a dict, using a tuple for now.
    dataset = (tasksobsfeats, tasksobsperf, taskspredfeats,
               trustpred, tasksobsids, taskpredids, taskpredtrust, data["labels"])

    return dataset


def computeTSNEFeatures(wordfeatures):
    tsnefeatures = TSNE(n_components=3, perplexity=5).fit_transform(np.array(wordfeatures))
    tsnefeatures = tsnefeatures / np.max(tsnefeatures) * 3
    return tsnefeatures


def computePCAFeatures(wordfeatures):
    global npcafeats
    pca = PCA(n_components=npcafeats)
    pca.fit(wordfeatures)
    return pca.transform(wordfeatures)


# pval is the validation proportion
def getTrainTestValSplit(data, dataset, splittype, excludeid=None, pval=0.1, nfolds=10):
    tasksobsfeats, tasksobsperf, taskspredfeats, trustpred, tasksobsids, taskpredids, taskpredtrust, labels = dataset

    obsseqlen = 2  # length of observation sequence
    predseqlen = 3  # length of prediction sequence

    # tasks_obs_perf = data["tasks_obs_perf"]
    tasks_obs = data["tasks_obs"]
    tasks_pred = data["tasks_pred"]
    # trust_scores = data["trust_scores"]
    pretasks_pred = data["pretasks_pred"]
    # pretrust_scores = data["pretrust_scores"]
    nparts = data["nparts"]

    ntotalpred = trustpred.shape[0]
    ntotalprepred = int(np.prod(pretasks_pred.shape))
    tasks_pred_T = tasks_pred.transpose().reshape([int(np.prod(tasks_pred.shape)), 1])

    # trust_scores_T = trust_scores.transpose().reshape([ntotalpred, 1])
    # pretrust_scores_T = pretrust_scores.transpose().reshape([ntotalprepred, 1])
    pretasks_pred_T = pretasks_pred.transpose().reshape([int(np.prod(pretasks_pred.shape)), 1])

    trainids = []
    testids = []
    valids = []

    if splittype == "random":
        # Random splits
        # split into test and train set
        ntrain = int(np.floor(0.9 * ntotalpred))
        rids = np.random.permutation(ntotalpred)
        trainids = rids[0:ntrain]
        testids = rids[ntrain + 1:]

        nval = int(np.floor(pval * ntrain))
        valids = trainids[0:nval]
        trainids = np.setdiff1d(trainids, valids)

    elif splittype == "3participant":
        
        ntestparts = int(nparts/nfolds)
        partid = excludeid*ntestparts
        print("Num Test Participants: ", ntestparts)
        partids = [] #[partid, partid+1, partid+2]
        
        for i in range(ntestparts):
            partids += [partid + i]
        
        # ridx = np.random.permutation(nparts)
        # for i in range(ntestparts):
        #     partids += [ridx[i]]        
    
        if include_prepreds:
            sidx = 0
            eidx = predseqlen * 2
        else:
            sidx = 0
            eidx = predseqlen 
            
        for partid in partids:
            for i in range(sidx, eidx):
                testids += [i * nparts + partid]
        
        trainids = np.setdiff1d(range(ntotalpred), testids)

        ntrain = len(trainids)
        nval = int(np.floor(pval * ntrain))
        arr = np.arange(ntrain)
        rids = np.random.permutation(arr)
        valids = trainids[rids[0:nval]]
        # print("valids", valids)
        trainids = np.setdiff1d(trainids, valids)
        # print(trainids)        
    elif splittype == "LOOtask":
        # note that task ids range from 1 to nparts-1
        # remove all participants who observed the task

        taskid = excludeid
        print(labels[excludeid])
        partids = []
        testids = []
        for i in range(nparts):
            for t in range(obsseqlen):
                if tasks_obs[i, t] == taskid:
                    partids += [i]

        preshapesize = 0
        if include_prepreds:
            preshapesize = pretasks_pred_T.shape[0]
                    
        if include_prepreds:
            for partid in partids:
                for i in range(predseqlen):
                    testids += [i * nparts + partid + preshapesize]

        # remove all training samples where the prediction (pre and post were the task)
        if include_prepreds:
            for i in range(pretasks_pred_T.shape[0]):
                if pretasks_pred_T[i] == taskid:
                    testids += [i]

        
        for i in range(tasks_pred_T.shape[0]):
            if tasks_pred_T[i] == taskid:
                testids += [preshapesize + i]  # adding the size of pretasks because we concatenate the vectors
                    
        
        testids = np.sort(np.unique(testids))
        trainids = np.setdiff1d(range(ntotalpred), testids)
        ntrain = len(trainids)
        nval = int(np.floor(pval * ntrain))
        rids = np.random.permutation(ntrain)
        valids = trainids[rids[0:nval]]
        
        
        trainids = np.setdiff1d(trainids, valids)

    tasksobsfeats_train = tasksobsfeats[:, trainids, :]
    tasksobsperf_train = tasksobsperf[:, trainids, :]
    taskspredfeats_train = taskspredfeats[trainids, :]
    trustpred_train = trustpred[trainids]

    tasksobsfeats_val = tasksobsfeats[:, valids, :]
    tasksobsperf_val = tasksobsperf[:, valids, :]
    taskspredfeats_val = taskspredfeats[valids, :]
    trustpred_val = trustpred[valids]

    tasksobsfeats_test = tasksobsfeats[:, testids, :]
    tasksobsperf_test = tasksobsperf[:, testids, :]
    taskspredfeats_test = taskspredfeats[testids, :]
    trustpred_test = trustpred[testids]

    expdata = {
        "tasksobsfeats_train": tasksobsfeats_train,
        "tasksobsperf_train": tasksobsperf_train,
        "taskspredfeats_train": taskspredfeats_train,
        "trustpred_train": trustpred_train,
        "tasksobsfeats_val": tasksobsfeats_val,
        "tasksobsperf_val": tasksobsperf_val,
        "taskspredfeats_val": taskspredfeats_val,
        "trustpred_val": trustpred_val,
        "tasksobsfeats_test": tasksobsfeats_test,
        "tasksobsperf_test": tasksobsperf_test,
        "taskspredfeats_test": taskspredfeats_test,
        "trustpred_test": trustpred_test,
        "labels": data["labels"]
    }

    return expdata


# Utility function: to plot the task representations
def plotEmbeddings(m, labels, reptype, feats, use_tsne=True):
    ntasks = len(labels)
    nfeats = feats.shape[0]
    taskreps = m.getTaskEmbeddings(nfeats, reptype=reptype, feats=feats)
    # print(taskreps)
    # print(taskreps.shape)
    taskreps = taskreps[1:, :]
    use_tsne = use_tsne
    if use_tsne:
        taskreps2d = TSNE(n_components=2, perplexity=2).fit_transform(taskreps)
    else:
        taskreps2d = taskreps

    fig, ax = plt.subplots()
    plt.scatter(taskreps2d[:, 0], taskreps2d[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')

    for i, txt in enumerate(labels[1:]):
        ax.annotate(txt, (taskreps2d[i, 0], taskreps2d[i, 1]))

    return taskreps2d


def getInitialProjectionMatrix(taskfeatures, reptype, taskrepsize, doplot=False, labels=None):
    # use dists to compute pre latent positions

    ntasks, nfeats = taskfeatures.shape

    class ProjMDS(torch.nn.Module):
        def __init__(self, nfeats, repsize=2):
            super(ProjMDS, self).__init__()
            # linearly transform one hot / features into a latent representation
            self.zrep = nn.Linear(nfeats, repsize, bias=False)
            # print(self.zrep.weight.shape)
            # self.zrep.weight = Parameter(torch.Tensor(np.random.rand(repsize,nfeats)*(1/50.)))
            # initz = np.random.rand(repsize,nfeats)
            # self.zrep.weight = Parameter(torch.Tensor(initz))

        def forward(self, pairs, taskfeats):
            N = pairs.shape[0]
            preddist = Variable(torch.FloatTensor(np.zeros((N, 1))))
            k = 0
            for pair in pairs:
                i = int(pair.data[0])
                j = int(pair.data[1])
                xi = taskfeats[i]
                xj = taskfeats[j]
                # print(i, j)
                # print(xi)
                zi = self.zrep(xi)
                zj = self.zrep(xj)
                # print('zi', zi)
                # print('zj', zj)
                dij = torch.norm(zi - zj)
                # print(dij)
                preddist[k] = dij
                k += 1

            return preddist

    # reduction to 2D
    # taskreps2d = TSNE(n_components=2, perplexity=2).fit_transform(np.array(taskreps.data))
    taskreps2d = TSNE(n_components=taskrepsize, perplexity=5).fit_transform(np.array(taskfeatures))
    dists = metrics.pairwise.pairwise_distances(taskreps2d / np.max(taskreps2d))
    # dists = metrics.pairwise.pairwise_distances(taskfeatures)

    featpairs = []
    featdists = []
    for i in range(ntasks):
        for j in range(i, ntasks):
            featpairs += [[i, j]]
            featdists += [dists[i, j]]

    featpairs = np.array(featpairs)
    featdists = np.array(featdists)
    featdists = np.power(featdists, 2.0)

    # first match locally to get the initial transformation matrix
    inppairs = Variable(torch.FloatTensor(featpairs), requires_grad=False)
    inpdistlist = Variable(torch.FloatTensor(featdists), requires_grad=False)

    alltasks1hot = np.zeros((ntasks, ntasks))
    for i in range(ntasks):
        alltasks1hot[i, i] = 1

    inpalltasks = None
    if reptype == "1hot":
        inpalltasks = Variable(torch.FloatTensor(alltasks1hot), requires_grad=False)
    elif reptype == "wordfeat" or reptype == "tsne":
        inpalltasks = Variable(torch.FloatTensor(taskfeatures), requires_grad=False)

    mds = ProjMDS(nfeats, taskrepsize)

    learning_rate = 1e-1
    optimizer = torch.optim.LBFGS(mds.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(gpmodel.parameters(), lr=learning_rate)

    t0 = time.time()

    for t in range(50):

        # logloss = model(inptasksobs, inptasksperf, inptaskspred, outtrustpred)
        def closure():
            optimizer.zero_grad()
            preddists = mds(inppairs, inpalltasks)
            loss = torch.mean(torch.pow(preddists - inpdistlist, 2.0))
            loss.backward()
            return loss

        optimizer.step(closure)

        if t % 10 == 0:
            preddists = mds(inppairs, inpalltasks)
            loss = torch.mean(torch.pow(preddists - inpdistlist, 2.0))
            optimizer.zero_grad()
            print(t, loss.data[0])
            optimizer.zero_grad()

    t1 = time.time()
    print("Total time: ", t1 - t0)
    if doplot:

        taskreps = mds.zrep(inpalltasks)
        fig, ax = plt.subplots()
        plt.scatter(taskreps[:, 0], taskreps[:, 1])
        plt.gca().set_aspect('equal', adjustable='box')

        if labels is None:
            labels = [str(i) for i in range(13)]
        for i, txt in enumerate(labels):
            ax.annotate(txt, (taskreps[i, 0], taskreps[i, 1]))
        plt.show(block=True)

    return mds.zrep.weight.data

    
def getGPParams(mode):
    usepriormean = False
    usepriorpoints = False
    if mode == 1:
        usepriormean = True
    elif mode == 2:
        usepriorpoints = True
    elif mode == 3:
        usepriorpoints = True
        usepriormean = True
    else:
        print("No such mode. defaulting to regular gp")
    return (usepriormean,usepriorpoints)

def main(
        dom="driving",
        reptype="wordfeat",
        splittype="LOOtask",  
        excludeid=2,
        taskrepsize=2,
        modeltype="neural",
        gpmode = 0,
        pval=0.1,
        seed=0,
        nfolds=10
)   :
    
    
    modelname = modeltype + "_" + str(taskrepsize) + "_" + str(gpmode) + "_" + dom + "_" + splittype + "_" + str(excludeid)
    
    # check what kind of modifications to the GP we are using
    print("Modelname: ", modelname)
    usepriormean, usepriorpoints = getGPParams(gpmode)
    
    verbose = False

    torch.manual_seed(seed)  # set up our seed for reproducibility
    np.random.seed(seed)

    # load the data
    data, nparts = loadData(dom)
    # print(data)

    # recreate word vectors if needed
    # e.g., when you download new word features from glove.
    recreate_word_vectors = False
    if recreate_word_vectors:
        recreateWordVectors()

    # load word features 
    wordfeatures = loadWordFeatures(dom, loadpickle=True)
    print(wordfeatures.shape)
    
    # in the experiments in the paper, we use the word features directly. However, 
    # you can also use tsne or pca dim-reduced features. 
    tsnefeatures = computeTSNEFeatures(wordfeatures)
    pcafeatures = computePCAFeatures(wordfeatures)
    
    allfeatures = {"wordfeat": wordfeatures, "tsne": tsnefeatures, "pca": pcafeatures}

    # create primary dataset
    dataset = createDataset(data, reptype, allfeatures)

    # create dataset splits
    expdata = getTrainTestValSplit(data, dataset, splittype, excludeid=excludeid, pval=pval, nfolds=nfolds)
    
    nfeats = allfeatures[reptype].shape[1]

    # we don't use an initial projection matrix. You can substitute one here if you like
    Ainit = None 


    inptasksobs = Variable(dtype(expdata["tasksobsfeats_train"]), requires_grad=False)
    inptasksperf = Variable(dtype(expdata["tasksobsperf_train"]), requires_grad=False)
    inptaskspred = Variable(dtype(expdata["taskspredfeats_train"]), requires_grad=False)
    outtrustpred = Variable(dtype(expdata["trustpred_train"]), requires_grad=False)

    inptasksobs_val = Variable(dtype(expdata["tasksobsfeats_val"]), requires_grad=False)
    inptasksperf_val = Variable(dtype(expdata["tasksobsperf_val"]), requires_grad=False)
    inptaskspred_val = Variable(dtype(expdata["taskspredfeats_val"]), requires_grad=False)
    outtrustpred_val = Variable(dtype(expdata["trustpred_val"]), requires_grad=False)

    inptasksobs_test = Variable(dtype(expdata["tasksobsfeats_test"]), requires_grad=False)
    inptasksperf_test = Variable(dtype(expdata["tasksobsperf_test"]), requires_grad=False)
    inptaskspred_test = Variable(dtype(expdata["taskspredfeats_test"]), requires_grad=False)
    outtrustpred_test = Variable(dtype(expdata["trustpred_test"]), requires_grad=False)

    
    learning_rate = 1e-3

    if modeltype == "gp":
        learning_rate = 1e-1
        usepriormean = usepriormean
        obsseqlen = 2
        phiinit = 1.0
        weight_decay = 0.01 #0.01
        modelparams = {
            "inputsize": inptasksobs.shape[2],
            "reptype": reptype,
            "taskrepsize": taskrepsize,
            "phiinit": phiinit,
            "Ainit": None,# np.array(Ainit),
            "obsseqlen": obsseqlen,
            "verbose": verbose,
            "usepriormean":usepriormean,
            "usepriorpoints":usepriorpoints
        }
    elif modeltype == "neural":
        perfrepsize = taskrepsize
        numGRUlayers = 2
        nperf = 2
        weight_decay = 0.00
        modelparams = {
            "perfrepsize": perfrepsize,
            "numGRUlayers": numGRUlayers,
            "nperf": nperf,
            "verbose": verbose,
            "taskrepsize": taskrepsize,
            "Ainit": None, #np.array(Ainit), 
            "nfeats": inptasksobs.shape[2]
        }
    elif modeltype == "lineargaussian":
        obsseqlen = 2
        weight_decay = 0.01
        modelparams = {
            "inputsize": inptasksobs.shape[2],
            "obsseqlen": obsseqlen,
        }
    elif modeltype == "constant":
        obsseqlen = 2
        weight_decay = 0.01
        modelparams = {
            "inputsize": inptasksobs.shape[2],
            "obsseqlen": obsseqlen,
        }
    else:
        raise ValueError("No such model")

    verbose = False
    reportperiod = 1
    
    # these two parameters control the early stopping
    # we save the stopcount-th model after the best validation is achived
    # but keep the model running for burnin longer in case a better
    # model is attained
    if splittype=="3participant":
        stopcount = 3
        burnin = 50
    elif splittype == "LOOtask":
        stopcount = 3
        burnin = 50
    
    t0 = time.time()
    bestvalloss = 1e10
    
    modeldir = "savedmodels"
    
    for rep in range(1):
        print("REP", rep)
        model = initModel(modeltype, modelname, parameters=modelparams)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        #if modeltype == "neural"
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=10, max_eval=20)
        #optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
        counter = 0

        torch.save(model, os.path.join(modeldir, model.modelname + ".pth"))
        restartopt = False
        t = 1
        #l2comp = nn.L2Loss()
        while t < 500:

            def closure():
                N = inptaskspred.shape[0]
                predtrust = model(inptasksobs, inptasksperf, inptaskspred)
                predtrust = torch.squeeze(predtrust)
                # logloss = torch.mean(torch.pow(predtrust - outtrustpred, 2.0)) # / 2*torch.exp(obsnoise))
                loss = -(torch.dot(outtrustpred, torch.log(predtrust)) +
                         torch.dot((1 - outtrustpred), torch.log(1.0 - predtrust))) / N
                
                optimizer.zero_grad()
                loss.backward()
                return loss

            optimizer.step(closure)

            if t % reportperiod == 0:
                # compute training loss
                predtrust = model(inptasksobs, inptasksperf, inptaskspred)
                predtrust = torch.squeeze(predtrust)
                loss = -(torch.dot(outtrustpred, torch.log(predtrust)) +
                         torch.dot((1 - outtrustpred), torch.log(1.0 - predtrust))) / inptaskspred.shape[0]

                # compute validation loss
                predtrust_val = model(inptasksobs_val, inptasksperf_val, inptaskspred_val)
                predtrust_val = torch.squeeze(predtrust_val)
                valloss = -(torch.dot(outtrustpred_val, torch.log(predtrust_val)) +
                            torch.dot((1 - outtrustpred_val), torch.log(1.0 - predtrust_val))) / predtrust_val.shape[0]

                # compute prediction loss
                predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test))
                predloss = -(torch.dot(outtrustpred_test, torch.log(predtrust_test)) +
                             torch.dot((1 - outtrustpred_test), torch.log(1.0 - predtrust_test))) / predtrust_test.shape[0]


                #print(model.wb, model.wtp, model.trust0, model.sigma0)

                #check for nans
                checkval = np.sum(np.array(predtrust_test.data))
                if np.isnan(checkval) or np.isinf(checkval):
                    # check if we have already restarted once
                    if restartopt:
                        #we've already done this, fail out.
                        #break out.
                        print("Already restarted once. Quitting")
                        break

                    # reinitialize model and switch optimizer
                    print("NaN value encountered. Restarting opt")
                    model = initModel(modeltype, modelname, parameters=modelparams)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    t = 1
                    counter = 0
                    restartopt = True
                else:
                    # print(predtrust_test.data, outtrustpred_test.data)
                    mae = metrics.mean_absolute_error(predtrust_test.data, outtrustpred_test.data)

                    print(t, loss.data[0], valloss.data[0], predloss.data[0], mae)
                    optimizer.zero_grad()

                    # if validation loss has increased for stopcount iterations

                    augname = model.modelname + "_" + str(excludeid) + ".pth"
                    if valloss.data[0] <= bestvalloss:
                        torch.save(model, os.path.join(modeldir, augname) )
                        print(valloss.data[0], bestvalloss, "Model saved")
                        bestvalloss = valloss.data[0]
                        counter = 0
                    else:
                        if counter < stopcount and (valloss.data[0]-bestvalloss) <= 0.1:
                            torch.save(model, os.path.join(modeldir, augname))
                            print(valloss.data[0], bestvalloss, "Model saved : POST", counter)
                        counter += 1

            if counter >= stopcount and t > burnin:
                #torch.save(model, modeldir+ model.modelname + ".pth")
                break

            t = t + 1

    t1 = time.time()
    print("Total time: ", t1 - t0)
    model = torch.load(os.path.join(modeldir,  modelname + "_" + str(excludeid) + ".pth"))

    # make predictions using trained model and compute metrics
    predtrust_test = torch.squeeze(model(inptasksobs_test, inptasksperf_test, inptaskspred_test))

    res = np.zeros((predtrust_test.shape[0], 2))
    res[:, 0] = predtrust_test.data[:]
    res[:, 1] = outtrustpred_test.data[:]
    print(res)

    mae = metrics.mean_absolute_error(predtrust_test.data, outtrustpred_test.data)
    predloss = -(torch.dot(outtrustpred_test, torch.log(predtrust_test)) + torch.dot((1 - outtrustpred_test),
                                                                                     torch.log(1.0 - predtrust_test))) / \
               predtrust_test.shape[0]
    predloss = predloss.data[0]

    return (mae, predloss, res)


# In[176]:

if __name__ == "__main__":

    dom = sys.argv[1] #"household" or "driving"
    reptype = "wordfeat"
    splittype = sys.argv[2] #"3participant" or "LOOtask"
    modeltype = sys.argv[3] #"neural" or "gp"
    gpmode = int(sys.argv[4]) 
    taskrepsize = int(sys.argv[5])
    
    print(dom, modeltype, splittype, gpmode)
    
    if dom == "household" or dom == "driving":
        _, nparts = loadData(dom)
    else:
        raise ValueError("No such domain")

    ntasks = 13
    if splittype == "3participant":
        start = 0
        nfolds = 10
        end = nfolds
        pval = 0.15  # validation proportion
    elif splittype == "LOOtask":
        start = 1
        end = ntasks
        nfolds = ntasks
        pval = 0.15  # validation proportion
    elif splittype == "random":
        start = 0
        end = 10
        nfolds = 10
    else:
        raise ValueError("No such splittype")

    allresults = []
    print(start, end)
    for excludeid in range(start, end):
        print("Test id:", excludeid)
        result = main(
            dom=dom,
            reptype=reptype,
            splittype=splittype,
            excludeid=excludeid,
            taskrepsize=taskrepsize,
            modeltype=modeltype,
            gpmode=gpmode,
            pval=pval,
            seed=0,
            nfolds=nfolds
        )
        allresults += [result]


        # save to disk after each run
        print(result)
        resultsdir = "results"
        filename =  dom + "_" + modeltype + "_" + str(taskrepsize) + "_" + splittype + "_" + str(gpmode) + ".pkl"
        with open(os.path.join(resultsdir, filename), 'wb') as output:
            pickle.dump(allresults, output, pickle.HIGHEST_PROTOCOL)
