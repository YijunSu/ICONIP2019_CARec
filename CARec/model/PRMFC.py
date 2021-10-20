# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import defaultdict
import time
import sys

class PRMFCModel(object):
    def __init__(self, numUsers, numItems, K=15,lamda_p = 0.01, lamda_q = 0.01,  learningRate = 0.005):
        self.K = K
        self._numUsers = numUsers
        self._numItems = numItems
        self._lamda_p = lamda_p
        self._lamda_q = lamda_q
        self._learningRate = learningRate
        self._users = set()
        self._items = set()
        self._Iu = defaultdict(set)
        self._Cu = defaultdict(list)
        self.U, self.L = None, None

    def save_model(self, path):
        ctime = time.time()
        print("Saving U and L...",)
        np.save(path + "U", self.U)
        np.save(path + "L", self.L)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        ctime = time.time()
        print("Loading U and L...",)
        self.U = np.load(path + "U.npy")
        self.L = np.load(path + "L.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def _sampling(self, N,preference_matrix):
        sys.stderr.write("Generating %s random training samples\n" % str(N))
        userList = list(self._users)
        
        userIndex = np.random.choice(list(self._users), size=N)
        
        iItems, jItems, kItems = [], [], []
        
        
        for index in userIndex:
            u = index 
            i = self._trainDict[u][np.random.randint(len(self._Iu[u]))]
            iItems.append(i)
            j = self.sample_negative_item(self._Iu[u])           
            k = self.sample_negative_item(self._Iu[u])           
            while k==j:
                k = self.sample_negative_item(self._Iu[u])           
            if preference_matrix[u][k] > preference_matrix[u][j]:
                k,j = j,k
            jItems.append(j)
            kItems.append(k)
        return userIndex, iItems, jItems, kItems
    

    def sample_negative_item(self, user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def random_item(self):
        return random.randint(0, self._numItems-1)
        
    def moid(self, x):
        if x > 20.0:
            return 1.0
        elif x<-20.0:
            return 0.0
        else:
            return 1. / (1. + np.exp(-x))

    def init(self):      
        self.U = np.random.uniform(0,1,(self._numUsers,self.K))
        self.L = np.random.uniform(0,1,(self._numItems,self.K))

    def train(self, trainData, preference_matrix, epochs=2, batchSize=500):
        self.init()
        if len(trainData) < batchSize:
            batchSize = len(trainData)
        self._trainDict, self._users, self._items = self._dataPretreatment(trainData)
        N = len(trainData) * epochs
        users, iItems, jItems, kItems = self._sampling(N,preference_matrix)
        
        
        itr = 0
        t2 = t0 = time.time()
        while (itr+1)*batchSize < N:
            self._mbgd(
                users[itr*batchSize: (itr+1)*batchSize],
                iItems[itr*batchSize: (itr+1)*batchSize],
                jItems[itr*batchSize: (itr+1)*batchSize],
                kItems[itr*batchSize: (itr+1)*batchSize],
                trainData
            )
            itr += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.3f%% ) in %.1f seconds" %(str(itr*batchSize), 100.0 * float(itr*batchSize)/N, t2 - t0))
            sys.stderr.flush()
        if N > 0:
            sys.stderr.write("\nTotal training time %.2f seconds; %.2f samples per second\n" % (t2 - t0, N*1.0/(t2 - t0)))
            sys.stderr.flush()
            

    def _mbgd(self, users, iItems, jItems, kItems, trainData):
        for _ in range(30):
            obj = 0.0
            for ind in range(len(users)):
                u, i, j, k = users[ind], int (iItems[ind]), int (jItems[ind]), int(kItems[ind])
                self.update_factors(u, i, j, k)

    def update_factors(self, u, i, j, k):
        xi = np.dot(self.U[u, :], self.L[i, :].T)
        xj = np.dot(self.U[u, :], self.L[j, :].T)
        xk = np.dot(self.U[u, :], self.L[k, :].T)
        xij = xi - xj
        xjk = xj - xk
        
        d_U = self.moid(-xjk)*(self.L[i, : ]-self.L[j, : ])+self.moid(-xjk)*(self.L[j, : ]-self.L[k, : ])-self._lamda_p*self.U[u, : ]  
        self.U[u, : ] += self._learningRate*d_U
        
        d_Li = self.moid(-xij)*self.U[u, : ]-self._lamda_q*self.L[i, : ]
        self.L[i, : ] += self._learningRate*d_Li
        
        d_Lj = (self.moid(-xjk)-self.moid(-xij))*self.U[u, :]-self._lamda_q*self.L[j, :]
        self.L[j, : ] += self._learningRate*d_Lj
        
        d_Lk = -self.moid(-xjk)*self.U[u, :]-self._lamda_q*self.L[k, :]
        self.L[k, : ] += self._learningRate*d_Lk
                
    def _dataPretreatment(self, data):
        dataDict = defaultdict(list)
        items = set()
        for rec in data:
            u = rec[0]
            i = rec[1]
            u = int(u)
            self._Iu[u].add(int(i))
            #self._Cu[u] = [int(c) for c in candicate_POIs[u]]
            dataDict[u].append(int(i)) #{用户：地点list}
            items.add(int(i))
        return dataDict, set(dataDict.keys()), items

    def predict(self, uid, lid):
        return np.dot(self.U[uid], self.L[lid])