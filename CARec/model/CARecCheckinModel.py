# -*- coding: utf-8 -*-

import numpy as np
from model.PMF import PMF
from scipy.sparse import csc_matrix


class carecCheckinModel(object):
    def __init__(self, uNum, pNum):
        self.uNum = uNum
        self.pNum = pNum
        self.checkinPMFModel = None
        self.K = None
        self.alpha_U = 0.02
        self.alpha_L = 0.02
        self.U = None
        self.L = None

    def checkinScoring(self, checkinData):
        checkinScoreList = []
        checkinLen = len(checkinData)
        count = 0
        for checkin in checkinData:
            count += 1
            freq = int(checkin[2])
            cScore = freq
            checkinScoreList.append([int(checkin[0]), int(checkin[1]), cScore])
        if count % 1000 == 0:
            print('checkin-scoring ', count, '/', checkinLen, '...')
        return np.array(checkinScoreList)

    def scoreList2Matrix(self, scoreList):
        checkins_preference_mat = np.zeros((self.uNum, self.pNum), dtype=np.int)
        for rec in scoreList:
            checkins_preference_mat[int(rec[0])][int(rec[1])] = int(rec[2])
        return csc_matrix(checkins_preference_mat)

    def trainCheckinModel(self, train, K=15, alpha_U=0.5, alpha_L=0.5, max_iters=500, learning_rate=1e-4):
        self.K = K
        self.alpha_U = alpha_U
        self.alpha_L = alpha_L

        checkin_preference_matrix = self.scoreList2Matrix(train)
        print('training PMF for checkins...')
        self.checkinPMFModel = PMF(K=self.K, alpha_U=self.alpha_U, alpha_L=self.alpha_L)
        self.checkinPMFModel.train(checkin_preference_matrix, max_iters=max_iters, learning_rate=learning_rate)
        self.U = self.checkinPMFModel.U
        self.L = self.checkinPMFModel.L

    def saveCheckinModel(self, path):
        self.checkinPMFModel.save_model(path)

    def loadCheckinModel(self, path):
        self.checkinPMFModel = PMF()
        self.checkinPMFModel.load_model(path)
        self.uNum = len(self.checkinPMFModel.U)
        self.pNum = len(self.checkinPMFModel.L)
        self.K = len(self.checkinPMFModel.U[0])
        self.U = self.checkinPMFModel.U
        self.L = self.checkinPMFModel.L

    def testCheckinModel(self, user, poi, sigmoid=False):
        return self.checkinPMFModel.predict(user, poi, sigmoid=sigmoid)
