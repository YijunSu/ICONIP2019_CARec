# -*- coding: utf-8 -*-

import numpy as np
from model.CARecSentiModel import carecSentiModel
from model.CARecCheckinModel import carecCheckinModel
from model.CARecTopicModel import carecTopicModel
from model.PRMFC import PRMFCModel
from sklearn.preprocessing import normalize


class carecModel(object):
    def __init__(self, uNum, pNum):
        self.uNum = uNum
        self.pNum = pNum
        self.sentiModel = None
        self.checkinModel = None
        self.topicModel = None
        self.checkin_U = None
        self.checkin_L = None
        self.senti_U = None
        self.senti_L = None
        self.userDistribution = None
        self.poiDistribution = None
        self.carecPRMFCModel = None
        self.carec_U = None
        self.carec_L = None

        self.checkinPreMat = np.zeros((self.uNum, self.pNum))
        self.sentiPreMat = np.zeros((self.uNum, self.pNum))
        self.topicPreMat = np.zeros((self.uNum, self.pNum))

        self.userPreferenceMatrix = np.zeros((self.uNum, self.pNum))

    def Normalize(data):
        m = np.mean(data)
        mx = max(data)
        mn = min(data)
        return [(float(i) - m) / (mx - mn) for i in data]

    def fusingFrame(self):
        self.checkin_U = self.checkinModel.checkinPMFModel.U
        self.checkin_L = self.checkinModel.checkinPMFModel.L
        self.senti_U = self.sentiModel.sentiPMFModel.U
        self.senti_L = self.sentiModel.sentiPMFModel.L
        self.userTopic = self.topicModel.userDistributionMatrix
        self.poiTopic = self.topicModel.poiDistributionMatrix
        count = 0
        for u in range(self.uNum):
            count += 1
            for p in range(self.pNum):
                self.checkinPreMat[u][p] = self.checkin_U[u].dot(self.checkin_L[p])
                self.sentiPreMat[u][p] = self.senti_U[u].dot(self.senti_L[p])
                self.topicPreMat[u][p] = self.userDistribution[u].dot(self.poiDistribution[p])
            if count % 1000 == 0:
                print("proceesing sub preference", count, '/', self.uNum, '...')
        self.checkinPreMat = normalize(self.checkinPreMat)
        self.sentiPreMat = normalize(self.sentiPreMat)
        self.topicPreMat = normalize(self.topicPreMat)
        for u in range(self.uNum):
            for p in range(self.pNum):
                self.userPreferenceMatrix[u][p] = self.checkinPreMat[u][p] * self.sentiPreMat[u][p] * \
                                                  self.topicPreMat[u][p]
        del self.checkinPreMat
        del self.sentiPreMat
        del self.topicPreMat

    def trainCheckinModel(self, train, checkin_K=15, alpha_U=0.5, alpha_L=0.5, max_iters=500, lr=1e-4):
        self.checkinModel = carecCheckinModel(self.uNum, self.pNum)
        self.checkinModel.trainCheckinModel(train, K=checkin_K, alpha_U=alpha_U, alpha_L=alpha_L, max_iters=max_iters,
                                            learning_rate=lr)
        self.checkin_U = self.checkinModel.U
        self.checkin_L = self.checkinModel.L

    def saveCheckinModel(self, savePath='tmp/checkin_model/checkin_model_'):
        self.checkinModel.saveCheckinModel(savePath)

    def loadCheckinModel(self, filepath='tmp/checkin_model/checkin_model_'):
        self.checkinModel = carecCheckinModel(self.uNum, self.pNum)
        self.checkinModel.loadCheckinModel(filepath)
        self.checkin_U = self.checkinModel.U
        self.checkin_L = self.checkinModel.L

    def trainSentiModel(self, train, senti_K=15, alpha_U=0.2, alpha_L=0.2, max_iters=500, lr=1e-4):
        self.sentiModel = carecSentiModel(self.uNum, self.pNum)
        self.sentiModel.trainSentiModel(train, K=senti_K, alpha_U=alpha_U, alpha_L=alpha_L, max_iters=max_iters,
                                        learning_rate=lr)
        self.senti_U = self.sentiModel.U
        self.senti_L = self.sentiModel.L

    def saveSentiModel(self, savePath='tmp/senti_model/senti_model_'):
        self.sentiModel.saveSentiModel(savePath)

    def loadSentiModel(self, path):
        self.sentiModel = carecSentiModel(self.uNum, self.pNum)
        self.sentiModel.loadSentiModel(path)
        self.senti_U = self.sentiModel.sentiPMFModel.U
        self.senti_L = self.sentiModel.sentiPMFModel.L

    def trainTopicModel(self, userDocs, poiDocs, poiAuthors, topicNum=15):
        self.topicModel = carecTopicModel(self.uNum, self.pNum, topicNum=topicNum, userDocs=userDocs, poiDocs=poiDocs,
                                          poiAuthors=poiAuthors)
        self.topicModel.trainCarecTopicModel()

        self.userDistribution = self.topicModel.userDistributionMatrix
        self.poiDistribution = self.topicModel.poiDistributionMatrix

    def saveTopicModel(self, topicModelPath_U='./tmp/topic_model/user_topic_model_',
                       topicModelPath_P='./tmp/topic_model/poi_topic_model_'):
        self.topicModel.saveCarecTopicModel(topicModelPath_U, topicModelPath_P)

    def loadTopicModel(self, userDocs, poiDocs, poiAuthors, topicNum=10,
                       topicModelPath_U='./tmp/topic_model/user_topic_model', topicModelPath_P=
                       './tmp/topic_model/poi_topic_model'):
        self.topicModel = carecTopicModel(self.uNum, self.pNum, topicNum=topicNum, userDocs=userDocs, poiDocs=poiDocs,
                                          poiAuthors=poiAuthors)
        self.topicModel.loadCarecTopicModel(topicModelPath_U, topicModelPath_P)
        self.topicModel.loadDistribution(topicModelPath_U, topicModelPath_P)
        self.userDistribution = self.topicModel.userDistributionMatrix
        self.poiDistribution = self.topicModel.poiDistributionMatrix

    def trainPRMFC(self, trainData, K=50, lamda_p=0.01, lamda_q=0.01, learningRate=0.005, epochs=10, batchSize=1000):
        self.carecPRMFCModel = PRMFCModel(self.uNum, self.pNum, K=K, lamda_p=lamda_p, lamda_q=lamda_q,
                                          learningRate=learningRate)
        self.carecPRMFCModel.train(trainData, self.userPreferenceMatrix, epochs=epochs, batchSize=batchSize)
        self.carec_U = self.carecPRMFCModel.U
        self.carec_L = self.carecPRMFCModel.L

    def savePRMFC(self, filepath):
        self.carecPRMFCModel.save_model(filepath)
        print('model saved!')

    def loadPRMFC(self, filepath):
        self.carecPRMFCModel = PRMFCModel(self.uNum, self.pNum, K=50, lamda_p=0.01, lamda_q=0.01, learningRate=0.005)
        self.carecPRMFCModel.load_model(filepath)
        self.carec_U = self.carecPRMFCModel.U
        self.carec_L = self.carecPRMFCModel.L

    def predict(self, uid, pid):
        return np.dot(self.carec_U[uid], self.carec_L[pid])
