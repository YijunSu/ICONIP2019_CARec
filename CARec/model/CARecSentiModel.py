# -*- coding: utf-8 -*-

import numpy as np
from model.sentiment import SentimentAnalysis
from model.PMF import PMF
from scipy.sparse import csc_matrix

class carecSentiModel(object):
    def __init__(self,uNum,pNum):
        self.uNum = uNum
        self.pNum = pNum
        self.sentiPMFModel = None
        self.K = None
        self.alpha_U = 0.02
        self.alpha_L = 0.02
        self.U = None
        self.L = None
       
    def sentiScoring(self,reviews):
        reviewScoresList = []
        s = SentimentAnalysis(filename='./tmp/sentiWordNet/SentiWordNet_3.0.0.txt',weighting='geometric')  #sentimentAnalysiser settings
        reviewLen = len(reviews)
        count = 0
        for review in reviews:
            count+=1
            rawScore=s.score(review[2])
            if rawScore>=-1 and rawScore<-0.05:
                score = 1
            if rawScore>=-0.05 and rawScore<-0.01:
                score = 2
            if rawScore>=-0.01 and rawScore<0.01:
                score = 3
            if rawScore>=0.01 and rawScore<0.05:
                score = 4
            if rawScore>=0.05 and rawScore<1:
                score = 5
            reviewScoresList.append([review[0],review[1],score])
            if count %10000 == 0:
                print('sentiment-scoring ',count,'/',reviewLen,'...')
        return reviewScoresList
    
    def scoreList2Matrix(self,scoreList):
        sentiment_preference_mat = np.zeros((self.uNum,self.pNum),dtype=np.int)
        for rec in scoreList:
            user = int(rec[0])
            poi = int(rec[1])
            ss = int(rec[2])
            sentiment_preference_mat[user][poi] = ss
        return csc_matrix(sentiment_preference_mat)
    
    def trainSentiModel(self,train,K=15,alpha_U=0.2,alpha_L=0.2,max_iters=500,learning_rate=1e-4):
        #reviewScores = self.sentiScoring(reviews)
        self.K = K
        self.alpha_L = alpha_L
        self.alpha_U = alpha_U
        
        #train,test = train_test_split(reviews,test_size=0.2)#sentiment scores
        sentiment_preference_matrix = self.scoreList2Matrix(train)
        print('training PMF for sentiment...')
        #PMF train step
        self.sentiPMFModel = PMF(K=self.K, alpha_U=self.alpha_U, alpha_L=self.alpha_L)
        self.sentiPMFModel.train(sentiment_preference_matrix,max_iters=max_iters,learning_rate=learning_rate)
        self.U = self.sentiPMFModel.U
        self.L = self.sentiPMFModel.L

    def saveSentiModel(self,path):
        self.sentiPMFModel.save_model(path)  
        
    def loadSentiModel(self,path):
        self.sentiPMFModel = PMF()
        self.sentiPMFModel.load_model(path)
        self.uNum = len(self.sentiPMFModel.U)
        self.pNum = len(self.sentiPMFModel.L)
        self.K = len(self.sentiPMFModel.U[0])
    
    def testSentiModel(self,user,poi,sigmoid=False):
        return self.sentiPMFModel.predict(user,poi,sigmoid=sigmoid)
