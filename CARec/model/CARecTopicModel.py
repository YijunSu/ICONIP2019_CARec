# -*- coding: utf-8 -*-

import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import numpy as np
# from NP_chunking import *
Lda = gensim.models.ldamodel.LdaModel

class carecTopicModel(object):
    def __init__(self, uNum, pNum, topicNum, userDocs, poiDocs, poiAuthors):
        self.uNum = uNum
        self.pNum = pNum
        self.userModel = None
        self.poiModel = None
        self.topicNum = topicNum
        self.userDocs = userDocs
        self.userCleanDocs = None
        self.userDocTermMat = None
        self.docDictionary = None
        self.poiDocs = poiDocs
        self.poiAuthors = poiAuthors
        self.poiCleanDocs = None
        self.poiDocTermMat = None
        self.userDistributionMatrix = None
        self.poiDistributionMatrix = None

    def clean(self, doc):
        stop = set(stopwords.words('english'))
        exclude = set(string.punctuation)
        lemma = WordNetLemmatizer()
        stop_free = " ".join([i for i in doc.lower().split() if (i not in stop)])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    def docs2Dict(self, objDocs):
        doc_clean = [self.clean(objDocs[i]) for i in range(len(objDocs))]
        doc_clean = [i.split() for i in doc_clean]
        return doc_clean, corpora.Dictionary(doc_clean)

    def dict2DocTermMatrix(self, clean_docs):
        return [self.docDictionary.doc2bow(doc) for doc in clean_docs]

    def trainUserTopicModel(self, passes=100, minimum_pro=0, alpha_U='symmetric', eta_U=0, iter_U=50):
        self.userCleanDocs, self.docDictionary = self.docs2Dict(self.userDocs)
        print("file ready!")
        self.userDocTermMat = self.dict2DocTermMatrix(self.userCleanDocs)
        print('file processed,applying LDA...')
        self.userModel = Lda(self.userDocTermMat, num_topics=self.topicNum, id2word=self.docDictionary, passes=passes,
                             minimum_probability=minimum_pro,
                             alpha=alpha_U, eta=eta_U, iterations=iter_U)

    def saveUserTopicModel(self, filepath):
        self.userModel.save(filepath)
        np.save(filepath + 'userDistribution', self.userDistributionMatrix)

    def loadUserTopicModel(self, userTopicModelPath, minimum_pro_U=0):
        self.userCleanDocs, self.docDictionary = self.docs2Dict(self.userDocs)
        self.userDocTermMat = self.dict2DocTermMatrix(self.userCleanDocs)
        self.userModel = Lda(id2word=self.docDictionary, num_topics=10, minimum_probability=minimum_pro_U)
        self.userModel.load(userTopicModelPath)

    def userModel2Matrix(self):
        self.userDistributionMatrix = np.zeros((self.uNum, self.topicNum))
        for u in range(self.uNum):
            topicTuples = self.userModel.get_document_topics(self.userDocTermMat[u])
            for tpTuple in topicTuples:
                self.userDistributionMatrix[u][tpTuple[0]] = tpTuple[1]

    def poiModel2Matrix(self):
        self.poiDistributionMatrix = np.zeros((self.pNum, self.topicNum))
        for u in range(self.pNum):
            topicTuples = self.poiModel.get_document_topics(self.poiDocTermMat[u])
            for tpTuple in topicTuples:
                self.poiDistributionMatrix[u][tpTuple[0]] = tpTuple[1]

    def trainPoiTopicModel(self, poiDocsData, passes=100, minimum_pro=0, alpha_P='symmetric', eta_P=0, iter_P=50):
        self.poiCleanDocs, _ = self.docs2Dict(poiDocsData)
        self.poiDocTermMat = self.dict2DocTermMatrix(self.poiCleanDocs)
        print('file processed,applying LDA...')
        self.poiModel = Lda(self.poiDocTermMat, num_topics=self.topicNum, id2word=self.docDictionary,
                            passes=100, minimum_probability=minimum_pro, alpha=alpha_P, eta=eta_P, iterations=iter_P)

    def savePoiTopicModel(self, filepath):
        self.poiModel.save(filepath)
        np.save(filepath + 'poiDistribution', self.poiDistributionMatrix)

    def loadPoiTopicModel(self, poiTopicModelPath, minimum_pro_L=0):
        self.poiCleanDocs, _ = self.docs2Dict(self.poiDocs)
        self.poiDocTermMat = self.dict2DocTermMatrix(self.poiCleanDocs)
        self.poiModel = Lda(id2word=self.docDictionary, num_topics=10, minimum_probability=minimum_pro_L)
        self.poiModel.load(poiTopicModelPath)

    def loadDistribution(self, filepath_U, filepath_P):
        self.userDistributionMatrix = np.load(filepath_U + 'userDistribution.npy')
        self.poiDistributionMatrix = np.load(filepath_P + 'poiDistribution.npy')

    def trainCarecTopicModel(self, passes_U=100, minimum_pro_U=0, passes_P=100, minimum_pro_P=0, alpha_U='symmetric',
                             alpha_P='symmetric', eta_U=0, eta_P=0, iter_U=500, iter_P=500, reg_POI_topic=True):
        print("training user model...")
        self.trainUserTopicModel(passes=passes_U, minimum_pro=minimum_pro_U, alpha_U=alpha_U, eta_U=eta_U,
                                 iter_U=iter_U)
        self.trainPoiTopicModel(self.poiDocs, passes=passes_P, minimum_pro=minimum_pro_P, alpha_P=alpha_P, eta_P=eta_P,
                                iter_P=iter_P)
        print('training topic model...')
        if reg_POI_topic:
            self.fusingStep()
        self.userModel2Matrix()
        self.poiModel2Matrix()

    def saveCarecTopicModel(self, userTopicModelPath, poiTopicModelPath):
        self.saveUserTopicModel(userTopicModelPath)
        self.savePoiTopicModel(poiTopicModelPath)

    def loadCarecTopicModel(self, userTopicModelPath, poiTopicModelPath, minimum_pro_U=0, minimum_pro_L=0,
                            reg_POI_topic=True):
        print('loading user topic model...')
        self.loadUserTopicModel(userTopicModelPath, minimum_pro_U=minimum_pro_U)
        print('loading location topic model...')
        self.loadPoiTopicModel(poiTopicModelPath, minimum_pro_L=minimum_pro_L)
        self.userModel2Matrix()
        if reg_POI_topic:
            self.fusingStep()
        self.poiModel2Matrix()

    def fusingStep(self):
        self.poiDistributionMatrix = np.zeros((self.pNum, self.topicNum))
        count = 0
        for p in range(self.pNum):  # 选一个位置
            count += 1
            userAffect = {}  # 将当前位置用户对每个主题的影响计算为一个字典，然后一起加入原分布
            for author in self.poiAuthors[p]:  # 该位置的访客
                for tp in range(self.topicNum):  # 访客的一个话题概率
                    topicTuple = self.userModel.get_document_topics(self.userDocTermMat[author])[tp]  # 一个 话题编号-概率 二元组
                    if int(topicTuple[0]) not in userAffect:
                        userAffect.update({int(topicTuple[0]): 0})
                    userAffect[int(topicTuple[0])] += topicTuple[1]
            for topic in range(len(self.poiModel.get_document_topics(self.poiDocTermMat[p]))):  # 分布的二元组
                tpTuple = self.poiModel.get_document_topics(self.poiDocTermMat[p])[topic]  # 一个二元组
                topicNum = int(tpTuple[0])
                topicP = tpTuple[1]
                self.poiDistributionMatrix[p][topicNum] = topicP * userAffect[topicNum]
            regFactor = sum(self.poiDistributionMatrix[p])
            self.poiDistributionMatrix[p] = self.poiDistributionMatrix[p] / regFactor

            if count % 1000 == 0:
                print('processing POI ', count, '/', self.pNum, '...')
