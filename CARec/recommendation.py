# -*- coding: utf-8 -*-

import sys
import argparse
from collections import defaultdict
import numpy as np
from model.CARecModel import carecModel
from metric.metrics import precisionk, recallk, ndcgk


def loadReviews(filepath):
    reviewsFile = open(filepath, 'r', encoding='utf-8')
    reviewsContent = reviewsFile.readlines()
    reviewsFile.close()
    reviewsList = []
    for review in reviewsContent:
        review = review.split()
        reviewsList.append([int(review[0]), int(review[1]), int(review[2])])
    return np.array(reviewsList)


def loadCheckin(filepath):
    checkinFile = open(filepath, 'r', encoding='utf-8')
    chekcins = checkinFile.readlines()
    checkinFile.close()
    checkinList = []
    for checkin in chekcins:
        checkin = checkin.split()
        checkinList.append([int(checkin[0]), int(checkin[1]), int(checkin[2])])
    return np.array(checkinList)


def loadDocs(filepath):
    docFile = open(filepath, 'r', encoding='utf-8')
    docRecords = docFile.readlines()
    obj_doc = {}
    for rec in docRecords:
        rec = rec.split(' ', 1)
        obj_doc.update({int(rec[0]): rec[1]})
    return obj_doc


def loadPoiAuthor(filepath):
    poiAuthorFile = open(filepath, 'r', encoding='utf-8')
    poiAuthors = poiAuthorFile.readlines()
    poiAuthorFile.close()
    poiAuthorsDict = {}
    for rec in poiAuthors:
        rec = rec.split(' ')
        aList = []
        for a in rec[1:-1]:
            aList.append(int(a))
        poiAuthorsDict.update({int(rec[0]): aList})
    return poiAuthorsDict


def main(args):
    print("dataset:", args.dataset)
    if args.dataset != "Foursquare" and args.dataset != "Yelp":
        print("wrong dataset: Foursquare/Yelp")
        exit
    dataset = args.dataset
    datasetPath = '../datasets/' + dataset
    topK = args.topK

    checkinTrain = loadCheckin(datasetPath + '/' + dataset + '_checkins_train.txt')
    checkinTest = loadCheckin(datasetPath + '/' + dataset + '_checkins_test.txt')

    poiAuthors = loadPoiAuthor(datasetPath + '/' + dataset + '_poi_author.txt')
    userDocs = loadDocs(datasetPath + '/' + dataset + '_user_doc.txt')
    poiDocs = loadDocs(datasetPath + '/' + dataset + '_poi_doc.txt')

    reviewTrain = loadReviews(datasetPath + '/' + dataset + '_reviewsScores.txt')

    uNum = int(np.amax(checkinTrain[:, 0])) + 1
    pNum = int(np.amax(reviewTrain[:, 1])) + 1
    print(uNum, ' user in trainset')
    print(pNum, ' POIs in trainset')

    myCARecModel = carecModel(uNum=uNum, pNum=pNum)

    if args.trainCheckin:
        myCARecModel.trainCheckinModel(checkinTrain, checkin_K=40, alpha_U=0.2, alpha_L=0.1, max_iters=10, lr=1e-3)
        myCARecModel.saveCheckinModel(savePath='tmp/' + dataset + '/checkin_model/checkin_model_')
    else:
        myCARecModel.loadCheckinModel('tmp/' + dataset + '/checkin_model/checkin_model_')
    print('\n checkin model ready!\n')

    if args.trainSenti:
        myCARecModel.trainSentiModel(reviewTrain, senti_K=40, alpha_U=0.2, alpha_L=0.2, max_iters=10, lr=1e-3)
        myCARecModel.saveSentiModel(savePath='tmp/' + dataset + '/senti_model/senti_model_')
    else:
        myCARecModel.loadSentiModel('tmp/' + dataset + '/senti_model/senti_model_')
    print('\n sentiment model ready!')

    if args.trainTopic:
        myCARecModel.trainTopicModel(userDocs, poiDocs, poiAuthors, topicNum=15)
        myCARecModel.saveTopicModel(topicModelPath_U='tmp/' + dataset + '/topic_model/user_topic_model_',
                                    topicModelPath_P='tmp/' + dataset + '/topic_model/poi_topic_model_')
    else:
        myCARecModel.loadTopicModel(userDocs, poiDocs, poiAuthors, topicNum=50,
                                    topicModelPath_U='tmp/' + dataset + '/topic_model/user_topic_model_',
                                    topicModelPath_P='tmp/' + dataset + '/topic_model/poi_topic_model_')
    print('\n topic model ready!')

    if args.trainBPR:
        myCARecModel.fusingFrame()
        myCARecModel.trainPRMFC(checkinTrain, K=40, lamda_p=0.2, lamda_q=0.2, learningRate=0.005, epochs=10,
                                batchSize=500, )
        myCARecModel.savePRMFC('tmp/' + dataset + '/PRMFC/')
    else:
        myCARecModel.loadPRMFC('tmp/' + dataset + '/PRMFC/')

    ground_truth = defaultdict(set)
    for rec in checkinTest:
        ground_truth[rec[0]].add(rec[1])

    trainTuple = {}
    for i in checkinTrain:
        trainTuple.update({(str(i[0]) + '#' + str(i[1])): i[2]})

    precision5, recall5, ndcg5 = [], [], []
    precision10, recall10, ndcg10 = [], [], []
    precision20, recall20, ndcg20 = [], [], []
    precision50, recall50, ndcg50 = [], [], []

    for uid in range(uNum):
        if uid in ground_truth:
            print('\n processing user: ', uid)
            predictions = []
            count = 0
            countN = 0
            countAll = 0
            for pid in range(pNum):
                countAll += 1
                if (str(uid) + '#' + str(pid)) not in trainTuple.keys():
                    countN += 1
                    preScore = myCARecModel.carec_U[uid].dot(myCARecModel.carec_L[pid])
                else:
                    preScore = sys.float_info.min
                    count += 1
                predictions.append(preScore)
            predicted = list(reversed(np.array(predictions).argsort()))[:topK]
            actual = ground_truth[uid]

            precision5.append(precisionk(actual, predicted[:5]))
            recall5.append(recallk(actual, predicted[:5]))
            ndcg5.append(ndcgk(actual, predicted[:5], 5))

            precision10.append(precisionk(actual, predicted[:10]))
            recall10.append(recallk(actual, predicted[:10]))
            ndcg10.append(ndcgk(actual, predicted[:10], 10))

            precision20.append(precisionk(actual, predicted[:20]))
            recall20.append(recallk(actual, predicted[:20]))
            ndcg20.append(ndcgk(actual, predicted[:20], 20))

            precision50.append(precisionk(actual, predicted[:50]))
            recall50.append(recallk(actual, predicted[:50]))
            ndcg50.append(ndcgk(actual, predicted[:50], 50))

            print(uid, "pre@5:", np.mean(precision5), "rec@5:", np.mean(recall5), "ndcg@5:",
                  np.mean(ndcg5))
            print(uid, "pre@10:", np.mean(precision10), "rec@10:", np.mean(recall10),
                  "ndcg@10:", np.mean(ndcg10))
            print(uid, "pre@20:", np.mean(precision20), "rec@20:", np.mean(recall20),
                  "ndcg@20:", np.mean(ndcg20))
            print(uid, "pre@50:", np.mean(precision50), "rec@50:", np.mean(recall50),
                  "ndcg@50:", np.mean(ndcg50))

    resFile = open('result/'+ dataset + '_MainRes.txt', 'w', encoding='utf-8')

    resFile.write("%.5f" % (np.mean(precision5)))
    resFile.write(' ')
    resFile.write("%.5f" % (np.mean(recall5)))
    resFile.write(' ')
    resFile.write("%.5f" % (np.mean(ndcg5)))
    resFile.write('\n')

    resFile.write("%.5f" % (np.mean(precision10)))
    resFile.write(' ')
    resFile.write("%.5f" % (np.mean(recall10)))
    resFile.write(' ')
    resFile.write("%.5f" % (np.mean(ndcg10)))
    resFile.write('\n')

    resFile.write("%.5f" % (np.mean(precision20)))
    resFile.write(' ')
    resFile.write("%.5f" % (np.mean(recall20)))
    resFile.write(' ')
    resFile.write("%.5f" % (np.mean(ndcg20)))
    resFile.write('\n')

    resFile.write("%.5f" % (np.mean(precision50)))
    resFile.write(' ')
    resFile.write("%.5f" % (np.mean(recall50)))
    resFile.write(' ')
    resFile.write("%.5f" % (np.mean(ndcg50)))
    resFile.write('\n')
    resFile.close()


parser = argparse.ArgumentParser()
parser.add_argument("-dataset", "--dataset", default="Foursquare",
                    help="select dataset：Foursquare and Yelp")
parser.add_argument("-topK", "--topK", default=100, help="topK of Pois, 100 for default")
parser.add_argument("-trainCheckin", "--trainCheckin", default=False, help="train Check-in model")
parser.add_argument("-trainSenti", "--trainSenti", default=False, help="train Sentiment model")
parser.add_argument("-trainTopic", "--trainTopic", default=True, help="train Topic model")
parser.add_argument("-trainBPR", "--trainBPR", default=False, help="train BPR")
parser.add_argument("-topicReg", "--topicReg", action="store_true", default=True,
                    help="regulized POI topic model with topic model of users")
args = parser.parse_args()

main(args)
