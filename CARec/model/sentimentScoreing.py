# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:13:40 2019

@author: L_lab
"""
from sentiment import SentimentAnalysis
foursquarePath = './dataset/Foursquare/'
s = SentimentAnalysis(filename='./tmp/sentiWordNet/SentiWordNet_3.0.0.txt',weighting='geometric')

def loadReviews(filepath):
    reviewsFile = open(filepath,'r',encoding='utf-8')
    reviewsContent = reviewsFile.readlines()
    reviewsFile.close()
    scoreList = []
    count = 0
    zeroCount = 0
    for review in  reviewsContent:
        count+=1
        review = review.split(' ',2)
        ss = s.score(review[2])
        scoreList.append(ss)
        if ss==0:
            zeroCount+=1
        if count%1000 == 0:
            print(count)
    return scoreList,zeroCount


reviews,zeros = loadReviews(foursquarePath+'Foursquare_reviews.txt')
