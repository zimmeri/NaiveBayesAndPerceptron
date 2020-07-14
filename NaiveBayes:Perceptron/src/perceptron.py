#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""
from utils import get_feature_vectors
from classifier import BinaryClassifier


class Perceptron(BinaryClassifier):
    
    def __init__(self, args): 
        self.num_iter = args.num_iter 
        self.lr = args.lr 
        self.bin_feats = args.bin_feats 
        self.weight = [0.0 for x in range(args.vocab_size)] 
        self.bias = 0.0 

        
    def fit(self, train_data):
        vec = get_feature_vectors(train_data[0], self.bin_feats)
        for count in range(self.num_iter):
            row = vec[count]
            realLabel = train_data[1][count]
            guessLabel = self.guess(row) 
            if (realLabel != guessLabel):
                self.calcBias(realLabel)
                self.calcWeight(row, realLabel)


    def predict(self, test_x):
        p = []
        prediction = 0
        featureVecs = get_feature_vectors(test_x, self.bin_feats)
        for row in featureVecs:
            prediction = self.guess(row)
            p.append(prediction)
        return p

    def calcWeight(self, row, label):
        length = len(self.weight)
        for ft in range(length):
            self.weight[ft] = (label * self.lr * row[ft]) + self.weight[ft]
 
        
        
    def calcBias(self, label):
        self.bias = (label * self.lr) + self.bias

    def guess(self, row):
        guessLabel= self.bias
        length = len(row)
        for ft in range(length):
            guessLabel = (row[ft] * self.weight[ft]) + guessLabel
        if guessLabel >= 0:
            guessLabel = 1
        else:
            guessLabel = -1
        return guessLabel
        

class AveragedPerceptron(BinaryClassifier):
    
    def __init__(self, args): 
        self.num_iter = args.num_iter 
        self.lr = args.lr 
        self.bin_feats = args.bin_feats 
        self.weight = [0.0 for x in range(args.vocab_size)] 
        self.bias = 0.0 
        self.survival = 0.0

        
    def fit(self, train_data):
        vec = get_feature_vectors(train_data[0], self.bin_feats)
        for count in range(self.num_iter):
            row = vec[count]
            realLabel = train_data[1][count]
            guessLabel = self.guess(row) 
            if (realLabel != guessLabel):
                self.calcBias(realLabel)
                self.calcWeight(row, realLabel)
            else:
                self.survival = self.survival + 1


    def predict(self, test_x):
        p = []
        prediction = 0
        featureVecs = get_feature_vectors(test_x, self.bin_feats)
        for row in featureVecs:
            prediction = self.guess(row)
            p.append(prediction)
        return p

    def calcWeight(self, row, label):
        length = len(self.weight)
        denom = self.survival + 1
        for ft in range(length):
            w = (label * self.lr * row[ft]) + self.weight[ft]
            self.weight[ft] = ((self.weight[ft] * self.survival)+w)/denom
        self.survival = 1
 
        
        
    def calcBias(self, label):
        denom = self.survival + 1
        self.bias = ((label * self.lr) + self.bias)/denom

    def guess(self, row):
        guessLabel= self.bias
        length = len(row)
        for ft in range(length):
            guessLabel = (row[ft] * self.weight[ft]) + guessLabel
        if guessLabel >= 0:
            guessLabel = 1
        else:
            guessLabel = -1
        return guessLabel