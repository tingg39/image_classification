# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import numpy as np
class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]

    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """ 
    if (len(self.legalLabels) > 2 ):
        w,h = 28,28
    else:
        w,h = 70,60
    matrix0 = [[0 for x in range(w)] for y in range(h)]
    matrix1 = [[0 for x in range(w)] for y in range(h)]
    matrix2 = [[0 for x in range(w)] for y in range(h)]
    matrix3 = [[0 for x in range(w)] for y in range(h)]
    matrix4 = [[0 for x in range(w)] for y in range(h)]
    matrix5 = [[0 for x in range(w)] for y in range(h)]
    matrix6 = [[0 for x in range(w)] for y in range(h)]
    matrix7 = [[0 for x in range(w)] for y in range(h)]
    matrix8 = [[0 for x in range(w)] for y in range(h)]
    matrix9 = [[0 for x in range(w)] for y in range(h)]
    counter0,counter1,counter2,counter3,counter4,counter5,counter6,counter7,counter8,counter9, = 0,0,0,0,0,0,0,0,0,0
    
    for i in range(0,len(trainingData)): #i = each image
        for j in range(0,len(trainingData[i].values())): #j = iteration of value in image i
            if (trainingData[i].values()[j] == 1): #if the feature value = 1
                Fij = str(trainingData[i].keys()[j]) #features = 
                coordinates = Fij.split(",")
                x = int(coordinates[0][1:])
                y = int(coordinates[1][:-1])
                if (len(self.legalLabels) > 2 ): # digits
                    if (trainingLabels[i] == 0):
                        matrix0[x][y] = matrix0[x][y] + 1   
                    elif (trainingLabels[i] == 1):
                        matrix1[x][y] = matrix1[x][y] + 1      
                    elif (trainingLabels[i] == 2):
                        matrix2[x][y] = matrix2[x][y] + 1   
                    elif (trainingLabels[i] == 3):
                        matrix3[x][y] = matrix3[x][y] + 1   
                    elif (trainingLabels[i] == 4):
                        matrix4[x][y] = matrix4[x][y] + 1   
                    elif (trainingLabels[i] == 5):
                        matrix5[x][y] = matrix5[x][y] + 1   
                    elif (trainingLabels[i] == 6):
                        matrix6[x][y] = matrix6[x][y] + 1   
                    elif (trainingLabels[i] == 7):
                        matrix7[x][y] = matrix7[x][y] + 1   
                    elif (trainingLabels[i] == 8):
                        matrix8[x][y] = matrix8[x][y] + 1   
                    elif (trainingLabels[i] == 9):       
                        matrix9[x][y] = matrix9[x][y] + 1                
                else: # faces
                    if (trainingLabels[i] == 0):
                        matrix0[x][y] = matrix0[x][y] + 1
                    elif (trainingLabels[i] == 1):
                        matrix1[x][y] = matrix1[x][y] + 1 
        if (trainingLabels[i] == 0):
            counter0 = counter0 + 1
        if (trainingLabels[i] == 1):
            counter1 = counter1 + 1
        if (trainingLabels[i] == 2):
            counter2 = counter2 + 1
        if (trainingLabels[i] == 3):
            counter3 = counter3 + 1
        if (trainingLabels[i] == 4):
            counter4 = counter4 + 1
        if (trainingLabels[i] == 5):
            counter5 = counter5 + 1
        if (trainingLabels[i] == 6):
            counter6 = counter6 + 1
        if (trainingLabels[i] == 7):
            counter7 = counter7 + 1
        if (trainingLabels[i] == 8):
            counter8 = counter8 + 1
        if (trainingLabels[i] == 9):
            counter9 = counter9 + 1    
                        

               
    kgrid = int(kgrid[0])
    # probabilities for each class given feature = 1
    matrix = []
    matrix.append((np.array(matrix0)+kgrid)/(2*kgrid + float(counter0)))
    matrix.append((np.array(matrix1)+kgrid)/(2*kgrid + float(counter1)))
    if (len(self.legalLabels) > 2 ):
        matrix.append((np.array(matrix2)+kgrid)/(2*kgrid + float(counter2)))
        matrix.append((np.array(matrix3)+kgrid)/(2*kgrid + float(counter3)))
        matrix.append((np.array(matrix4)+kgrid)/(2*kgrid + float(counter4)))
        matrix.append((np.array(matrix5)+kgrid)/(2*kgrid + float(counter5)))
        matrix.append((np.array(matrix6)+kgrid)/(2*kgrid + float(counter6)))
        matrix.append((np.array(matrix7)+kgrid)/(2*kgrid + float(counter7)))
        matrix.append((np.array(matrix8)+kgrid)/(2*kgrid + float(counter8)))
        matrix.append((np.array(matrix9)+kgrid)/(2*kgrid + float(counter9)))
    #priors for each class
    priors = []
    priors.append((float(counter0)/len(trainingData)))
    priors.append((float(counter1)/len(trainingData)))
    if (len(self.legalLabels) > 2 ):
        priors.append((float(counter2)/len(trainingData)))
        priors.append((float(counter3)/len(trainingData)))
        priors.append((float(counter4)/len(trainingData)))
        priors.append((float(counter5)/len(trainingData)))
        priors.append((float(counter6)/len(trainingData)))
        priors.append((float(counter7)/len(trainingData)))
        priors.append((float(counter8)/len(trainingData)))
        priors.append((float(counter9)/len(trainingData)))   

    self.probabilityMatrix = matrix
    self.priorsList = priors

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
        posterior = self.calculateLogJointProbabilities(datum)
        guesses.append(posterior.index(max(posterior)))
        self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    position = 0
    counter = 0
    count = 0

    if (len(self.legalLabels) > 2 ):
        logJoint = [0,0,0,0,0,0,0,0,0,0]
    else:
        logJoint = [0,0]

    for i in datum.values(): 

        classPosition = 0
        posterior = 0
        Fij = str(datum.keys()[position]) 
        coordinates = Fij.split(",")
        x = int(coordinates[0][1:])
        y = int(coordinates[1][:-1])
        if (i == 1):   
            for c in self.probabilityMatrix: #calculating posterior for each class
                logJoint[classPosition] = logJoint[classPosition] + math.log(c[x][y])
                classPosition = classPosition + 1
        else:
            for c in self.probabilityMatrix: #calculating posterior for each class
                
                logJoint[classPosition] = logJoint[classPosition] + math.log(1 - c[x][y])
                classPosition = classPosition + 1
        for j in range(0,len(self.priorsList)):
            logJoint[j] = logJoint[j] + self.priorsList[j]
        position = position + 1
    return logJoint

