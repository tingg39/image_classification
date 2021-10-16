import math

import numpy
import samples
import knn
import dataClassifier

nums = [(1, 23, 3), (4, 15, 6), (7, 81, 9), (10, 11, 12), (13, 141, 15)]
output = [number[1] for number in nums]
#print output

nums.sort(key=lambda tup: tup[1])
#print nums

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70

numTraining = 100
numTest = 2

rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)

def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = numpy.empty((28,28))
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

featureFunction = basicFeatureExtractorDigit

# Extract features
print "Extracting features..."
trainingData = map(featureFunction, rawTrainingData)
testData = map(featureFunction, rawTestData)


def euclidean_distance(img_test, img_new):
    return numpy.sum((img_test - img_new) ** 2)


distance = euclidean_distance(trainingData, trainingData)
print distance
exit()

classifier = knn.KnnClassifier("knn")

print "KNN chosen - does not require training nor validation"
print "Testing..."
#guesses = classifier.classify(trainingData, trainingLabels, testData)
#correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
#print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
exit()
