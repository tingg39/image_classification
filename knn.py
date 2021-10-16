import numpy


class KnnClassifier:

    def __init__(self, type_text):
        self.type = type_text

    # Calculate the Euclidean distance between two images
    def euclidean_distance(self, img_test, img_new):
        return numpy.sum((img_test - img_new) ** 2)

    # Calculate K - nearest neighbors
    def get_neighbors(self, training_set, training_labels, img_new, k):
        distances = list()
        counter = 0
        for image in training_set:
            distance = self.euclidean_distance(image, img_new)
            distances.append((image, training_labels[counter],  distance))
            counter += 1
        distances.sort(key=lambda tup: tup[2])
        k_neighbors = list()
        for i in range(k):
            k_neighbors.append((distances[i][0], distances[i][1]))
        return k_neighbors

    # Classify a single image using k - nearest neighbors
    def predict_classification(self, training_set, training_labels, img_new, k):
        neighbors = self.get_neighbors(training_set, training_labels, img_new, k)
        output = [image[1] for image in neighbors]
        prediction = max(set(output), key=output.count)
        return prediction

    # Classify set of images
    def classify(self, training_set, training_labels, test_set):
        guesses = []

        if len(training_set) <= 451:
            k = len(training_set)/4
        else:
            k = 95

        for new_image in test_set:
            prediction = self.predict_classification(training_set, training_labels, new_image, k)
            guesses.append(prediction)
        return guesses

