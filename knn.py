#!/usr/bin/env python
# Program that performs knn training and testing

from collections import defaultdict
from sorted_list import SortedList

# We look for 10 nearest points
K = 10


def train(image_list):
    """No training required in knn, as list of training images itself is the model
    :param image_list: List of training images
    :return: Model to serialize (list of training images)
    """
    return image_list


def test(image_list, model_images):
    """Performs testing by performing K-nearest neighbors classification
       Stores the predicted orientation in the object itself
    :param image_list: List of test images
    :param model_images: List of training images
    """

    def calculate_distance(image1, image2):
        """Calculates the Euclidean distance between 2 images
        :param image1: First image object
        :param image2: Second image object
        :return: Euclidean distance between the 2 images
        """
        dist = 0
        for f_image1, f_image2 in zip(image1.features, image2.features):
            dist += (f_image1 - f_image2) ** 2
        return 0.0 + dist

    # test() starts from here
    for test_image in image_list:
        # Traversing all the model images and calculating the distance
        least_dist = SortedList(K)
        for model_image in model_images:
            curr_dist = calculate_distance(test_image, model_image)
            least_dist.insert(curr_dist, model_image)

        # Get a voting from all the K nearest neighbors
        model_dict = defaultdict(lambda: 0)
        for index in range(K):
            curr_img = least_dist.get(index)
            model_dict[curr_img.orientation] += 1

        # print 'Printing model_dict:', model_dict
        max_orientation = max(model_dict, key=model_dict.get)
        test_image.pred_orientation = max_orientation
