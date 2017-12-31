#!/usr/bin/env python
# Contains the main program for performing image orientation detection
# Usage: python orient.py [train / test] [train-data.txt / test-data.txt] [model-file.txt] [nearest / adaboost / nnet]

import sys
from image import Image
import knn
import neural_net
import cPickle as Pickle
from time import time
from adaboost import Adaboost


def print_input():
    """Prints the program inputs
    """
    print 'Phase:', phase
    print 'Model:', model
    for image in image_list:
        print image


def serialize_to_file(obj, filename):
    """Serializes object to a file
    :param obj: Object that needs to be serialized to the file
    :param filename: File in which the object needs to be serialized
    """
    Pickle.dump(obj, open(filename, 'wb'), protocol=Pickle.HIGHEST_PROTOCOL)


def deserialize_from_file(filename):
    """Deserializes the object from the file
    :param filename: Name of the file from which the object needs be deserialized
    :return: The deserialized object from the file
    """
    return Pickle.load(open(filename, 'rb'))


def save_output(img_list, outfile_name):
    """Writes the output in the correct format in the output file
    :param img_list: List of images
    :param outfile_name: Name of the output file
    """
    with open(outfile_name, 'w') as out_file:
        out_str = ''
        for image in img_list:
            out_str += image.id + ' ' + str(image.pred_orientation) + '\n'
        out_file.write(out_str)


def get_accuracy(img_list):
    """Determines the accuracy by comparing actual orientation with predicted orientation
    :param img_list: List of images for which accuracy needs to be tested
    :return: Double accuracy
    """
    correct = 0
    for image in img_list:
        if image.orientation == image.pred_orientation:
            correct += 1
    return (0.0 + correct) / len(img_list)


# Main program
if __name__ == '__main__':
    # Getting command line inputs
    phase = sys.argv[1]  # 'train' phase or 'test' phase
    file_name = sys.argv[2]  # Name of file that contains train images in train phase and test images in test phase
    model_file = sys.argv[3]  # Name of file that contains the training model in test phase
    model = sys.argv[4]  # Model that is to be used to train or test. Values ('nearest', 'adaboost', 'nnet')

    # sys.stdout = open('sysout.txt', 'w')

    # Parsing the input file and creating the image objects
    image_list = list()
    with open(file_name, 'r') as t_file:
        for line in t_file:
            image_list.append(Image(line))

    print 'Start time', time()

    if model == 'best':
        model = 'nnet'

    # K-Nearest neighbors
    if model == 'nearest' and phase == 'train':
        model = knn.train(image_list)
        serialize_to_file(model, model_file)
    elif model == 'nearest' and phase == 'test':
        model = deserialize_from_file(model_file)
        knn.test(image_list, model)

    # ADA boost
    elif model == "adaboost" and phase == "train":
        params = Adaboost(image_list).adaboost()
        serialize_to_file(params, model_file)
    elif model == "adaboost" and phase == "test":
        params = deserialize_from_file(model_file)
        Adaboost(image_list).adaboost_test(image_list, params)

    # Neural net
    elif model == 'nnet' and phase == 'train':
        net = neural_net.train(image_list)
        serialize_to_file(net, model_file)
    elif model == 'nnet' and phase == 'test':
        net = deserialize_from_file(model_file)
        neural_net.test(net, image_list)

    print 'End time', time()

    if phase == 'test':
        accuracy = get_accuracy(image_list)
        save_output(image_list, 'output.txt')
        print 'Accuracy ->', accuracy
