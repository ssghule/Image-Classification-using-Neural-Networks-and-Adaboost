#!/usr/bin/env python
# Contains the testing and training functions for neural net
# Also contains the building blocks of the neural network like the class Net, Neuron and Connection
#
# We implemented neural net models with a single hidden layer.The number of nodes in the input layer were 192, these 
# coincide with the number of 64 pixels * 3(R,G,B) values of an image. We identified the epochs, learning rate and number 
# of neurons in the hidden layer as the most significant factors that could affect our model.The output neurons were fixed at 
# 4 in accordance with 4 labels. The activation functions used on the hidden and output layers were sigmoid and tanh.  
#

import math
import random
from time import time

random.seed(250)  # Seeding in order to get deterministic random values


class Connection:
    """Class that contains the attributes for representing a connection between 2 neurons
    """

    def __init__(self, weight):
        """Constructor for Connection
        :param weight: Double weight of an edge between 2 neurons
        """
        self.weight = weight  # Weight of an edge between 2 neurons
        self.delta_weight = 0.0  # The change in weight computed after back propagation


class Neuron:
    """Class that contains the attributes and methods for a neuron
    """

    # Class constants
    eta = 0.15  # Overall net learning rate [0.0, 1.0]
    alpha = 0.50  # Multiplier of last weight change, momentum [0.0, 1.0]

    def __init__(self, my_idx, num_outputs):
        """Constructor for Neuron
        :param num_outputs: The number of out-links this neuron has
        """
        self.output_val = 1.0
        self.my_idx = my_idx
        self.gradient = 0.0
        # Each element in output weights is a Connection object
        self.output_weights = list()
        for i in range(num_outputs):
            self.output_weights.append(Connection(self.random_weight()))

    @staticmethod
    def random_weight():
        """Generates and returns a random number
        :return: A random double in the range (-1, 1)
        """
        return random.uniform(-1, 1)

    @staticmethod
    def transfer_function(weighted_sum):
        """Performs activation function on the weighted_sum
        :param weighted_sum: The weighted sum from the previous layer
        :return: Transformed weighted sum
        """
        # Performing tanh on the weighted sum. Range (-1.0, 1.0)
        return math.tanh(weighted_sum)
        # Performing sigmoid transformation of the value
        # return 1 / (1 + math.exp(-weighted_sum))

    @staticmethod
    def transfer_function_derivative(weighted_sum):
        """Performs derivative of activation function on the weighted_sum
        :param weighted_sum: The weighted sum from the previous layer
        :return: Transformed weighted sum
        """
        # derivative of tanh(x) is (1 - tanh(x)^2)
        # The weighted sum is already tanh(x) so we just return 1 - weighted_sum ^ 2
        return 1.0 - (weighted_sum ** 2)

        # If y = sigmoid(x), then dy/dx = y(1 - y)
        # return weighted_sum * (1 - weighted_sum)

    def feed_forward(self, prev_layer):
        """Performs forward propagation by computing output value of a neuron
        :param prev_layer: List of previous layer neurons
        """
        weighted_sum = 0.0

        for prev_neuron in prev_layer:
            weighted_sum += prev_neuron.output_val * prev_neuron.output_weights[self.my_idx].weight

        # print 'Weighted sum ->', weighted_sum
        self.output_val = self.transfer_function(weighted_sum)

    def calc_output_gradient(self, target_val):
        """Computes the target values for the output neuron and stores it in the attribute 'gradient'
        :param target_val: The actual value of the output neuron
        """
        delta = target_val - self.output_val
        self.gradient = delta * self.transfer_function_derivative(self.output_val)
        # print 'Output gradient ->', str(self.gradient), 'for neuron ->', self.my_idx, \
        #       'Target value ->', target_val, 'Output value ->', self.output_val

    def sum_derivatives_of_weight(self, next_layer):
        """Computes sum of derivatives of weights of the next layer
        :param next_layer: List of next layer neurons
        :return: Sum of weight derivatives
        """
        dow_sum = 0.0
        # Sum our contributions of the errors at the nodes we feed
        for neuron_idx in range(len(next_layer) - 1):
            dow_sum += (self.output_weights[neuron_idx].weight * next_layer[neuron_idx].gradient)
        return dow_sum

    def calc_hidden_gradient(self, next_layer):
        """Computes gradient of the hidden layer neurons and store it in the attribute 'gradient'
        :param next_layer: List of next layer neurons
        """
        dow_sum = self.sum_derivatives_of_weight(next_layer)
        self.gradient = dow_sum * self.transfer_function_derivative(self.output_val)

    def update_input_weights(self, prev_layer):
        """This method is called after back propagation to update the input weights
        :param prev_layer: List of previous layer neurons
        """
        # The weights to be updated are in the connection container
        # in the neurons in the preceding layer
        for neuron in prev_layer:
            old_delta_weight = neuron.output_weights[self.my_idx].delta_weight
            new_delta_weight = (neuron.eta * neuron.output_val * self.gradient) + (neuron.alpha * old_delta_weight)
            neuron.output_weights[self.my_idx].delta_weight = new_delta_weight
            neuron.output_weights[self.my_idx].weight += new_delta_weight

    def __str__(self):
        """Returns the string representation of the object for printing
        :return: The string representation of the object
        """
        ret_str = 'My index:' + str(self.my_idx) + '\nOutput weights:'
        for conn in self.output_weights:
            ret_str += str(conn.weight) + ' '
        ret_str += '\n'
        return ret_str


class Net:
    """Class that contains the attributes and methods for neural network
    """

    normalization_factor = 255.0  # The max pixel value

    def __init__(self, topology):
        """Constructor to create a neural net
        :param topology: List that contains the number of neurons in each layer
        """
        # Represent the layers in the neural net
        self.layers = list()

        # Each layer contains a list of neurons
        for layer_num, neuron_count in enumerate(topology):
            neuron_list = list()
            num_outputs = 0 if layer_num == len(topology) - 1 else topology[layer_num + 1]
            for index in range(neuron_count + 1):
                neuron_list.append(Neuron(index, num_outputs))
            self.layers.append(neuron_list)

        self.error = 0.0

    def feed_forward(self, input_vals):
        """Performs forward propagation in the network
        :param input_vals: List of input parameters
        """
        assert len(input_vals) == len(self.layers[0]) - 1

        # Assigning input values to the input neurons
        for index in range(len(input_vals)):
            self.layers[0][index].output_val = (input_vals[index] / self.normalization_factor)

        # Forward propagation
        for index in range(1, len(self.layers)):
            prev_layer = self.layers[index - 1]
            for neuron_idx in range(len(self.layers[index]) - 1):
                self.layers[index][neuron_idx].feed_forward(prev_layer)

    def back_prop(self, target_vals):
        """Performs back propagation in the network by computing gradients of all the neurons (except the input layer)
        :param target_vals: List of output values
        """
        # Calculate overall net error (RMS of output neuron errors)
        output_layer = self.layers[len(self.layers) - 1]
        error = 0.0

        for neuron_idx in range(len(target_vals)):
            delta = target_vals[neuron_idx] - output_layer[neuron_idx].output_val
            error += (delta ** 2)

        error /= (len(target_vals))  # Average error squared
        self.error = math.sqrt(error)  # RMS
        # print 'Total error ->', self.error

        # Calculate output layer gradients
        for neuron_idx in range(0, len(output_layer) - 1):
            output_layer[neuron_idx].calc_output_gradient(target_vals[neuron_idx])

        # Calculate gradients on hidden layers
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            hidden_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            for hidden_neuron in hidden_layer:
                hidden_neuron.calc_hidden_gradient(next_layer)

        # For all layers from output to first hidden layer,
        # update connection weights
        for layer_idx in range(len(self.layers) - 1, 0, -1):
            curr_layer = self.layers[layer_idx]
            prev_layer = self.layers[layer_idx - 1]

            for neuron_idx in range(0, len(curr_layer) - 1):
                neuron = curr_layer[neuron_idx]
                neuron.update_input_weights(prev_layer)

    def get_results(self):
        """Returns the results of the output layer (that are produced after forward propagation)
        :return: An array of predicted output
        """
        results = list()
        output_layer = self.layers[len(self.layers) - 1]
        for idx in range(0, len(output_layer) - 1):
            results.append(output_layer[idx].output_val)
        return results

    def __str__(self):
        """Returns the string representation of the neural net object for printing
        :return: The string representation of the Neural Net object
        """
        ret_str = ''
        for layer_num, layer in enumerate(self.layers):
            ret_str += ('Layer num ->' + str(layer_num) + '\n')
            for neuron in layer:
                ret_str += str(neuron)
        ret_str += 'Error ->' + str(self.error) + '\n'
        return ret_str


def train(train_images):
    """Performs training on the images and computes a neural net classifier
    :param train_images: List of image objects
    :return: The trained neural net (Net object)
    """
    # topology is a list that contains the number of neurons in each layer
    # [3, 2, 1] suggests 3 neurons in the input layer, 2 neurons in the hidden layer
    # and 1 neuron in the output layer
    topology = [192, 10, 4]
    net = Net(topology)
    epochs = 10  # The number of times the training images is fed into the network

    for i in range(epochs):
        print time()
        print 'Epoch ->', i
        for image in train_images:
            net.feed_forward(image.mini_features)
            net.back_prop(image.output_vector)
    return net


def test(net, test_images):
    """Finds the orientation of the test images
    :param net: Net object (contains the trained neural net object)
    :param test_images: List of images for which orientation need to be determined
    """
    for image in test_images:
        net.feed_forward(image.mini_features)
        result_vals = net.get_results()
        print 'Result values ->', result_vals
        image.pred_orientation = result_vals.index(max(result_vals)) * 90
