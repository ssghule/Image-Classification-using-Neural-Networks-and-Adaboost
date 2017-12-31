#!/usr/bin/env python
# Script that contains the Image class


class Image:
    """Object representation of an image
    """
    def __init__(self, line_str):
        """Constructor that parses a string and stores the fields in the fields
        :param line_str: line from the training file
        """
        temp_arr = line_str.split()

        # Integer ID
        self.id = temp_arr[0]

        # Integer orientation
        temp_arr = [int(i) for i in temp_arr[1:]]
        self.orientation = temp_arr[0]

        # Tuple of features
        self.features = tuple(temp_arr[1:])

        # Taking less features
        self.mini_features = self.features[0:192]

        # Predicted orientation
        self.pred_orientation = None

        # Tuple representing output vector. For eg: for 90 degree orientation, output vector will be (0, 1, 0, 0)
        self.output_vector = self.compute_output_vector(self.orientation)

    def __str__(self):
        """Method to print the current object
        """
        print_str = 'ID: ' + self.id + '  Orientation: ' + str(self.orientation)
        print_str += '\nFeatures: ' + str(self.features)
        return print_str

    @staticmethod
    def compute_output_vector(orientation):
        """Computes the output vector. For eg: for 90 degree orientation, output vector will be (0, 1, 0, 0)
        :return: Tuple representing output vector
        """
        out_vector_list = [0] * 4
        index = orientation / 90
        out_vector_list[index] += 1.0
        return tuple(out_vector_list)
