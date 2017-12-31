#!/usr/bin/env python
# Contains the data structure for storing sorted list (in ascending order) of a fixed size


class SortedList:
    """Represents a fixed length sorted list (sorted in ascending order)
    """

    def __init__(self, max_length):
        """Constructor for creating a sorted list (sorted in ascending order)
        :param max_length: The max length of the list
        """
        self.max_length = max_length  # The maximum size of the list
        self.list = list()  # List of tuples where each tuple is of the form (key, val)

    def __str__(self):
        """Returns the string representation of the sorted list
        :return: String representation of the sorted list
        """
        ret_str = ''
        for item in self.list:
            ret_str += 'Key ->' + str(item[0]) + ' Val ->' + str(item[1]) + '\n'
        return ret_str

    def __len__(self):
        """Returns the length of the sorted list
        :return: Length of the sorted list
        """
        return len(self.list)

    def insert(self, key, value):
        """Inserts the key-value tuple in the list sorted by the key
        :param key: int or double object which will be used to insert the object
        :param value: Any object
        """
        if len(self.list) == 0 or key < self.list[len(self.list) - 1]:
            # Finding the index where the item needs to be inserted
            index = 0
            for item in self.list:
                if item[0] >= key:
                    break
                index += 1
            self.list.insert(index, (key, value))
            # If the size of list exceeds the max length, remove the last element
            if len(self.list) > self.max_length:
                del self.list[-1]

    def get(self, index):
        """Returns the value at the given index of the sorted list
        :param index: Index of the item to be fetched
        :return: Value of the item at the given index
        """
        if len(self.list) >= index:
            val = self.list[index][1]
            return val
