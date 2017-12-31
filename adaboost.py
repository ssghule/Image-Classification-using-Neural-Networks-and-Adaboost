#!/usr/bin/python
#
# Authors: Sharad Ghule, Gaurav Derasaria, Saurabh Agrawal
# (based on skeleton code by D. Crandall, Oct 2017)
#

import sys
import numpy as np
import math
import datetime


class Adaboost:
    def __init__(self, train_data):
        self.id = []
        self.train = []
        self.labs = []
        self.nrec = len(train_data)
        self.weights = np.array([[1 / float(self.nrec)] * 4] * self.nrec)
        self.curr_labs = []
        self.alpha = dict()
        self.learners = dict()
        self.outputs = dict()
        self.iterations = 20
        self.prediction = []

        # Below loop converts the orientation in [0,90,180,270] format to a binary vector for each sample.
        # E.g. 0 would be [1,-1,-1,-1] and 90 would be [-1,1,-1,-1]
        # This had to be done so that each orientation could be used as a binary classifier.
        for i in train_data:
            self.id.append(i.id)
            self.train.append(np.array(i.features))
            self.labs.append(i.orientation)
            self.curr_labs.append([1 if i.orientation == 0 else -1] + [1 if i.orientation == 90 else -1] + [
                1 if i.orientation == 180 else -1] + [1 if i.orientation == 270 else -1])
        self.curr_labs = np.array(self.curr_labs)

    def train_learners(self):
        '''This function trains the weak learners where a learner outputs 1 if a condition is matched and -1 when it is not.
        Here, the condition is if a pixel is greater than another. Only top two rows of pixels are compared with the bottom two to improve performance.'''
        l = range(192)
        rr = l[18::24] + l[21::24]
        gr = l[19::24] + l[22::24]
        br = l[20::24] + l[23::24]
        rl = l[0::24] + l[3::24]
        gl = l[1::24] + l[4::24]
        bl = l[2::24] + l[5::24]
        right = rr + gr + br
        left = rl + gl + bl
        print "started training learners at", datetime.datetime.now()
        for i in range(48):
            for j in range(144, 192):
                cname = (i, j)  # classifier i>j
                output = np.array([1 if x[i] > x[j] else -1 for x in self.train])
                self.outputs[cname] = output
        '''for i in left:
            for j in right:
                cname = (i, j)  # classifier i>j
        if cname not in self.outputs.keys():
                    output = np.array([1 if x[i] > x[j] else -1 for x in self.train])
            self.outputs[cname] = output'''
        print "finished training learners at", datetime.datetime.now()

    def get_best_learner(self):
        '''This function finds the weak classifiers for each of 0,90,180 and 270 binary classifiers basis the no. of examples that are misclassified.
        Error rate is calculated for each of the classifiers and the one with minimum error is chosen.'''
        print "finding best learners:", datetime.datetime.now()
        error_0 = dict()
        error_90 = dict()
        error_180 = dict()
        error_270 = dict()

        for learner in self.outputs:
            error_0[learner] = sum([y if x == -1 else 0 for x, y in
                                    zip(self.outputs[learner] * self.curr_labs[:, 0], self.weights[:, 0])]) / float(
                self.nrec)
            error_90[learner] = sum([y if x == -1 else 0 for x, y in
                                     zip(self.outputs[learner] * self.curr_labs[:, 1], self.weights[:, 1])]) / float(
                self.nrec)
            error_180[learner] = sum([y if x == -1 else 0 for x, y in
                                      zip(self.outputs[learner] * self.curr_labs[:, 2], self.weights[:, 2])]) / float(
                self.nrec)
            error_270[learner] = sum([y if x == -1 else 0 for x, y in
                                      zip(self.outputs[learner] * self.curr_labs[:, 3], self.weights[:, 3])]) / float(
                self.nrec)

        # print "sum of lsums", np.sum(lsums_0[(11,108)])
        v_0 = list(error_0.values())
        k_0 = list(error_0.keys())

        v_90 = list(error_90.values())
        k_90 = list(error_90.keys())

        v_180 = list(error_180.values())
        k_180 = list(error_180.keys())

        v_270 = list(error_270.values())
        k_270 = list(error_270.keys())

        best_0 = k_0[v_0.index(min(v_0))]
        best_90 = k_90[v_90.index(min(v_90))]
        best_180 = k_180[v_180.index(min(v_180))]
        best_270 = k_270[v_270.index(min(v_270))]

        # Alpha, i.e. a weights to be given to each weak classifier is calculated basis the error rate
        # and is used to predict the final output.
        print "calculating alphas at", datetime.datetime.now()

        alpha_0 = 0.5 * math.log((1 - min(v_0)) / min(v_0))
        alpha_90 = 0.5 * math.log((1 - min(v_90)) / min(v_90))
        alpha_180 = 0.5 * math.log((1 - min(v_180)) / min(v_180))
        alpha_270 = 0.5 * math.log((1 - min(v_270)) / min(v_270))
        return [(best_0, alpha_0), (best_90, alpha_90), (best_180, alpha_180), (best_270, alpha_270)]

    def update_weights(self, learners):
        '''This function updates the weights of samples basis the classification.
        The misclassified examples are given more weight and the correctly classified ones are given less weight.'''
        print "updating weights at", datetime.datetime.now()
        l_0 = learners[0][0]
        alpha_0 = learners[0][1]
        l_90 = learners[1][0]
        alpha_90 = learners[1][1]
        l_180 = learners[2][0]
        alpha_180 = learners[2][1]
        l_270 = learners[3][0]
        alpha_270 = learners[3][1]

        # print "y", self.curr_labs[:,0]
        # print "learner",self.learners[l_0][:,0]
        wt_0 = [self.weights[:, 0] * np.exp(-alpha_0 * self.outputs[l_0] * self.curr_labs[:, 0])]
        self.weights[:, 0] = wt_0 / np.sum(wt_0)

        wt_90 = [self.weights[:, 1] * np.exp(-alpha_90 * self.outputs[l_90] * self.curr_labs[:, 1])]
        self.weights[:, 1] = wt_90 / np.sum(wt_90)

        wt_180 = [self.weights[:, 2] * np.exp(-alpha_180 * self.outputs[l_180] * self.curr_labs[:, 2])]
        self.weights[:, 2] = wt_180 / np.sum(wt_180)

        wt_270 = [self.weights[:, 3] * np.exp(-alpha_270 * self.outputs[l_270] * self.curr_labs[:, 3])]
        self.weights[:, 3] = wt_270 / np.sum(wt_270)
        print "finished updating weights at", datetime.datetime.now()

    def predict(self, cl_list):
        ''' This function predicts the final classification basis alpha values of each of the classifiers and sums the total value.
        '''
        output = np.array([[float(0)] * 4] * self.nrec)
        for cls in cl_list:
            cls_0 = cls[0][0]
            alpha_0 = cls[0][1]

            cls_90 = cls[1][0]
            alpha_90 = cls[1][1]

            cls_180 = cls[2][0]
            alpha_180 = cls[2][1]

            cls_270 = cls[3][0]
            alpha_270 = cls[3][1]

            output[:, 0] += self.outputs[cls_0] * alpha_0
            output[:, 1] += self.outputs[cls_90] * alpha_90
            output[:, 2] += self.outputs[cls_180] * alpha_180
            output[:, 3] += self.outputs[cls_270] * alpha_270
        #		print output
        return output

    def predict_orientation(self, output):
        '''This function predicts the orientation basis the weight each example has for each classfier output. If classifier for 0 has highest weight
        among classifiers for 90, 180 and 270, it would be classified as 0. '''
        orientation = []
        for t in output:
            o = t.tolist()
            orientation.append(o.index(max(o)))
        orientation = [0 if o == 0 else 90 if o == 1 else 180 if o == 2 else 270 for o in orientation]
        return orientation

    def check_accuracy(self, orientation):
        cnt = 0
        for i, j in zip(self.labs, orientation):
            if i == j:
                cnt += 1
        print "Accuracy=", cnt / float(self.nrec)

    def adaboost(self):
        '''This function trains the classifiers and chooses best classifiers iteratively.'''
        print datetime.datetime.now()
        self.train_learners()
        classifier_list = []
        for t in range(self.iterations):
            print "\nIteration", t, "started at", datetime.datetime.now()
            l = self.get_best_learner()
            classifier_list.append(l)
            self.update_weights(l)
        print "finished training at", datetime.datetime.now()
        return classifier_list

    def adaboost_test(self, train_data, params):
        '''Test function'''
        self.train_learners()
        output = self.predict(params)
        orientation = self.predict_orientation(output)
        for i, j in zip(train_data, orientation):
            i.pred_orientation = j
