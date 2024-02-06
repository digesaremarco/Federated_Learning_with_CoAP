# class that plots the data
import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self):
        self.loss = [1]
        self.accuracy = [0]
        self.round = [0]
        self.precision = [0]
        self.recall = [0]
        self.f1 = [0]

    # receive the loss vector and calculate the average loss
    def add_loss(self, loss):
        loss = np.mean(loss)  # calculate the average loss
        self.loss.append(loss)

    # receive the accuracy vector and calculate the average accuracy
    def add_accuracy(self, accuracy):
        accuracy = np.mean(accuracy)  # calculate the average accuracy
        self.accuracy.append(accuracy)

    # receive the precision vector and calculate the average precision
    def add_precision(self, precision):
        precision = np.mean(precision)  # calculate the average precision
        self.precision.append(precision)

    # receive the recall vector and calculate the average recall
    def add_recall(self, recall):
        recall = np.mean(recall)  # calculate the average recall
        self.recall.append(recall)

    # receive the f1 vector and calculate the average f1
    def add_f1(self, f1):
        f1 = np.mean(f1)  # calculate the average f1
        self.f1.append(f1)

    # receive the round number
    def add_round(self, round):
        self.round.append(round)

    # plot method
    def plot(self, x, y, xlabel, ylabel, title):
        plt.clf()  # clear the current figure
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(title + ".png")

    # plot the loss
    def plot_loss(self):
        self.plot(self.round, self.loss, "Round", "Loss", "Federated Learning Loss")

    # plot the accuracy
    def plot_accuracy(self):
        self.plot(self.round, self.accuracy, "Round", "Accuracy", "Federated Learning Accuracy")

    # plot the precision
    def plot_precision(self):
        self.plot(self.round, self.precision, "Round", "Precision", "Federated Learning Precision")

    # plot the recall
    def plot_recall(self):
        self.plot(self.round, self.recall, "Round", "Recall", "Federated Learning Recall")

    # plot the f1
    def plot_f1(self):
        self.plot(self.round, self.f1, "Round", "F1", "Federated Learning F1")
