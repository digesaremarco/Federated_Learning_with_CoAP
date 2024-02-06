# class that plots the data
import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.round = []

    # receive the loss vector and calculate the average loss
    def add_loss(self, loss):
        loss = np.mean(loss) # calculate the average loss
        self.loss.append(loss)

    # receive the accuracy vector and calculate the average accuracy
    def add_accuracy(self, accuracy):
        accuracy = np.mean(accuracy) # calculate the average accuracy
        self.accuracy.append(accuracy)

    # receive the round number
    def add_round(self, round):
        self.round.append(round)

    # plot method
    def plot(self, x, y, xlabel, ylabel, title):
        plt.clf() # clear the current figure
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