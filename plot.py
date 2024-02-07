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
        self.loss_rate = []
        self.weights_percentage = []

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

    # receive the loss rate vector
    def add_loss_rate(self, loss_rate):
        self.loss_rate.append(loss_rate.copy())  # add the loss rate to the list by copying it to avoid reference

    # receive the weights percentage vector
    def add_weights_percentage(self, weights_percentage):
        self.weights_percentage.append(weights_percentage.copy())  # add the weights percentage to the list by copying it to avoid reference

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


    # generic plot table method
    def plot_table(self, num_clients, data, title):
        plt.clf()
        # set the number of rows and columns in the table
        rows = num_clients + 1
        columns = len(self.round)
        cell_text = []  # create an empty list to store the cell text
        for i in range(rows):
            cell_text.append([])
            for j in range(columns):
                if i == 0 and j == 0:
                    cell_text[i].append("Round/Client")
                if i == 0 and j > 0:
                    cell_text[i].append("Round " + str(j))
                if i > 0 and j == 0:
                    cell_text[i].append("Client " + str(i))
                if i > 0 and j > 0:
                    cell_text[i].append(str(round(data[j - 1][i - 1], 4)))
        # create a table
        table = plt.table(cellText=cell_text, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 1.5)
        plt.axis('off')
        plt.savefig(title + ".png")


    # plot the loss rate table for each round and each client
    def plot_loss_rate_table(self, num_clients):
        self.plot_table(num_clients, self.loss_rate, "Federated Learning Loss Rate Table")

    # plot the weights percentage table for each round and each client
    def plot_weights_percentage_table(self, num_clients):
        self.plot_table(num_clients, self.weights_percentage, "Federated Learning Weights Percentage Table")
