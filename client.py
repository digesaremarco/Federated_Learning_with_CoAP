# federated learning client
import tensorflow as tf
from aiocoap import Context, Message, GET, POST, PUT
import asyncio
import json
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from plot import Plot
from sklearn.metrics import confusion_matrix


class Client:
    def __init__(self, server_ip):
        self.server_ip = server_ip
        self.loss_rate = 0
        #self.weights_percentage = 0
        self.dataframe = pd.DataFrame()

        # definition of GRU model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(128, input_shape=(1, 46)),
            tf.keras.layers.Dense(23, activation='softmax')
        ])

        # compile the model
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    # send model's weights to the server
    async def send_weights(self):
        protocol = await Context.create_client_context()  # create a client context

        model_weights = self.model.get_weights()  # get the weights of the model
        model_weights = [w.tolist() for w in model_weights]  # convert the weights to list
        model_weights_json = json.dumps(model_weights)  # convert the weights to json format
        model_weights_bytes = model_weights_json.encode()  # convert the weights to bytes

        # print the size of the payload in megabytes
        print("Payload size: ", len(model_weights_bytes) / (1024 * 1024), "MB")

        # create a post request with the weights as payload
        request = Message(code=PUT, payload=model_weights_bytes)
        request.set_request_uri(self.server_ip)
        await asyncio.wait_for(protocol.request(request).response, timeout=1000)

    # receive updated weights from the server and update the model
    async def receive_weights(self):
        protocol = await Context.create_client_context()  # create a client context

        # receive the updated weights from the server
        request = Message(code=GET)
        request.set_request_uri(self.server_ip)
        response = await protocol.request(request).response
        # print('Result: %s\n%r' % (response.code, response.payload))

        if response.payload:
            # print the size of the payload in megabytes
            print("Payload size: ", len(response.payload) / (1024 * 1024), "MB")
            weights_json = response.payload.decode()  # convert the payload to string
            weights = json.loads(weights_json)  # extract the weights from the payload
            weights = [tf.convert_to_tensor(w) for w in weights]  # convert the weights to tensors
            self.model.set_weights(weights)
        else:
            print("Received empty payload from the server. No model weights were updated.")

    async def simulate_client(self, loss, accuracy, precision, recall, f1, confusion_matrix_list, loss_or_not):
        pckt_loss = False

        # split the dataframe into train and test
        X = self.dataframe.drop('label', axis=1)  # drop the label column
        y = self.dataframe['label']  # set the label column as target
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # split the dataframe into train and test
        x_train = np.expand_dims(x_train, axis=1)  # expand the dimensions of the train set to fit the model
        x_test = np.expand_dims(x_test, axis=1)  # expand the dimensions of the test set to fit the model

        # one-hot encoding of the labels of 23 classes
        y_train = to_categorical(y_train, num_classes=23)
        y_test = to_categorical(y_test, num_classes=23)

        # train the model
        self.model.fit(x_train, y_train, epochs=1, batch_size=32)

        # simulate packet loss with a probability of loss_rate, change weights into all zeros
        if np.random.rand() < self.loss_rate:
            print("Packet loss occurred.")
            pckt_loss = True
            self.model.set_weights([tf.zeros_like(w) for w in self.model.get_weights()])  # set the weights to zeros
            #weights = self.model.get_weights()
            #num_weights_to_zero = int(self.weights_percentage * len(weights)) # calculate the number of weights to set to zero
            #indices_to_zero = np.random.choice(range(len(weights)), num_weights_to_zero, replace=False) # select the indices of the weights to set to zero
            #for index in indices_to_zero:
            #    weights[index] = np.zeros(weights[index].shape) # set the weights to zero
            #self.model.set_weights(weights) # update the model's weights

        # add the result to the loss_or_not list
        if(pckt_loss):
            loss_or_not.append("Loss")
        else:
            loss_or_not.append("No Loss")

        # send the model to the server
        await self.send_weights()

        # receive the updated model from the server
        await self.receive_weights()

        # evaluate the model
        loss1, accuracy1 = self.model.evaluate(x_test, y_test, batch_size=32)
        loss.append(loss1)
        accuracy.append(accuracy1)

        # calculate the precision, recall and f1 score
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        precision1 = tf.keras.metrics.Precision()
        precision1.update_state(y_test, y_pred)
        precision.append(precision1.result().numpy())
        recall1 = tf.keras.metrics.Recall()
        recall1.update_state(y_test, y_pred)
        recall.append(recall1.result().numpy())
        f1_score = 2 * (precision1.result().numpy() * recall1.result().numpy()) / (
                    precision1.result().numpy() + recall1.result().numpy())
        f1.append(f1_score)

        # calculate the confusion matrix
        confusion_matrix1 = confusion_matrix(y_test, y_pred)
        confusion_matrix_list.append(confusion_matrix1)


# main
async def main():
    # load the datset "C:/Users/diges/Desktop/dataset/prova"
    directory = "C:/Users/diges/Desktop/dataset/prova"
    file_path = []

    # search for all the csv files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path.append(os.path.join(directory, filename))

    # create a dataframe for each csv file
    df = []
    for file in file_path:
        df.append(pd.read_csv(file))

    # concatenate all the dataframes
    dataframe = pd.concat(df, ignore_index=True)  # ignore_index=True is needed to reset the index of the dataframe

    # remove the rows with missing values
    dataframe = dataframe.dropna()

    num_classi = dataframe['label'].nunique()
    print("Numero di classi nel dataset prima del bilanciamento:", num_classi)

    # plot histogram of the label column and save
    #dataframe['label'].value_counts().plot(kind='barh', figsize=(10, 10))
    #plt.savefig("label_histogram.png")

    # balance the dataset
    # Separate classes
    majority_classes = ["DDoS-ICMP_Fragmentation", "MITM-ArpSpoofing", "DDoS-UDP_Fragmentation",
                        "DDoS-ACK_Fragmentation", "DNS_Spoofing", "Mirai-greeth_flood", "Mirai-udpplain",
                        "Mirai-greip_flood"]
    minority_classes = ["Recon-HostDiscovery", "Recon-OSScan", "Recon-PortScan", "DoS-HTTP_Flood", "VulnerabilityScan",
                        "DDoS-HTTP_Flood", "DDoS-SlowLoris", "DictionaryBruteForce", "BrowserHijacking", "SqlInjection",
                        "CommandInjection", "XSS", "Backdoor_Malware", "Recon-PingSweep", "Uploading_Attack"]

    # Sample majority classes to match minority class sizes
    majority_downsampled = pd.concat([dataframe[dataframe['label'] == cls].sample(
        n=max(1, len(dataframe[dataframe['label'] == minority_classes[0]])), random_state=42) for cls in
        majority_classes])

    # Combine downsampled majority classes with minority classes
    dataframe = pd.concat([majority_downsampled, dataframe[dataframe['label'].isin(minority_classes)]])

    # Plot histogram of the label column and save
    #dataframe['label'].value_counts().plot(kind='barh', figsize=(10, 10))
    #plt.savefig("balanced_label_histogram.png")

    num_classi = dataframe['label'].nunique()
    print("Numero di classi nel dataset dopo il bilanciamento:", num_classi)

    # convert the label column to int64
    label_encoder = LabelEncoder()  # create a label encoder
    dataframe['label'] = label_encoder.fit_transform(dataframe['label'])  # encode the labels

    # normalize the dataframe
    scaler = StandardScaler()  # create a scaler
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)  # normalize the dataframe

    num_classi = dataframe['label'].nunique()
    print("Numero di classi nel dataset dopo normalizzazione:", num_classi)

    plot = Plot()  # create an instance of the Plot class
    loss = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    confusion_matrix = []
    loss_rate_list = []
    #weights_percentage_list = []
    loss_or_not = []
    num_clients = 3
    clients = []
    overlap = []
    overlpagging = 0 # overlapping percentage

    # create clients
    for i in range(num_clients):
        clients.append(Client('coap://127.0.0.1:5683/model'))

    # simulate some rounds of federated learning
    for i in range(3):
        print("Round: ", i + 1)
        tasks = []

        for client in clients:
            # create dataframe samples for each client with overlapping
            dataframe_client = dataframe.sample(frac=0.2, random_state=None, replace=True, axis=0) # sample of the dataframe
            overlap.append(dataframe_client.sample(frac=overlpagging, random_state=None, replace=True, axis=0)) # overlapping
            if (len(overlap) > 1):
                # remove rows from dataframe_client and add overlap to dataframe_client to simulate overlapping
                dataframe_client = dataframe_client.sample(frac=1 - overlpagging, random_state=None, replace=True, axis=0)
                dataframe_client = pd.concat([dataframe_client, overlap[-1]], ignore_index=True)
            client.dataframe = dataframe_client

            # simulate packet loss with a probability of loss_rate, change weights into zeros
            loss_rate = np.random.rand() / 10 # loss rate
            #weights_percentage = np.random.rand()  # percentage of weights to set to zero
            #weights_percentage_list.append(round(weights_percentage, 4))
            loss_rate_list.append(round(loss_rate, 4))
            client.loss_rate = 0
            #client.weights_percentage = weights_percentage

            # create a task for each client
            tasks.append(client.simulate_client(loss, accuracy, precision, recall, f1, confusion_matrix, loss_or_not))

        await asyncio.gather(*tasks)  # run all the tasks concurrently

        # add the results to the plot
        plot.add_loss(loss)
        plot.add_accuracy(accuracy)
        plot.add_precision(precision)
        plot.add_recall(recall)
        plot.add_f1(f1)
        plot.add_confusion_matrix(confusion_matrix)
        plot.add_round(i + 1)
        plot.add_loss_rate(loss_rate_list)
        #plot.add_weights_percentage(weights_percentage_list)
        plot.add_loss_or_not(loss_or_not)

        # clear the lists
        loss.clear()
        accuracy.clear()
        precision.clear()
        recall.clear()
        f1.clear()
        loss_rate_list.clear()
        #weights_percentage_list.clear()
        loss_or_not.clear()

    # plot the results
    plot.plot_accuracy()
    plot.plot_loss()
    plot.plot_precision()
    plot.plot_recall()
    plot.plot_f1()
    plot.plot_confusion_matrix()
    plot.plot_loss_rate_table(num_clients)
    #plot.plot_weights_percentage_table(num_clients)
    plot.plot_loss_or_not_table(num_clients)


asyncio.run(main())  # run the main function
