# federated learning client
import tensorflow as tf
from aiocoap import Context, Message, GET, POST, PUT
import asyncio
import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from plot import Plot


class Client:
    def __init__(self, server_ip):
        self.server_ip = server_ip

        # definition of GRU model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(128, input_shape=(1, 46)),
            tf.keras.layers.Dense(4, activation='softmax')
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
            weights_json = response.payload.decode()  # convert the payload to string
            weights = json.loads(weights_json)  # extract the weights from the payload
            weights = [tf.convert_to_tensor(w) for w in weights]  # convert the weights to tensors
            self.model.set_weights(weights)
        else:
            print("Received empty payload from the server. No model weights were updated.")

    async def simulate_client(self, client, dataframe, loss, accuracy, precision, recall, f1, loss_rate):
        # take 50% random of the dataframe
        dataframe = dataframe.sample(frac=0.5, random_state=1)

        # split the dataframe into train and test
        X = dataframe.drop('label', axis=1)  # drop the label column
        y = dataframe['label']  # set the label column as target
        x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3)  # split the dataframe into train and test
        x_train = np.expand_dims(x_train, axis=1)  # expand the dimensions of the train set to fit the model
        x_test = np.expand_dims(x_test, axis=1)  # expand the dimensions of the test set to fit the model

        # one-hot encoding of the labels
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # train the model
        client.model.fit(x_train, y_train, epochs=1, batch_size=32)

        # simulate packet loss with a probability of loss_rate, change weights into zeros
        if np.random.rand() < loss_rate:
            print("Packet loss occurred.")
            weights = client.model.get_weights()
            weights_percentage = 0.4 # percentage of weights to set to zero
            num_weights_to_zero = int(weights_percentage * len(weights)) # calculate the number of weights to set to zero
            indices_to_zero = np.random.choice(range(len(weights)), num_weights_to_zero, replace=False) # select the indices of the weights to set to zero
            for index in indices_to_zero:
                weights[index] = np.zeros(weights[index].shape) # set the weights to zero
            client.model.set_weights(weights) # update the model's weights

        # send the model to the server
        await client.send_weights()

        # receive the updated model from the server
        await client.receive_weights()

        # evaluate the model
        loss1, accuracy1 = client.model.evaluate(x_test, y_test, batch_size=32)
        loss.append(loss1)
        accuracy.append(accuracy1)

        # calculate the precision, recall and f1 score
        y_pred = client.model.predict(x_test)
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

    # convert the label column to int64
    label_encoder = LabelEncoder()  # create a label encoder
    dataframe['label'] = label_encoder.fit_transform(dataframe['label'])  # encode the labels

    # remove the rows with missing values
    dataframe = dataframe.dropna()

    # normalize the dataframe
    scaler = StandardScaler()  # create a scaler
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)  # normalize the dataframe

    # preprocessing of the dataframe
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # create an imputer
    dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)  # impute the missing values

    plot = Plot()  # create an instance of the Plot class
    loss = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    loss_rate = 0
    loss_rate_list = [0]
    num_clients = 2
    clients = []
    # create 2 clients
    for i in range(num_clients):
        clients.append(Client('coap://127.0.0.1:5683/model'))
    # simulate 10 rounds of federated learning
    for i in range(3):
        print("Round: ", i + 1)
        tasks = []
        for client in clients:
            tasks.append(client.simulate_client(client, dataframe, loss, accuracy, precision, recall, f1, loss_rate))
            loss_rate_list.append(loss_rate)
        await asyncio.gather(*tasks)  # run all the tasks concurrently
        plot.add_loss(loss)
        plot.add_accuracy(accuracy)
        plot.add_precision(precision)
        plot.add_recall(recall)
        plot.add_f1(f1)
        plot.add_round(i + 1)
        plot.add_loss_rate(loss_rate_list)
        loss.clear()
        accuracy.clear()
        precision.clear()
        recall.clear()
        f1.clear()
        loss_rate_list = [0]
        loss_rate += 0.1
    plot.plot_accuracy()
    plot.plot_loss()
    plot.plot_precision()
    plot.plot_recall()
    plot.plot_f1()
    plot.plot_loss_rate_table(num_clients)


asyncio.run(main())  # run the main function
