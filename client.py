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



class Client:
    def __init__(self, server_ip):
        self.server_ip = server_ip

        # definition of GRU model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(128, input_shape=(1, 46)),
            tf.keras.layers.Dense(34, activation='softmax')
        ])

        # compile the model
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # send model's weights to the server
    async def send_weights(self):
        protocol = await Context.create_client_context()  # create a client context

        model_weights = self.model.get_weights()  # get the weights of the model
        model_weights = [w.tolist() for w in model_weights] # convert the weights to list
        model_weights_json = json.dumps(model_weights)  # convert the weights to json format
        model_weights_bytes = model_weights_json.encode() # convert the weights to bytes

        # create a post request with the weights as payload
        request = Message(code=PUT, payload=model_weights_bytes)
        request.set_request_uri(self.server_ip)
        print('attesa')
        response = await asyncio.wait_for(protocol.request(request).response, timeout=1000)
        #response = await protocol.request(request).response
        print('Result: %s\n%r' % (response.code, response.payload))

    # receive updated weights from the server and update the model
    async def receive_weights(self):
        protocol = await Context.create_client_context()  # create a client context

        # receive the updated weights from the server
        request = Message(code=GET)
        request.set_request_uri(self.server_ip)
        response = await protocol.request(request).response
        print('Result: %s\n%r' % (response.code, response.payload))

        if response.payload:
            weights_json = response.payload.decode() # convert the payload to string
            weights = json.loads(weights_json) # extract the weights from the payload
            weights = [tf.convert_to_tensor(w) for w in weights] # convert the weights to tensors
            self.model.set_weights(weights)
        else:
            print("Received empty payload from the server. No model weights were updated.")



    # main
async def main():
    # client creation
    client = Client('coap://127.0.0.1:5683/model')

    # load the datset "C:\Users\diges\Desktop\dataset\prova"
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
    dataframe = pd.concat(df, ignore_index=True) # ignore_index=True is needed to reset the index of the dataframe
    print(dataframe.head())  # print the first 5 rows of the dataframe
    #print the columns of the dataframe and their type
    print(dataframe.dtypes)

    # convert the label column to int64
    label_encoder = LabelEncoder() # create a label encoder
    dataframe['label'] = label_encoder.fit_transform(dataframe['label']) # encode the labels
    print(dataframe.dtypes)

    # split the dataframe into train and test
    X = dataframe.drop('label', axis=1) # drop the label column
    y = dataframe['label'] # set the label column as target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # split the dataframe into train and test
    x_train = np.expand_dims(x_train, axis=1) # expand the dimensions of the train set to fit the model
    x_test = np.expand_dims(x_test, axis=1) # expand the dimensions of the test set to fit the model

    # one-hot encoding of the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(y_test.shape)


    # train the model
    client.model.fit(x_train, y_train, epochs=1, batch_size=32)

    # send the model to the server
    await client.send_weights()
    print("Weights sent to the server.")


    # receive the updated model from the server
    await client.receive_weights()
    print("Weights received from the server.")

    # evaluate the model
    client.model.evaluate(x_test, y_test, batch_size=32)


asyncio.run(main()) # run the main function
