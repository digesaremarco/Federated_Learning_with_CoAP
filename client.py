# federated learning client
import tensorflow as tf
from aiocoap import Context, Message, GET, POST, PUT
import asyncio
import json


class Client:
    def __init__(self, server_ip):
        self.server_ip = server_ip

        # definition of GRU model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(128, input_shape=(28, 28)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # compile the model
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
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
        response = await asyncio.wait_for(protocol.request(request).response, timeout=10)
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

    # load the data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize the data

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
