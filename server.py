# federated learning server with federated averaging
import tensorflow as tf
from aiocoap import Context, Message, PUT, GET
from aiocoap.resource import Site, Resource
import asyncio
import json


class Server:
    def __init__(self):
        self.client_weights = []  # list of clients' weights
        # definition of GRU global model
        global_model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(128, input_shape=(28, 28)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # compile the model
        global_model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])

    # receive json format weights from client
    async def receive_weights(self, request):
        payload = request.payload
        weights = json.loads(payload) # extract the weights from the payload
        self.client_weights.append(weights) # add the weights to the list of clients' weights
        return Message(payload=b'Weights received successfully', code=PUT)

    # calculate the average of the clients' weights using client_weights list
    async def federated_averaging(self):
        # calculate the average of the clients' weights and save it to the global model
        global_weights = self.client_weights[0] # initialize the global weights with the first client's weights

        # calculate the average of the clients' weights
        for i in range(1, len(self.client_weights)):
            for j in range(len(global_weights)):
                global_weights[j] = (global_weights[j] + self.client_weights[i][j]) / len(self.client_weights)

        # save the average of the clients' weights to the global model
        self.global_model.set_weights(global_weights)

    # send the global model's weights to the client and clear the client_weights list
    async def send_weights(self):
        # send the global model's weights to the client
        weights = self.global_model.get_weights()
        weights = json.dumps(weights) # convert the weights to json format
        self.client_weights.clear() # clear the client_weights list
        return Message(payload=weights, code=GET)

# main function where the server is created and run federated learning algorithm for 10 rounds only if there are at least 2 clients
async def main():
    server = Server()

    # create a server context
    root = Site()
    root.add_resource(['model'], server) # add the server to the root context
    #root.add_resource(('receive_weights',), server.receive_weights())
    #root.add_resource(('federated_averaging',), server.federated_averaging())
    #root.add_resource(('send_weights',), server.send_weights())
    context = await Context.create_server_context(root, bind=('127.0.0.1', 5683))
    print('Server is running!')

    # wait for at least 2 clients to connect
    while len(context.servers) < 2:
        print("Waiting for at least 2 clients to connect...")
        await asyncio.sleep(5)

    print("At least 2 clients connected. Starting federated learning.")

    # run federated learning algorithm for 10 rounds
    for i in range(10):
        await server.federated_averaging()
        print('Round %d finished successfully!' % (i + 1))
        response = await server.send_weights()  # send the global model's weights to the client
        print('Result: %s\n%r' % (response.code, response.payload))

    await context.shutdown() # shutdown the server

asyncio.run(main()) # run the main function
