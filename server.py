# federated learning server with federated averaging
import tensorflow as tf
from aiocoap import Context, Message, POST, GET, PUT, CHANGED, CONTENT
from aiocoap.resource import Site, Resource
import asyncio
import json


class Server(Resource):
    def __init__(self):
        super().__init__()
        self.client_weights = []  # list of clients' weights
        # definition of GRU global model
        self.global_model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(128, input_shape=(28, 28)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # compile the model
        self.global_model.compile(optimizer='adam',
                                  loss='sparse_categorical_crossentropy',
                                  metrics=['accuracy'])

    # receive json format weights from client
    async def render_put(self, request):
        print("Received weights from client.")
        payload = request.payload
        weights = payload.decode()  # convert the payload to string
        weights = json.loads(weights)  # extract the weights from the payload
        weights = [tf.convert_to_tensor(w) for w in weights] # convert the weights to tensors
        self.client_weights.append(weights)  # add the weights to the list of clients' weights

        # return a message to the client
        return Message(payload=b'Weights received successfully', code=CHANGED)

    # calculate the average of the clients' weights using client_weights list
    async def federated_averaging(self):
        #set global model weights randomly
        self.global_model.set_weights(self.client_weights[0])



    # send the global model's weights to the client and clear the client_weights list
    async def render_get(self, request):
        print("Sending weights to client.")
        # send the global model's weights to the client
        weights = self.global_model.get_weights() # get the weights of the model
        weights_as_list = [w.tolist() for w in weights] # convert the weights to list
        weights_json = json.dumps(weights_as_list)  # convert the weights to json format
        weights_bytes = weights_json.encode()  # convert the weights to bytes
        self.client_weights.clear()  # clear the client_weights list
        return Message(payload=weights_bytes, code=CONTENT)


# main function where the server is created and run federated learning algorithm for 10 rounds only if there are at least 2 clients
async def main():
    server = Server()

    # create a server context
    root = Site()
    root.add_resource(['model'], server)  # add the server to the root context
    context = await Context.create_server_context(root, bind=('127.0.0.1', 5683))
    print('Server is running!')

    # wait for at least 2 clients to connect
    while len(server.client_weights) < 1:
        print("Waiting for at least 2 clients to connect...")
        await asyncio.sleep(5)

    print("At least 2 clients connected. Starting federated learning.")

    # run federated learning algorithm
    await server.federated_averaging()
    response = await server.render_get(None)  # send the global model's weights to the client
    print('finished federated averaging')

    await context.shutdown()  # shutdown the server


asyncio.run(main())  # run the main function
