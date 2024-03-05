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
        self.num_clients = 0  # number of clients connected to the server
        self.clients = 10  # number of clients needed to start federated averaging
        self.send_weights = asyncio.Event()  # event to send the weights to the server
        self.counter_loss = 0  # counter for the loss

        # definition of GRU model
        self.global_model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(128, input_shape=(1, 46)),
            tf.keras.layers.Dense(23, activation='softmax')
        ])

        # compile the model
        self.global_model.compile(optimizer='adam',
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

    # receive json format weights from client
    async def render_put(self, request):
        self.num_clients += 1  # increment the number of clients connected to the server
        if (len(self.client_weights) < self.clients):
            payload = request.payload
            weights = payload.decode()  # convert the payload to string
            weights = json.loads(weights)  # extract the weights from the payload
            weights = [tf.convert_to_tensor(w) for w in weights]  # convert the weights to tensors

            # check if the weights aren't all zeros
            if not all(tf.reduce_all(tf.equal(w, 0)) for w in weights):
                self.client_weights.append(weights)  # add the weights to the list of clients' weights
            else:
                self.counter_loss += 1  # increment the counter for the loss

            print(len(self.client_weights) + self.counter_loss, "clients connected to the server.")
            # return a message to the client
            return Message(payload=b'Weights received successfully', code=CHANGED)
        else:  # if there are already 2 clients connected to the server
            return Message(payload=b'Weights not saved.', code=CHANGED)

    # calculate the average of the clients' weights using client_weights list
    async def federated_averaging(self):
        # check if there is at least one model in the list
        if len(self.client_weights) > 0:
            averaged_weights = [tf.reduce_mean(weights, axis=0) for weights in
                                zip(*self.client_weights)]  # calculate the average of the clients' weights
        else: # set the weights to zeros if there are no models in the list
            averaged_weights = self.global_model.get_weights()
            averaged_weights = [tf.zeros_like(w) for w in averaged_weights]
        self.global_model.set_weights(averaged_weights)  # update the global model's weights

    # send the global model's weights to the client and clear the client_weights list
    async def render_get(self, request):
        await self.send_weights.wait()  # wait until the event is set
        print("Sending weights to client.")
        # send the global model's weights to the client
        weights = self.global_model.get_weights()  # get the weights of the model
        weights_as_list = [w.tolist() for w in weights]  # convert the weights to list
        weights_json = json.dumps(weights_as_list)  # convert the weights to json format
        weights_bytes = weights_json.encode()  # convert the weights to bytes
        return Message(payload=weights_bytes, code=CONTENT)

    # handle get requests from multiple clients
    async def handle_get_requests(self):
        tasks = []
        for i in range(self.num_clients):  # create a task for each client
            tasks.append(
                self.render_get(None))  # the request is None because the request is not needed in render_get function
        await asyncio.gather(*tasks)  # run all the tasks concurrently


# main function where the server is created and run federated learning algorithm for 10 rounds only if there are at least 2 clients
async def main():
    server = Server()

    # create a server context
    root = Site()
    root.add_resource(['model'], server)  # add the server to the root context
    context = await Context.create_server_context(root, bind=('127.0.0.1', 5683))
    print('Server is running!')

    while True:
        await asyncio.sleep(1)
        if (len(server.client_weights) + server.counter_loss) == server.clients:
            await server.federated_averaging()  # run federated averaging algorithm
            server.send_weights.set()  # set the event to send the weights to the clients
            await server.handle_get_requests()  # handle get requests from multiple clients
            server.client_weights.clear()  # clear the list of clients' weights
            server.send_weights.clear()  # clear the event
            server.num_clients = 0  # reset the number of clients connected to the server
            server.counter_loss = 0


asyncio.run(main())  # run the main function
