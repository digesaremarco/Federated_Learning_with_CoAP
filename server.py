# federated learning server
import asyncio
from aiocoap.resource import Site, Resource
from aiocoap import Context, Message, GET, PUT
import tensorflow as tf
import numpy as np
import math

# definition of GRU model
global_model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(128, input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the model
global_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])


# receive model from client
async def receive_model(request):
    payload = request.payload
    with open('model.h5', 'wb') as f:
        f.write(payload)
    global_model.load_weights('model.h5')
    return Message(code=PUT, payload=payload)


# send model to client
async def send_model(request):
    with open('model.h5', 'wb') as f:
        global_model.save_weights(f)
    with open('model.h5', 'rb') as f:
        payload = f.read()
    return Message(code=PUT, payload=payload)


# updated the global model with the average of the clients' models
async def federated_averaging(request):
    payload = request.payload
    with open('model.h5', 'wb') as f:
        f.write(payload)
    client_model = tf.keras.models.load_model('model.h5')
    client_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    client_weights = client_model.get_weights()
    global_weights = global_model.get_weights()
    for i in range(len(client_weights)):
        global_weights[i] = (global_weights[i] + client_weights[i]) / numclients
    global_model.set_weights(global_weights)
    with open('model.h5', 'wb') as f:
        global_model.save_weights(f)
    with open('model.h5', 'rb') as f:
        payload = f.read()
    return Message(code=PUT, payload=payload)


# main function where the server is created and run federated learning algorithm for 10 rounds only if there are at least 2 clients
async def main():
    global numclients  # number of clients
    numclients = 0
    server = 'coap://127.0.0.1:5683/model'  # server address
    root = Site()
    root.add_resource(['model'], Resource())  # add the resource to the site
    await Context.create_server_context(bind=('127.0.0.1', 5683), site=root)
    print("Server federated learning avviato.")

    for i in range(10):
        request = await root.resource.render_post(None)
        if request.code == PUT:
            numclients += 1
            if numclients >= 2:  # run federated learning algorithm only if there are at least 2 clients
                await federated_averaging(request)
            else:
                await receive_model(request)
        elif request.code == GET:
            await send_model(request)


asyncio.get_event_loop().run_until_complete(main())
