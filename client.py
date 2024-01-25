# federated learning client
import tensorflow as tf
from aiocoap import Context, Message, GET, POST
import asyncio


# send the model to the server
async def send_model(server_ip, model):
    protocol = await Context.create_client_context()  # create a client context

    # save the weights of the model
    with open('model.h5', 'wb') as f:
        model.save_weights(f)

    # send the model to the server
    with open('model.h5', 'rb') as f:
        payload = f.read()
        request = Message(code=POST, payload=payload)
        request.set_request_uri(server_ip)
        response = await protocol.request(request).response
        print('Result: %s\n%r' % (response.code, response.payload))


# receive updated weights from the server
async def receive_model(server_ip, model):
    protocol = await Context.create_client_context()  # create a client context

    # receive the updated weights from the server
    request = Message(code=GET)
    request.set_request_uri(server_ip)
    response = await protocol.request(request).response
    print('Result: %s\n%r' % (response.code, response.payload))

    # load the updated weights
    with open('model.h5', 'wb') as f:
        f.write(response.payload)
    model.load_weights('model.h5')


async def main():
    server = 'coap://localhost:5683/model'  # server address

    # load the data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize the data

    # definition of GRU model
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(128, input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, epochs=1, batch_size=32)

    # send the model to the server
    await send_model(server, model)

    # receive the updated weights from the server
    await receive_model(server, model)


asyncio.get_event_loop().run_until_complete(main())
