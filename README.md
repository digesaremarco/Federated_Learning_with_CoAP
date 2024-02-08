Federated Learning for Intrusion Detection over CoAP Networks
============================================================
Overview
--------
This project is a proof of concept for a federated learning system for intrusion detection over CoAP networks. The system is designed to be lightweight and efficient, and is intended to be used in IoT environments.
Federated learning is a machine learning technique that allows for training a model across multiple devices or servers holding local data samples, without exchanging them. The model is trained locally on each device, and only the model updates are sent to a central server, where they are aggregated and used to update the global model. This allows for training a model on a large number of devices without the need to send the data to a central server, which is particularly useful in IoT environments, where data privacy and communication costs are important concerns.
The system is designed to be used in CoAP networks, which are widely used in IoT environments. CoAP is a lightweight protocol designed for constrained devices and networks, and is intended to be used in environments where HTTP is too heavy.
The project is implemented in Python, uses the TensorFlow library for machine learning, and the aiocoap library for CoAP communication.
Contains the following files:
- `client.py`: CoAP client that simulates an IoT device and sends data to the server.
- `server.py`: CoAP server that receives data from the clients and build the global model.
- `plot.py`: used to plot the results.
- `requirements.txt`: contains the required libraries to run the project.

