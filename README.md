Federated Learning for Intrusion Detection over CoAP Networks
============================================================
Overview
--------
This project is a proof of concept for a federated learning system for intrusion detection over CoAP networks. The system is designed to be lightweight and efficient, and is intended to be used in IoT environments.
Federated learning is a machine learning technique that allows for training a model across multiple devices or servers holding local data samples, without exchanging them. 
The model is trained locally on each device, and only the model updates are sent to a central server, where they are aggregated and used to update the global model. 
This allows for training a model on a large number of devices without the need to send the data to a central server, which is particularly useful in IoT environments, where data privacy and communication costs are important concerns.
The system is designed to be used in CoAP networks, which are widely used in IoT environments. CoAP is a lightweight protocol designed for constrained devices and networks, and is intended to be used in environments where HTTP is too heavy.
The project is implemented in Python, uses the TensorFlow library for machine learning, and the aiocoap library for CoAP communication.
Contains the following files:
- `client.py`: CoAP client that simulates an IoT device and sends data to the server.
- `server.py`: CoAP server that receives data from the clients and build the global model.
- `plot.py`: used to plot the results.
- `requirements.txt`: contains the required libraries to run the project.

Requirements
------------
- Python 3.6 or higher
- TensorFlow 2.0 or higher
- aiocoap 0.4.2 or higher
- numpy 1.17.4 or higher
- matplotlib 3.1.2 or higher
- pandas 0.25.3 or higher
- scikit-learn 0.22.1 or higher

Usage
-----
1. Install the required libraries using the following command:
   ```
   pip install -r requirements.txt
   ```
2. Run the server using the following command:
   ```
    python server.py
    ```
3. Run the clients using the following command:
    ```
     python client.py
     ```
The file client.py simulate multiple clients sending data to the server for 10 rounds, including packet loss to simulate unreliable networks. 
The server will build the global model averaging the local models of the clients.
At the end of the simulation, the file client.py plot the results using the file plot.py.
For this simulation I used the CICIDS2023 dataset, which contains network traffic data for intrusion detection. The dataset is available at https://www.unb.ca/cic/datasets/iotdataset-2023.html.
Each client uses a different portion of the dataset to simulate different devices with different data samples.
The simulation uses a simple neural network with one hidden layer and 10 neurons, and is trained using the Adam optimizer and the binary cross-entropy loss function.
To evaluate the model I used: accuracy, precision, recall and f1-score. The results are plotted at the end of the simulation.