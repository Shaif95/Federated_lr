# Federated_lr

##Federated Learning framework that improves communication efficiency. 

#Requirements : 

flower==1.0.0           # Federated learning framework
tensorflow==2.10.0      # Deep learning framework
numpy==1.23.5           # For numerical operations

##Running Federated Learning Code

#Requirements

Ensure you have Python and all necessary libraries installed as per the requirements.txt. You can install them using:

pip install -r requirements.txt

#Server and Clients Setup

To run the federated learning setup with one server and multiple clients:

## Server Setup: Create a server.py script that initializes and starts the federated learning server.
## Client Setup: Create a client.py script that defines and starts the federated learning clients.

#Using train.bat to Start the System

##Execute the train.bat file to start the server and multiple client instances automatically:

bash
train.bat

This script will open one command prompt for the server and five command prompts for the clients. Make sure the flr virtual environment is set up correctly in your Python environment.


