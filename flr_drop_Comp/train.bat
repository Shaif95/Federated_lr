@echo off

REM Start a command prompt for the server
start cmd /k "activate federated && python server.py"

REM Start five command prompts for the clients
start cmd /k "activate federated && python client.py"
start cmd /k "activate federated && python client.py"
start cmd /k "activate federated && python client.py"
start cmd /k "activate federated && python client.py"
start cmd /k "activate federated && python client.py"
start cmd /k "activate federated && python client.py"
start cmd /k "activate federated && python client.py"
start cmd /k "activate federated && python client.py"
start cmd /k "activate federated && python client.py"
start cmd /k "activate federated && python client.py"

