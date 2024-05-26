@echo off

REM Start the server in a new command prompt with the conda environment activated
start cmd /k "conda activate federated && python server.py"

REM Add a short delay to ensure the server starts first
timeout /t 5 /nobreak >nul

REM Start five command prompts for the clients with the conda environment activated
start cmd /k "conda activate federated && python client.py"
start cmd /k "conda activate federated && python client.py"
