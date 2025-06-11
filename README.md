# TTS API and Real-time Server

This project includes a Text-to-Speech (TTS) API server, a real-time server, and a real-time client. Follow the instructions below to set up and run the servers.

## Prerequisites

Before you start, ensure you have `Python 3.x` installed. You can verify your Python installation by running:

```bash
python --version
````

## Install Requirements

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

This will install all required libraries for the API server, real-time server, and client.

## Run API Server

To start the TTS API server, follow these steps:

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Start the API server:

```bash
python f5_tts_server.py
```

The API server will be running and ready to accept requests.

## Run Real-time Server

To start the real-time server, run the following command:

```bash
python socket_server.py
```

The server will start and handle real-time communication.

## Run Real-time Client

To start the real-time client, run:

```bash
python main.py
```

The client will connect to the real-time server and interact with it.

## Notes

* Ensure your environment variables are correctly set in the `.env` file.
* If you encounter issues, refer to the error logs for debugging information.
