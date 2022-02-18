# Quickstart example from: https://flower.dev/docs/quickstart_tensorflow.html
import flwr as fl

# Simply workloads can be automatically configured by Flower. For this example only one line is needed
fl.server.start_server(config={"num_rounds": 3})