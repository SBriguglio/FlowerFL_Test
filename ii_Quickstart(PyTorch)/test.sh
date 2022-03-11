#!/bin/bash

# start the server first
gnome-terminal -- sh -c 'echo "Server Started" && python server.py && bash'

# now start the client(s)
for client in 1 2 3; do
  gnome-terminal -- sh -c 'echo "Client $client" && python client.py && bash'
done