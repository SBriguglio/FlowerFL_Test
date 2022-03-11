#!/bin/bash

# start the server first
gnome-terminal -- sh -c 'echo "Server Started" && python server.py && bash'

# now start the client(s)
for client in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  gnome-terminal -- sh -c 'echo "Client $client" && python client.py && bash'
done