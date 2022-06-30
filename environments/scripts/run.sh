#!/bin/bash

source .env

docker run --gpus all \
           -it \
           -d \
           --rm \
           -h $HOST_NAME \
           -v $PWD:$PWD \
           -v $PWD/environments/etc/ssh:/etc/ssh \
           -v $PWD/environments/etc/home/.ssh/:/home/$(id -un)/.ssh \
           -p 127.0.0.1:$PORT_SSH:22 \
           -p 127.0.0.1:$PORT_NOTEBOOK:8888 \
           --name $CONTAINER_NAME \
           $IMAGE_TAG