#!/bin/bash

source .env
USER_NAME=$(id -un)

ssh ${USER_NAME}@localhost -p ${PORT_SSH}