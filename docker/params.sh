#!/bin/bash
cd $(dirname $0)

NAME="argus-quick-draw"
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
NET="--net=host"
IPC="--ipc=host"

VOLUMES="-v $(pwd)/..:/workdir"
