#!/bin/bash

VOLUME="${PWD}:/app"
ENV="PORT=8899"
PORT="8899:8899"
IMAGE="pablorr10/bert-toxic"

# Start a docker container 
docker run --rm -dit -v ${PWD}:/app --name bash-remote ${IMAGE} bash
docker run --rm -dit -v ${PWD}:/app --name bash-remote --runtime nvidia ${IMAGE} bash

# Start a jupyter notebook
docker run --rm -dit \
    -v ${PWD}:/app \
    -e ${ENV} \
    -p ${PORT} \
    --name jupyter-remote \
    --runtime nvidia \
    ${IMAGE} jupyter notebook --ip='0.0.0.0' --port=8899 --allow-root --no-browser --notebook-dir=/app


# docker ps --format "table {{.ID}} \t {{.Names}} \t {{.Ports}}"