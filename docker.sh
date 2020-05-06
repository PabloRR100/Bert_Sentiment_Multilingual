#!/bin/bash

MODE="--rm -dit --runtime nvidia"
VOLUME="${PWD}:/app"
ENV="PORT=8899"
PORT="8899:8899"

docker run --rm -dit -v ${PWD}:/app --name bash-remote -e PORT=8899 -p 8899:8899 --runtime nvidia pablorr10/harvard:latest bash

docker run --rm -dit -v ${PWD}:/app --name jupyter-remote -e PORT=8899 -p 8999:8999 --runtime nvidia pablorr10/harvard:latest jupyter notebook --ip='0.0.0.0' --port=8899 --allow-root --no-browser --notebook-dir=/app

docker ps --format "table {{.ID}} \t {{.Names}} \t {{.Ports}}"