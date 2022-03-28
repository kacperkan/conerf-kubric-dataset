#!/bin/bash

docker run \
    --rm \
    --interactive \
    --user $(id -u):$(id -g) \
    --volume "$(pwd):/kubric" \
    kubricdockerhub/kubruntu \
    /usr/bin/python3 trio.py 