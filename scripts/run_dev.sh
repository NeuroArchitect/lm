#!/usr/bin/env bash
set -x 
docker run --rm -it -v $PWD:/lm -v $HOME/Library/Caches/pip:/root/.cache/pip/ --entrypoint /bin/bash lm:dev 