#!/bin/sh

# adapted from: https://spin.atomicobject.com/2015/11/30/command-line-tools-docker/
# lm-cli-app
# A wrapper script for invoking lm-cli-app with docker

PROGNAME="$(basename $0)"
VERSION="v0.0.1"

# Helper functions for guards
error(){
  error_code=$1
  echo "ERROR: $2" >&2
  echo "($PROGNAME wrapper version: $VERSION, error code: $error_code )" &>2
  exit $1
}
check_cmd_in_path(){
  cmd=$1
  which $cmd > /dev/null 2>&1 || error 1 "$cmd not found!"
}

# Guards (checks for dependencies)
check_cmd_in_path docker
docker system info > /dev/null 2>&1 || error 2 "no active docker found."

# Set up mounted volumes, environment, and run our containerized command
exec docker run \
  --interactive --tty --rm \
  --volume "$PWD":/wd \
  --workdir /wd \
  "nlpz/lm:$VERSION" "$@"
