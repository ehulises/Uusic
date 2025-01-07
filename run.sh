#!/bin/bash

# run any file passed into the first arg
# but set the PYTHONPATH to pwd

PYTHONPATH=$(pwd)
export PYTHONPATH

# $1 is the python file to run
# ${@:2} passes all arguments except the first
python "$1" "${@:2}"
