#!/bin/bash
PYTHONPATH=$(pwd) 
export PYTHONPATH
python uusic/models/train_embedding.py
