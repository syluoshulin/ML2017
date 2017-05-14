#!/bin/bash
python gen.py
python predict.py $1 $2
