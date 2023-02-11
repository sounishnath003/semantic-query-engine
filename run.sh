#!/bin/bash/python

rm -fr ./**/__pycache__
clear
black .
python -m src.main