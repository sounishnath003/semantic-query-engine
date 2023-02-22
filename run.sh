#!/bin/bash/python

rm -fr ./**/__pycache__
clear
black .
TOKENIZERS_PARALLELISM=false
echo TOKENIZERS_PARALLELISM $TOKENIZERS_PARALLELISM
python -m src.main