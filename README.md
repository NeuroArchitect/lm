---
title: Language Model (lm) End to End Pipeline 
---

[![image](https://img.shields.io/pypi/v/lm.svg)](https://pypi.python.org/pypi/lm)

[![image](https://img.shields.io/travis/NeuroArchitect/lm.svg)](https://travis-ci.com/NeuroArchitect/lm)

[![Documentation Status](https://readthedocs.org/projects/lm/badge/?version=latest)](https://lm.readthedocs.io/en/latest/?badge=latest)

[![Updates](https://pyup.io/repos/github/NeuroArchitect/lm/shield.svg)](https://pyup.io/repos/github/NeuroArchitect/lm/)




End to End Language Model Pipeline built for training speed

There are few frameworks out there that focus on sequence to sequence neural network models.
Most notables are the ones built by [Google](github.com/tensorflow/seq2seq) and [Facebook](github.com/pytorch/fairseq).
This repository focuses on seq2seq and language model (next token prediction) using an opinionated end to end setup.
The project objective is to create a *production* pipeline that runs end to end and contains all the professional steps required to achieve state of the art language models.

It leverages:
- mesh tensorflow to train on 8, 32, 256, 512 TPUs
- jsonnet configuration files
- docker/kubeflow for orchestrating the various experiments
- absl for process management, flags, unittest

It uses and supports *ONLY*: 
- Tensorflow (1.15)
- Tensorflow Mesh 
- TPUs (maybe GPUs cluster in the future, maybe)
- Docker / Kubeflow setup

# Useful Commands
# TLDR

```
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e . 
lm cleantxt 
lm encode 

## lm encode

Turns text files into encoded .tfrecords.

```bash
!mkdir -p /tmp/datasets/tfrecords/

ENCODE_INPUT=/tmp/datasets/txt
ENCODE_OUTPUT=/tmp/datasets/tfrecords/
NAME=my_dataset

# short
lm encode \
    ${ENCODE_INPUT}/\* \
    ${ENCODE_OUTPUT} 

# expanded 
lm encode \
    --name $NAME \
    --encoder gpt2 \
    --size 200MiB \
    --nproc 0 \
    ${ENCODE_INPUT}/\* \
    ${ENCODE_OUTPUT} 
```

Add e.g. `--size 300` to add 300MiB (300 * 2 * 20) of uncompressed input text into
each tfrecord file. 

Set `--nproc 1` to disable multiprocessing. Useful for debugging via
pdb.


# Sponsor
- contact to sponsor this project

# License
-   Free software: Apache Software License 2.0

# Credits
This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
