# lm: The Language Model Pipeline

[![image](https://img.shields.io/pypi/v/lm.svg)](https://pypi.python.org/pypi/lm)
<a href="https://github.com/NeuroArchitect/lm/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/NeuroArchitect/lm.svg?color=blue">
</a>
[![Actions Status](https://github.com/NeuroArchitect/lm/workflows/build/badge.svg)](https://github.com/NeuroArchitect/lm/actions)


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
- Tensorflow (1.15.2+)
- Tensorflow Mesh 
- TPUs (maybe GPUs cluster in the future, maybe)
- Docker / Kubeflow setup

# Useful Commands

## TL;DR

```bash
# install the library and create gpt2 encoded binaries
pip3 install lm
export INPUT=/tmp/some/path/
export OUTPUT=/tmp/some/path/
lm hashsort ${INPUT} /tmp/lm.index.uniq.txt
lm encode /tpm/lm.index.uniq.txt  ${OUTPUT}
```

## lm encode

Turns text files into gpt2 encoded `.tfrecords`.

```bash
!mkdir -p /tmp/datasets/tfrecords/

ENCODE_INPUT=/tmp/datasets/txt
ENCODE_OUTPUT=/tmp/datasets/tfrecords/
NAME=my_dataset

# short
lm encode ${ENCODE_INPUT} ${ENCODE_OUTPUT} 

# expanded 
lm encode \
    --name $NAME \
    --encoder gpt2 \
    --size 200MiB \
    --nproc 0 \
    --compress zlib \
    ${ENCODE_INPUT} \
    ${ENCODE_OUTPUT} 
```

Add 
`--size 300` to add 300MiB (300 * 2 * 20) of uncompressed input text into each tfrecord file. 

Set `--nproc 1` to disable multiprocessing (useful for debugging).

# License
-   Free software: [Apache Software License 2.0](LICENSE)

# Credits
This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.


# Sponsor the Project

<style>.bmc-button img{height: 34px !important;width: 35px !important;margin-bottom: 1px !important;box-shadow: none !important;border: none !important;vertical-align: middle !important;}.bmc-button{padding: 7px 15px 7px 10px !important;line-height: 35px !important;height:51px !important;text-decoration: none !important;display:inline-flex !important;color:#ffffff !important;background-color:#000000 !important;border-radius: 8px !important;border: 1px solid transparent !important;font-size: 18px !important;letter-spacing:0.6px !important;box-shadow: 0px 1px 2px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;margin: 0 auto !important;font-family:'Arial', cursive !important;-webkit-box-sizing: border-box !important;box-sizing: border-box !important;}.bmc-button:hover, .bmc-button:active, .bmc-button:focus {-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;text-decoration: none !important;box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;opacity: 0.85 !important;color:#ffffff !important;}</style><link href="https://fonts.googleapis.com/css?family=Arial" rel="stylesheet"><a class="bmc-button" target="_blank" href="https://www.buymeacoffee.com/fabrizio">☕<span style="margin-left:5px;font-size:18px !important;">Buy me a triple espresso</span></a>
<br/>

