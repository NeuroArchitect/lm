#!/usr/bin/env bash 
set -e

rm -fr /tmp/tfrecord

# create unique list of files
CLEANTXT_INPUT=${PWD}/data/uds
CLEANTXT_OUTPUT=/tmp/index.txt

find ${CLEANTXT_INPUT} -type f -exec readlink -f {} \; > /tmp/index.txt

lm hashsort /tmp/index.txt ${CLEANTXT_OUTPUT} --nproc 1
exit

# clean utf8
CLEANTXT_INPUT=data/uds
CLEANTXT_OUTPUT=/tmp/cleantxt
lm cleantxt ${CLEANTXT_INPUT} ${CLEANTXT_OUTPUT} --force

# train encoder
TOKENIZER_INPUT=${CLEANTXT_OUTPUT}
TOKENIZER_OUTPUT=/tmp/tokenizer/
lm_train_tokenizer --vocab_size 1010 --input ${TOKENIZER_INPUT} --output ${TOKENIZER_OUTPUT}

# converts to tfrecord 
ENCODE_INPUT=${CLEANTXT_OUTPUT}
ENCODE_OUTPUT=/tmp/tfrecord
lm encode --encoder ${TOKENIZER_OUTPUT} ${ENCODE_INPUT} ${ENCODE_OUTPUT} --nproc 1

ENCODE_INPUT=${CLEANTXT_OUTPUT}
ENCODE_OUTPUT=/tmp/tfrecord/parallel
lm encode --encoder ${TOKENIZER_OUTPUT} ${ENCODE_INPUT}/\*.\* ${ENCODE_OUTPUT}


# check output
lm_check_dataset ${ENCODE_OUTPUT}/\*.tfrecord --encoder ${TOKENIZER_OUTPUT}

echo "\n\n\n\test gzip\n\n\n"
ENCODE_INPUT=${CLEANTXT_OUTPUT}
ENCODE_OUTPUT=/tmp/tfrecord/gzip

lm encode --encoder ${TOKENIZER_OUTPUT} ${ENCODE_INPUT} ${ENCODE_OUTPUT} --compress gz

# check output
lm_check_dataset ${ENCODE_OUTPUT} --encoder ${TOKENIZER_OUTPUT}


