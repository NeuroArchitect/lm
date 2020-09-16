#!/usr/bin/env bash 
set -e
set -x 

rm -fr /tmp/tfrecord

# create unique list of files
FILES_TO_INDEX_INPUT=${PWD}/data/uds
INDEX=/tmp/index.txt

# Index the dataset
lm index ${FILES_TO_INDEX_INPUT} ${INDEX} --force

# Hash sort to unique files
HASHSORT_OUTPUT=/tmp/hashsorted
lm hashsort --nproc 1 ${INDEX} ${HASHSORT_OUTPUT} 

# clean the text files
CLEANTXT_OUTPUT=/tmp/cleantxt
lm cleantxt ${HASHSORT_OUTPUT} ${CLEANTXT_OUTPUT} --force

# train tokenizer on the clean dataset
TOKENIZER_INPUT=${CLEANTXT_OUTPUT}
TOKENIZER_OUTPUT=/tmp/tokenizer/
lm_train_tokenizer --vocab_size 10 --input ${TOKENIZER_INPUT} --output ${TOKENIZER_OUTPUT}

# converts to tfrecord 
ENCODE_INPUT=${CLEANTXT_OUTPUT}
ENCODE_OUTPUT=/tmp/tfrecord
lm encode --encoder ${TOKENIZER_OUTPUT} ${ENCODE_INPUT} ${ENCODE_OUTPUT} --nproc 1

ENCODE_INPUT=${CLEANTXT_OUTPUT}
ENCODE_OUTPUT=/tmp/tfrecord/parallel
lm encode --encoder ${TOKENIZER_OUTPUT} ${ENCODE_INPUT}/\*.\* ${ENCODE_OUTPUT}

# check output of plain text
lm_check_dataset ${ENCODE_OUTPUT}/\*.tfrecord --encoder ${TOKENIZER_OUTPUT}

# check output of gzip text
ENCODE_INPUT=${CLEANTXT_OUTPUT}
ENCODE_OUTPUT=/tmp/tfrecord/gzip

lm encode --encoder ${TOKENIZER_OUTPUT} ${ENCODE_INPUT} ${ENCODE_OUTPUT} --compress gz

lm_check_dataset ${ENCODE_OUTPUT} --encoder ${TOKENIZER_OUTPUT}
