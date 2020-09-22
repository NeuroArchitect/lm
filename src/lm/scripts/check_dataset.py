import random

import tensorflow as tf
import tensorflow.compat.v1 as v1
from absl import app, logging
from absl.flags import argparse_flags

import lm.config
import lm.encoders
import lm.examples
import lm.human


def parse_args(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="Path to where your files are located. Files ending in .zst are treated as \
                        archives, all others as raw text.",
    )
    parser.add_argument(
        "--encoder", type=str, required=True, help="Name or path of a tokenizer spec"
    )
    parser.add_argument(
        "--sample_size", type=int, default=10, help="the size of samples to inspect"
    )
    args = parser.parse_args(argv[1:])
    return args


def main(args):
    encfg = lm.config.load(args.encoder)
    tokenizer = lm.encoders.from_config(encfg)
    with tf.Session() as sess:
        files = lm.human.filepaths_from_user_input(args.input)
        if len(files) == 0:
            logging.error("no file found at %s", args.input)
            return

        sampled_files = random.choices(files, k=args.sample_size)

        ds = tf.data.Dataset.from_tensor_slices(sampled_files)
        ds = ds.interleave(lm.examples.from_file_list(sampled_files), cycle_length=4)
        ds = ds.map(lm.examples.read_example)
        ds = ds.shuffle(32)
        ds = ds.take(args.sample_size)

        it = v1.data.make_one_shot_iterator(ds)
        example = it.get_next()

        from tokenizers.decoders import ByteLevel
        blvl = ByteLevel()

        while True:
            try:
                result = sess.run(example)  # , max_id_tf, min_id_tf])
                pt = lm.examples.PreProcessedTextLine(
                    id=result["id"],
                    content=result["content"],
                    tokens=result["tokens"],
                    offsets_start=result["offsets_start"],
                    offsets_end=result["offsets_end"],
                )

                ids = tokenizer.decode(result["tokens"])
                vocab = tokenizer.get_vocab()
                inv_vocab = { v:k for k,v in vocab.items() }
                
                logging.info("content:      %r", pt.content)
                logging.info("tokens:       %r", pt.tokens.tolist())
                logging.info("gold text:    %r", pt.content.decode("utf-8"))
                logging.info("inv_vocab:    %r", [ blvl.decode(inv_vocab[v]) for v in pt.tokens])
                logging.info("decoded:      %r", ids)
                logging.info(
                    "tokenization: %s",
                    [
                        pt.content.decode("utf-8")[slice(int(start), int(end))]
                        for start, end in zip(pt.offsets_start, pt.offsets_end)
                    ],
                )
                logging.info("-" * 10)
            except tf.errors.OutOfRangeError:
                break


def apprun():
    app.run(main, flags_parser=parse_args)


if __name__ == "__main__":
    apprun()
