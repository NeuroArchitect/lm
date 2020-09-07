{
    RandomSumGenerator(examples_to_gen=1000, 
                       len_sequence=256, 
                       seed=1337,
                       vocab_size=256) :: {
        kind: 'datasets.RandomSumGenerator',
        seed: seed,
        len_sequence: len_sequence,
        examples_count: examples_to_gen,
        vocab_size: vocab_size,
        special_tokens : {
            PAD: 0,
            EOS: 1,
            BOS: 2,
        } 
    },
    
    Seq2SeqTFRecordDataset(location="", n_samples=0, vocab_size=0, context_length=0) :: {
        kind: 'datasets.Seq2SeqTFRecordDataset',
        len_sequence: len_sequence,
        n_samples: n_samples,
        vocab_size: vocab_size,
    },

}