/* 
Infeed library. Supported infeeds:
- ExampleGenerator
- TFRecordDataset: feeds the training process using a set of files written in the tfrecord format
*/
{
    // ExampleGenerator(dataset, seed=1337, batch_size=32) :: {
    //     // An infeed that Generates from a tf.Example producer
    //     kind: 'lm.infeeds.ExampleGenerator',
    //     dataset: dataset,
    //     batch_size: batch_size,
    //     max_sequence_length: dataset.max_sequence_length
    // },

    TFRecordDatasetReader(sources, batch_size=1, compression_type=null)::{
        kind: 'lm.infeeds.TFRecordDatasetReader',
        batch_size: batch_size,
        compression_type: if compression_type == null then 'none' else compression_type,
        sources: sources,
    },
}