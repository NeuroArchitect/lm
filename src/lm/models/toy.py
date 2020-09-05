import mesh_tensorflow as mtf
import tensorflow as tf
from tensorflow.python.tpu import tpu_estimator

from . import nn


class Attention:
    def __init__(self):
        super().__init__()

    def __call__(self, mesh):

        io_size = 8
        n_heads = 4
        io_dim = mtf.Dimension("io", [io_size])
        kv_dim = mtf.Dimension("kv", [io_size])
        heads_dim = mtf.Dimension("heads", [n_heads])

        params = mtf.attention.AttentionParams(
            mesh,
            query_input_dim=io_dim,
            memory_input_dim=io_dim,
            output_dim=io_dim,
            key_dim=kv_dim,
            value_dim=kv_dim,
            query_heads_dims=[heads_dim],
            memory_heads_dims=[heads_dim],
            variable_dtype=variable_dtype,
        )


class ToyTransformerConfig:
    mesh_shape: str
    mesh_layout: str
    learning_rate: float
    optimizer: str
    num_hidden_layers: int
    use_bias: bool


class ToyTransformer:
    def __init__(self, config: ToyTransformerConfig):
        super().__init__()
        self.config = config

    @property
    def mesh_shape(self):
        return self.config.mesh_shape

    @property
    def mesh_layout(self):
        return self.config.layout

    @property
    def learning_rate(self) -> float:
        return self.config.learning_rate

    @property
    def optimizer(self) -> str:
        return self.config.optimizer

    def __call__(self, features, labels, mode, params):  # this is the model_fn

        """A model is called by TpuEstimator."""
        del labels
        global_step = tf.train.get_global_step()

        # Graph setup
        graph = mtf.Graph()
        mesh_shape = mtf.convert_to_shape(self.mesh_shape)
        layout_rules = mtf.convert_to_layout_rules(self.layout)
        if params["use_tpu"]:
            ctx = params["context"]
            num_hosts = ctx.num_hosts
            host_placement_fn = ctx.tpu_host_placement_function
            device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
            # Worker 0 caches all the TPU binaries.
            replica_cache_size = 300 * 1024 * 1024  # 300M per replica.
            worker0_mem = replica_cache_size * 8 * num_hosts
            devices_memory_usage = [worker0_mem] + [0] * (num_hosts - 1)
            var_placer = mtf.utils.BalancedVariablePlacer(
                device_list, devices_memory_usage
            )
            mesh = mtf.Mesh(graph, "my_mesh", var_placer)
            mesh_devices = [""] * mesh_shape.size

            mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
                mesh_shape, layout_rules, mesh_devices, devices_memory_usage
            )
        else:
            var_placer = None
            mesh_devices = [""] * mesh_shape.size
            mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
                mesh_shape, layout_rules, mesh_devices
            )

            mesh = mtf.Mesh(graph, "my_mesh", var_placer)

        # RUN Model
        with mtf.utils.outside_all_rewrites():
            logits, loss = self.model(mesh, features, params)

        # TRAIN mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            var_grads = mtf.gradients(
                [loss], [v.outputs[0] for v in graph.trainable_variables]
            )
            if self.optimizer == "Adafactor":
                optimizer = mtf.optimize.AdafactorOptimizer()
            else:
                assert self.optimizer == "SGD"
                optimizer = mtf.optimize.SgdOptimizer(learning_rate=self.learning_rate)
                update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)
        else:
            # for now, we can only export fully-replicated tensors.
            fully_replicated_logits = mtf.anonymize(logits)

        # covert back to tensorflow format
        lowering = mtf.Lowering(graph, {mesh: mesh_impl})
        tf_loss = tf.to_float(lowering.export_to_tf_tensor(loss))
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
            tf_update_ops.append(tf.assign_add(global_step, 1))
            tf.logging.info("tf_update_ops: {}".format(tf_update_ops))
            train_op = tf.group(tf_update_ops)
        else:
            tf_logits = lowering.export_to_tf_tensor(fully_replicated_logits)

        # create estimator
        with mtf.utils.outside_all_rewrites():
            # Copy master variables to slices. Must be called first.
            restore_hook = mtf.MtfRestoreHook(lowering)
            if mode == tf.estimator.ModeKeys.TRAIN:
                saver = tf.train.Saver(
                    tf.global_variables(),
                    sharded=True,
                    max_to_keep=10,
                    keep_checkpoint_every_n_hours=2,
                    defer_build=False,
                    save_relative_paths=True,
                )
                tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
                saver_listener = mtf.MtfCheckpointSaverListener(lowering)
                saver_hook = tf.train.CheckpointSaverHook(
                    self.model_dir,
                    save_steps=1000,
                    saver=saver,
                    listeners=[saver_listener],
                )

                return tpu_estimator.TPUEstimatorSpec(
                    tf.estimator.ModeKeys.TRAIN,
                    loss=tf_loss,
                    train_op=train_op,
                    training_hooks=[restore_hook, saver_hook],
                )
            elif mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(tf_logits):
                    mean_logits = tf.metrics.mean(tf_logits)
                    return {"mean_logits": mean_logits}

                eval_metrics = (metric_fn, [tf_logits])
                return tpu_estimator.TPUEstimatorSpec(
                    tf.estimator.ModeKeys.EVAL,
                    evaluation_hooks=[restore_hook],
                    loss=tf_loss,
                    eval_metrics=eval_metrics,
                )
            elif mode == tf.estimator.ModeKeys.PREDICT:
                return tpu_estimator.TPUEstimatorSpec(
                    tf.estimator.ModeKeys.PREDICT,
                    evaluation_hooks=[restore_hook],
                    loss=None,
                    eval_metrics=eval_metrics,
                )

        @property
        def dense_initializer(self):
            if self.config.initializer_range:
                return tf.truncated_normal_initializer(
                    stddev=self.config.initializer_range
                )
            else:
                return mtf.layers.VarianceScalingInitializer(scale=0.4)

        @property
        def embedding_initializer(self):
            initializer = self.dense_initializer
            if isinstance(initializer, mtf.layers.DenseInitializer):
                # embedding matrix is also used as classifier weight matrix.
                # scale it appropriately.
                return initializer(
                    reduced_dims=[self.model_dim], new_dims=[self.vocab_dim]
                )
            else:
                return initializer

        @property
        def num_hidden_layers(self):
            return self.config.num_hidden_layers

        def normalize(self, x, reduce_dim):
            return nn.layer_norm(
                x,
                reduce_dim,
                subtract_mean=self.config.use_bias,
                use_bias=self.config.use_bias,
            )

        def model(self, mesh, x, y, params):
            # x :: [batch, io, vocab]

            if params["precision"] == "bfloat16":
                dtype = tf.bfloat16
                # master has type float32, slice and activation have type bfloat16
                variable_dtype = mtf.VariableDType(tf.float32, tf.bfloat16, tf.bfloat16)
            else:
                dtype = tf.float32
                # master, slice and activate have all float16
                variable_dtype = mtf.VariableDType(tf.float32, tf.float32, tf.float32)

            # Build the actual model
            batch_dim = mtf.Dimension("batch", params["batch_size"])
            vocab_dim = mtf.Dimension("vocab", params["vocab_size"])
            io_dim = mtf.Dimension("sequence", params["io"])
            io_chan_dim = mtf.Dimension("io", params["io_channels"])

            # from input to mtf
            x = mtf.import_tf_tensor(mesh, x, mtf.Shape([batch_dim, io_dim, vocab_dim]))

            # Embeddings
            with tf.variable_scope(scope="toy", default_name="seq2seq"):
                with tf.variable_scope("embeddings"):
                    # Perform embedding lookup on the word ids.
                    embedding_table = mtf.get_variable(
                        mesh,
                        "word_embeddings",
                        mtf.Shape([vocab_dim, io_chan_dim]),
                        initializer=self.embedding_initializer,
                    )

                    word_embedding_output = mtf.gather(
                        embedding_table, x, dim=vocab_dim, output_shape=io_chan_dim
                    )

                    # Add positional embeddings and token type embeddings, then layer
                    # normalize and perform dropout.
                    embedding_output = word_embedding_output

                    pos_embedding = mtf.get_variable(
                        mesh,
                        "pos_embeddings",
                        mtf.Shape([io_dim, io_chan_dim]),
                        initializer=self.embedding_initializer,
                    )
                    embedding_output = self.normalize(embedding_output)
                    embedding_output = mtf.dropout(
                        embedding_output,
                        keep_prob=1.0 - self.config.layer_output_dropout_prob,
                    )

                # shift token by pos embeddings
                x = word_embedding_output + pos_embedding
                x = mtf.cast(x, variable_dtype.activation_dtype)

                h = x
                for lnum in range(1, self.num_hidden_layers + 2):
                    if lnum + 1 == self.num_hidden_layers + 2:
                        # output layer
                        dim = io_dim
                    elif lnum % 2 == 0:
                        dim = mtf.Dimension("hidden_even", io_chan_dim)
                    else:
                        dim = mtf.Dimension("hidden_odd", io_chan_dim)
                        h = mtf.layers.dense(
                            h,
                            dim,
                            use_bias=False,
                            master_dtype=variable_dtype.master_dtype,
                            slice_dtype=variable_dtype.slice_dtype,
                            name="layer_%d" % lnum,
                        )

                prediction = h
                # project back to token dimensions

                # compute the mean quare loss between the input and the output
                loss = mtf.reduce_mean(mtf.square(y - prediction))
                return prediction, loss
