import collections
import re
from typing import Any

import mesh_tensorflow as mtf
import tensorflow as tf
import tensorflow.compat.v1 as v1
from pydantic.dataclasses import dataclass
from tensorflow.python.tpu import tpu_estimator

import lm
from lm.builders import nn

from absl import logging

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


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        if "global_step" in name or "adam_" in name or "slot_" in name:
            continue
        name_to_variable[name] = var

    logging.info("init_checkpoint:{} ".format(init_checkpoint))
    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


@dataclass
class ToyTransformerConfig:
    mesh_shape: Any
    mesh_layout: Any
    len_sequence: int
    n_channels: int
    n_tokens: int
    n_layers: int
    # learning_rate: float
    # optimizer: str
    # num_hidden_layers: int
    use_bias: bool
    initializer_range: float = 0.1  # stddev
    channels_dropout_prob: float = 0.1
    layer_output_dropout_prob: float = 0.1
    activation_function: str = "gelu"


@lm.register_model("ToyTransformer", ToyTransformerConfig)
class ToyTransformer:
    def __init__(self, config: ToyTransformerConfig):
        super().__init__()
        self.config = config
        self.__is_training = False

    @property
    def mesh_shape(self):
        return self.config.mesh_shape

    @property
    def mesh_layout(self):
        return self.config.mesh_layout

    @property
    def learning_rate(self) -> float:
        return self.config.learning_rate

    def __call__(self, features, labels, params, mode):  # this is the model_fn
        """A model is called by TpuEstimator."""
        x_tf = features
        y_tf = labels
        
        global_step = v1.train.get_or_create_global_step()
        assert global_step is not None

        # Graph setup
        graph = mtf.Graph()
        mesh_shape = mtf.convert_to_shape(self.mesh_shape)
        layout_rules = mtf.convert_to_layout_rules(self.mesh_layout)
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

            # mesh_impl will be used for lowering
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

        # Compose Model
        n_tokens_dim = mtf.Dimension("n_tokens", params["n_tokens"])
        # n_channels_dim = mtf.Dimension('n_channels', params['n_channels'])
        with mtf.utils.outside_all_rewrites():
            logits, y = self.model(mesh, x_tf, y_tf, params, mode)
       
        if mode == tf.estimator.ModeKeys.PREDICT:
            # inputs = mtf.anonymize(features)
            logits = mtf.anonymize(logits)
            lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=True)
            # inputs = lowering.export_to_tf_tensor(inputs)
            outputs = lowering.export_to_tf_tensor(logits)
            predictions = {
                "inputs": x_tf,
                "outputs": outputs}
            
            scaffold_fn = self.restore_from_checkpoint(
                lowering,
                params["checkpoint_dir"], params["use_tpu"]
            )
            
            return tpu_estimator.TPUEstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                scaffold_fn=scaffold_fn,
                prediction_hooks=[mtf.MtfRestoreHook(lowering)])
        # configure optimizer
        elif mode == tf.estimator.ModeKeys.TRAIN:
            # compute loss
            predictions, loss = self.loss(logits, y, n_tokens_dim)
            var_grads = mtf.gradients(
                [loss], [v.value for v in graph.trainable_variables]
            )
            optimizer = lm.get_optimizer(params['optimizer'])
            learning_rate_fn = lm.get_component(params['learning_rate'])
            learning_rate = learning_rate_fn(params['total_steps'])
            update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables, learning_rate)
            
            # covert back to tensorflow format
            lowering = mtf.Lowering(graph, {mesh: mesh_impl})

            predictions_tf = lowering.export_to_tf_tensor(predictions)
            loss_tf = tf.cast(lowering.export_to_tf_tensor(loss), tf.float32)
 
            # compute gradients
            tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
            # update the global step
            tf_update_ops.append(v1.assign_add(global_step, 1))
            logging.info("tf_update_ops: %s", tf_update_ops)
            train_op = tf.group(tf_update_ops)
        elif mode == tf.estimator.ModeKeys.EVAL:
            logits_tf = lowering.export_to_tf_tensor(fully_replicated_logits)
        else:
            raise ValueError('wrong PREDICT logic path')

        scaffold_fn = self.restore_from_checkpoint(
                lowering, 
                params["checkpoint_dir"], params["use_tpu"]
            )

        # create estimator
        with mtf.utils.outside_all_rewrites():
            # Copy master variables to slices. Must be called first.
            restore_hook = mtf.MtfRestoreHook(lowering)
            if mode == tf.estimator.ModeKeys.TRAIN:
                saver = tf.train.Saver(
                    tf.global_variables(),
                    sharded=True,
                    max_to_keep=10,  # TODO this is a config
                    keep_checkpoint_every_n_hours=2,
                    defer_build=False,
                    save_relative_paths=True,
                )
                tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

                # checkpoint save hook
                saver_listener = mtf.MtfCheckpointSaverListener(lowering)
                saver_hook = tf.estimator.CheckpointSaverHook(
                    checkpoint_dir=params["checkpoint_dir"],
                    save_secs=None,
                    save_steps=params["steps_per_checkpoint"],
                    saver=saver,
                    checkpoint_basename='model.ckpt',
                    scaffold=None,
                    listeners=[saver_listener],
                )

                return tpu_estimator.TPUEstimatorSpec(
                    tf.estimator.ModeKeys.TRAIN,
                    loss=loss_tf,
                    train_op=train_op,
                    training_hooks=[restore_hook, saver_hook],
                    scaffold_fn=scaffold_fn,
                    export_outputs={'predictions': predictions_tf} 
                )
            elif mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(logits_tf):
                    mean_logits = tf.metrics.mean(logits_tf)
                    return {"mean_logits": mean_logits}

                eval_metrics = (metric_fn, [logits_tf])
                return (
                    tpu_estimator.TPUEstimatorSpec(
                        tf.estimator.ModeKeys.EVAL,
                        evaluation_hooks=[restore_hook],
                        loss=loss_tf,
                        eval_metrics=eval_metrics,
                        scaffold_fn=scaffold_fn,
                    ),
                    lowering,
                )  # FIXME
            else:
                raise ValueError("wrong PREDICT logic path")

    @property
    def dense_initializer(self):
        if self.config.initializer_range:
            return tf.truncated_normal_initializer(stddev=self.config.initializer_range)
        else:
            return mtf.layers.VarianceScalingInitializer(scale=0.4)

    @property
    def embedding_initializer(self):
        initializer = self.dense_initializer
        if isinstance(initializer, mtf.layers.DenseInitializer):
            # embedding matrix is also used as classifier weight matrix.
            # scale it appropriately.
            return initializer(reduced_dims=[self.model_dim], new_dims=[self.vocab_dim])
        else:
            return initializer

    @property
    def num_hidden_layers(self):
        return self.config.n_layers

    def normalize(self, x, reduce_dim):
        return nn.layer_norm(
            x,
            reduce_dim,
            subtract_mean=self.config.use_bias,
            use_bias=self.config.use_bias,
        )

    def channels_to_token_projection(
        self,
        mesh,
        x,
        batch_dim,
        sequence_dim,
        n_channels_dim,
        n_tokens_dim,
        variable_dtype,
    ):
        # Perform embedding lookup on the word ids.
        w = mtf.get_variable(
            mesh,
            "final_projection",
            mtf.Shape([n_channels_dim, n_tokens_dim]),
            initializer=self.embedding_initializer,
            master_dtype=variable_dtype.master_dtype,
            slice_dtype=variable_dtype.slice_dtype,
            activation_dtype=variable_dtype.activation_dtype,
        )
        # from input to mtf
        x = mtf.einsum([x, w], output_shape=[batch_dim, sequence_dim, n_tokens_dim])

        # x = self.normalize(x, n_tokens_dim)
        # x = self.regularize(x, n_tokens_dim)
        return x

    def restore_from_checkpoint(self, lowering, init_checkpoint_path, use_tpu: bool):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint_path is None:
            return scaffold_fn

        check_point_state = tf.train.get_checkpoint_state(init_checkpoint_path)

        if check_point_state is None:
            return scaffold_fn

        (
            assignment_map,
            initialized_variable_names,
        ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint_path)

        if use_tpu:

            def tpu_scaffold():
                return tf.train.Scaffold(
                    local_init_op=tf.group(
                        tf.train.Scaffold.default_local_init_op(),
                        lowering.copy_masters_to_slices(),
                        name="mtf_local_init_op"),
                    ready_op=tf.concat(
                        [tf.report_uninitialized_variables(),
                        resources.report_uninitialized_resources()],
                        axis=0,
                        name="mtf_ready_op"))

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint_path, assignment_map)

        # update name from checkpoint
        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                logging.info(
                    "  name = %s, shape = %s%s", var.name, var.shape, init_string
                )

        return scaffold_fn

    def model(self, mesh, x, y, params, mode):
        """
        The input variables and labels are still tensorflow variables.
        This function should take care of converting between mesh tensorflow 
        and tensorflow
        """
        # x :: [batch, io, vocab]
        # assert x.dtype in (tf.int32,), "only integer sequences are supported"
        # if y:
        #     assert x.shape == y.shape, "input shape must match %r != %r" % (x.shape, y.shape)
        #     assert x.shape == (params['batch_size'], params['len_sequence']), "input shape must be known"
        
        if params["precision"] == "bfloat16":
            dtype = tf.bfloat16
        else:
            dtype = tf.float32

        # master has always dtype float32, slices can have bfloat16 or float32
        variable_dtype = mtf.VariableDType(tf.float32, dtype, dtype)

        # Build the actual model
        if mode == tf.estimator.ModeKeys.PREDICT:
            batch_size = params['predict_batch_size']
        else:
            batch_size = params['batch_size']

        # allocate dimensions 
        batch_dim = mtf.Dimension("batch", batch_size)
        n_tokens_dim = mtf.Dimension("n_tokens", params["n_tokens"])
        sequence_dim = mtf.Dimension("sequence", params["len_sequence"])
        n_channels_dim = mtf.Dimension("channel", params["n_channels"])

        # from input to mtf
        x = mtf.import_tf_tensor(mesh, x, mtf.Shape([batch_dim, sequence_dim]))
        if y is not None:
            y = mtf.import_tf_tensor(mesh, y, mtf.Shape([batch_dim, sequence_dim]))
            y = mtf.cast(y, variable_dtype.activation_dtype)

        # Embeddings
        with v1.variable_scope("toy"):
            with v1.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                embedding_table = mtf.get_variable(
                    mesh,
                    "token_embeddings",
                    mtf.Shape([n_tokens_dim, n_channels_dim]),
                    initializer=self.embedding_initializer,
                    master_dtype=variable_dtype.master_dtype,
                    slice_dtype=variable_dtype.slice_dtype,
                    activation_dtype=variable_dtype.activation_dtype,
                )
                # project x into the embedding table to fetch the embeddings
                # for each input token in the sequence
                # >[batch, seq, n_tokens] x [n_tokens, n_channels] :
                # =[batch, seq, n_channels]
                index = x  # use the token ids as index in the lookup operation
                tok_embedding_output = mtf.gather(
                    embedding_table,
                    index,
                    dim=n_tokens_dim,  # output_shape=channels_dim
                )

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                embedding_output = tok_embedding_output

                # normalize
                embedding_output = self.normalize(
                    embedding_output, reduce_dim=n_channels_dim
                )
                # regularize
                embedding_output = mtf.dropout(
                    embedding_output,
                    keep_prob=1.0 - self.config.layer_output_dropout_prob,
                )

            pos_embedding = mtf.get_variable(
                mesh,
                "pos_embeddings",
                mtf.Shape([sequence_dim, n_channels_dim]),
                initializer=self.embedding_initializer,
            )

            # shift token by pos embeddings
            x = embedding_output + pos_embedding
            x = mtf.cast(x, variable_dtype.activation_dtype)

            # Add transformers blocks
            # h: is for hidden. this are the channels being passed down the pipeline
            h = x
            print("channels:", h.shape)
            dim = n_channels_dim
            for lnum in range(1, self.num_hidden_layers + 2):
                if lnum + 1 == self.num_hidden_layers + 2:
                    # output layer
                    dim = n_channels_dim
                else:
                    h = mtf.layers.dense(
                        h,
                        dim,
                        use_bias=False,
                        master_dtype=variable_dtype.master_dtype,
                        slice_dtype=variable_dtype.slice_dtype,
                        name="layer_%d" % lnum,
                    )
            last_layer = h

            # last temp layer
            logits = self.channels_to_token_projection(
                mesh,
                last_layer,
                batch_dim,
                sequence_dim,
                n_channels_dim,
                n_tokens_dim,
                variable_dtype,
            )

            logits = mtf.identity(logits, "logits")
            print("logits", logits.shape)
            return logits, y

    def train_mode(self):
        # TODO add context
        self.__is_training = True

    @property
    def is_training(self):
        return self.__is_training

    def loss(self, logits, y, reduce_dim):
        with tf.variable_scope("loss"):
            if self.is_training:
                # I.e., 0.1 dropout. does it make sense it here?
                logits = mtf.dropout(logits, keep_prob=0.9)
            # project back to token dimensions
            prediction = mtf.softmax(logits, reduce_dim, name="prediction")
            # compute the mean square loss between the input and the output
            loss = mtf.reduce_mean(mtf.square(y - prediction))
        return prediction, loss


# Original Toy model from mesh tensorflow
@dataclass
class ToyConfig:
    io_size: int
    master_dtype: int
    slice_dtype: int
    activation_dtype: int
    num_hidden_layers: int


@lm.register_model("toy")
class ClassifyToyModel:
    def __init__(self, config: ToyConfig):
        self.config = config

    @property
    def io_size(self):
        return self.config.io_size

    @property
    def master_dtype(self):
        return self.config.master_dtype

    @property
    def slice_dtype(self):
        return self.config.slice_dtype

    @property
    def activation_dtype(self):
        return self.config.activation_dtype

    @property
    def num_hidden_layers(self):
        return self.config.num_hidden_layers

    @property
    def hidden_size(self):
        return self.config.hidden_size

    def __call__(self, mesh, features, params):
        """A toy model implemented by mesh tensorlfow."""
        batch_size = params["batch_size"]

        batch_dim = mtf.Dimension("batch", batch_size)
        io_dim = mtf.Dimension("io", self.io_size)

        master_dtype = tf.as_dtype(self.master_dtype)
        slice_dtype = tf.as_dtype(self.slice_dtype)
        activation_dtype = tf.as_dtype(self.activation_dtype)

        x = mtf.import_tf_tensor(mesh, features, mtf.Shape([batch_dim, io_dim]))
        x = mtf.cast(x, activation_dtype)
        h = x
        for lnum in range(1, self.num_hidden_layers + 2):
            if lnum + 1 == self.num_hidden_layers + 2:
                # output layer
                dim = io_dim
            elif lnum % 2 == 0:
                dim = mtf.Dimension("hidden_even", self.hidden_size)
            else:
                dim = mtf.Dimension("hidden_odd", self.hidden_size)

            h = mtf.layers.dense(
                h,
                dim,
                use_bias=False,
                master_dtype=master_dtype,
                slice_dtype=slice_dtype,
                name="layer_%d" % lnum,
            )

        y = h

        loss = mtf.reduce_mean(mtf.square(y - x))
        return y, loss
