import lm
import re
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from pydantic import BaseModel
from enum import Enum

# adapted from mesh_tensorflow
class Optimizer(object):
    """Base optimizer class.

    Constructor of subclasses must take `learning_rate` as an argument.
    """

    def apply_grads(self, grads, variables, learning_rate):
        """
        Apply gradients to variables.

        Call this function externally instead of apply_grad().  This causes the
        operations to be combined, which is necessary for stacking variables
        see mtf.rewrite_stack_variables().

        Args:
            grads: a list of Tensor
            variables: a list of Variables
            learning_rate: the learning rate value to apply to the update
        Returns:
            a list of Operations
        """
        ops = []
        for grad, var in zip(grads, variables):
            ops.extend(self.apply_grad(grad, var, learning_rate))
        if not ops:
            return ops
        return variables[0].graph.combine_assignments(ops)

    def apply_grad(self, grad, var, learning_rate):
        """Update variable and accumulators.

        Args:
            grad: a Tensor
            var: a Variablle
        Returns:
            a list of Operations
        """
        raise ValueError("apply_grad not implemented %s %s" % (grad, var))


class AdamWeightDecayOptimizerConfig(BaseModel):
    name: str = "adam"
    beta1_adam: float = 0.9
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-09
    ada_epsilon1: float = 1e-30
    ada_epsilon2: float = 0.001
    weight_decay_rate: float = 0.0
    exclude_from_weight_decay: bool = False


@lm.register_optimizer("lm.optimizers.Adam", AdamWeightDecayOptimizerConfig)
class AdamWeightDecayOptimizer(Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self, config: AdamWeightDecayOptimizerConfig):
        """Constructs a AdamWeightDecayOptimizer."""
        self.weight_decay_rate = config.weight_decay_rate
        self.beta_1 = config.beta_1
        self.beta_2 = config.beta_2
        self.epsilon = config.epsilon
        self.exclude_from_weight_decay = config.exclude_from_weight_decay
        self.config = config

    def apply_grad(self, grad, var, learning_rate):
        """See base class."""
        if grad is None:
            tf.logging.warning("Gradient is None for variable %s" % var.name)
            return []

        grad = mtf.to_float(grad)

        assignments = []

        m = mtf.get_variable(
            var.mesh,
            var.name + "/adam_m",
            var.shape,
            initializer=tf.zeros_initializer(),
            trainable=False,
        )

        v = mtf.get_variable(
            var.mesh,
            var.name + "/adam_v",
            var.shape,
            initializer=tf.zeros_initializer(),
            trainable=False,
        )

        # Standard Adam update.
        next_m = self.beta_1 * m + (1.0 - self.beta_1) * grad
        next_v = self.beta_2 * v + (1.0 - self.beta_2) * mtf.square(grad)

        update = next_m / (mtf.sqrt(next_v) + self.epsilon)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        if self._do_use_weight_decay(var.name):
            update += mtf.to_float(var.value) * self.weight_decay_rate

        update_with_lr = learning_rate * update

        var_update = mtf.assign_sub(var, update_with_lr)

        assignments.extend([var_update, mtf.assign(m, next_m), mtf.assign(v, next_v)])
        return assignments

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


def clip_by_global_norm(grads, clip_norm):
    """Clip the grads by global norm."""
    global_norm = mtf.sqrt(
        mtf.add_n([mtf.reduce_sum(mtf.square(t)) for t in grads if t is not None])
    )
    multiplier = clip_norm / mtf.maximum(global_norm, clip_norm)
    clipped_grads = [None if t is None else t * multiplier for t in grads]
    return clipped_grads, global_norm

class LearningRateDecayEnum(str, Enum):
    linear: str = 'linear'
    cosine: str = 'cosine'

class LearningRateScheduleConfig(BaseModel):
    learning_rate: float
    warmup_steps: int
    decay: LearningRateDecayEnum = LearningRateDecayEnum.linear
    initial_decrease_percent: float = 0.1
    power: float = 1.0
    cycle: bool = False

@lm.register_component('lm.optimizers.LearningRateSchedule', LearningRateScheduleConfig)
class LearningRateSchedule:
    def __init__(self, config:LearningRateScheduleConfig):
        super().__init__()
        self.config = config

    def __call__(self, total_steps):
        global_step = tf.train.get_or_create_global_step()  # get global step
        learning_rate = tf.constant(
            value=self.config.learning_rate, shape=[], dtype=tf.float32
        )  # grab lr param

        if self.config.decay == "linear":
            learning_rate = tf.train.polynomial_decay(
                learning_rate,
                global_step,
                total_steps,
                end_learning_rate=self.config.learning_rate * self.config.initial_decrease_percent,  # decrease to 10% of initial LR according to GPT-3 paper
                power=1.0,
                cycle=self.config.cycle,
            )
        elif self.config.decay == "cosine":
            learning_rate = tf.train.cosine_decay(
                learning_rate,
                global_step,
                total_steps,
                alpha=0.1,  # alpha is min lr value as a fraction of init lr.
            )

        if self.config.warmup_steps > 0:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(self.config.warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = learning_rate * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                1.0 - is_warmup
            ) * learning_rate + is_warmup * warmup_learning_rate

        return learning_rate

def get_optimizer(loss, params, summary, inp_var_grads=None):
    """Creates and returns an optimizer training op."""

    global_step = tf.train.get_or_create_global_step()  # get global step
    mesh = loss.mesh  # get mesh info from loss
    graph = mesh.graph  # get graph info from mesh

    if inp_var_grads is None:
        var_grads = mtf.gradients(
            [loss], [v.outputs[0] for v in graph.trainable_variables]
        )
    else:
        var_grads = inp_var_grads

    learning_rate = tf.constant(
        value=params["lr"], shape=[], dtype=tf.float32
    )  # grab lr param

    if params["lr_decay"] == "linear":
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            params["train_steps"],
            end_learning_rate=params["lr"]
            * 0.1,  # decrease to 10% of initial LR according to GPT-3 paper
            power=1.0,
            cycle=False,
        )
    elif params["lr_decay"] == "cosine":
        learning_rate = tf.train.cosine_decay(
            learning_rate,
            global_step,
            params["train_steps"],
            alpha=0.1,  # alpha is min lr value as a fraction of init lr.
        )

    # if params["warmup_steps"] > 0:
    #     global_steps_int = tf.cast(global_step, tf.int32)
    #     warmup_steps_int = tf.constant(params["warmup_steps"], dtype=tf.int32)

    #     global_steps_float = tf.cast(global_steps_int, tf.float32)
    #     warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    #     warmup_percent_done = global_steps_float / warmup_steps_float
    #     warmup_learning_rate = learning_rate * warmup_percent_done

    #     is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    #     learning_rate = (
    #         1.0 - is_warmup
    #     ) * learning_rate + is_warmup * warmup_learning_rate

    summary.scalar("lr", learning_rate)

    if params["opt_name"].lower() == "adam":
        optimizer = mtf.optimize.AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=params["weight_decay"],
            beta_1=params["beta1"],
            beta_2=params["beta2"],
            epsilon=params["epsilon"],
            exclude_from_weight_decay=["norm", "bias"],
        )
    else:
        optimizer = mtf.optimize.AdafactorOptimizer(
            learning_rate=params["lr"],
            decay_rate=params["weight_decay"],
            beta1=params["beta1"],
            epsilon1=params["ada_epsilon1"],
            epsilon2=params["ada_epsilon2"],
        )

    if params["gradient_clipping"] is not None:
        clip_value = mtf.constant(mesh, params["gradient_clipping"], dtype=tf.float32)
        (var_grads, _) = clip_by_global_norm(var_grads, clip_norm=clip_value)

    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)
    return learning_rate, update_ops
