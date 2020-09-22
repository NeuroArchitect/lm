import mesh_tensorflow as mtf

from pydantic import BaseModel

class FeedForwardConfig(BaseModel):
    activation: str
    initializer: str
    io_dim: int
    hidden_dim: int
    use_bias:bool

class FeedForward:
    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.config = config

    def __call__(self, x):
        io_dim = self.config.io_dim
        hidden_dim = self.config.hidden_dim
        activation = self.config.activation
        initializer = self.config.initializer
        use_bias = self.config.use_bias

        intermediate = mtf.layers.dense(
            x, 
            reduced_dims=[io_dim],
            new_dims=[hidden_dim],
            activation=activation,
            kernel_initializer=initializer,
            name="dense_1", 
            use_bias=use_bias)
        return mtf.layers.dense(
            intermediate,
            reduced_dims=[hidden_dim],
            new_dims=[io_dim],
            kernel_initializer=initializer,
            name="dense_2", use_bias=use_bias)