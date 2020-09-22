{
    // GPT2(vocab_size=128) :: {
    //     kind: "neogpt.models.GPT2",
    //     n_ctx: 8,
    //     n_embd: 8,
    //     n_head: 8,
    //     n_vocab: vocab_size,
    //     n_layer: 2,
    //     scale_by_depth: false,
    //     scale_by_in: false,
    //     mesh_shape: "batch:1",
    //     layout: "batch:1",
    //     activation_function: "gelu",
    //     attention_types: [
    //         [["global"], self.n_layer],
    //     ],
    //     auto_layout: false,
    //     auto_layout_and_mesh_shape: false,
    //     stop_at_token: 2,
    //     remove_partial_sequences: true,
    //     scalenorm: true,
    //     no_weight_tie: false,
    //     regularization: {
    //         embed_dropout: 0.1,  // embedding dropout
    //         attn_dropout: 0.1,
    //         res_dropout:0.1,
    //     },
    // },

    ToyTransformer(len_sequence, n_tokens, n_channels=8, mesh_shape='all:1', layout='batch:all', n_layers=1) :: {
        kind: "lm.models.ToyTransformer",
        len_sequence: len_sequence,
        n_tokens: n_tokens,
        n_layers: n_layers,
        n_channels: 8,
        mesh_shape: mesh_shape,
        mesh_layout: layout,
        activation_function: "gelu",
        use_bias: true,
        // scalenorm: true,
        // no_weight_tie: false,
        // regularization: {
        //     embed_dropout: 0.1,  // embedding dropout
        //     attn_dropout: 0.1,
        //     res_dropout:0.1,
        // },
    },

    Transfomer(n_tokens, n_ctx, n_heads, n_layers=8) :: {
        kind="lm.models.StackedTransfomer",
        layers=[
            PositionalEmbeddings(n_tokens, n_ctx),
        ] + [ MultiHeadTransfomer(n_ctx, n_io, n_heads, name="layer_"+l) for l in std.range(n_layers) ]
    }
}