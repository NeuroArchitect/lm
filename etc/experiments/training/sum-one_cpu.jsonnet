local optimizers = import 'optimizers.libsonnet';
local models = import 'models.libsonnet';
local infeeds = import 'infeeds.libsonnet';

local lr_schedule(learning_rate=0.0001, decay='cosine', warmup_steps=0) = {
   kind: 'lm.optimizers.LearningRateSchedule',
   learning_rate: learning_rate,
   decay: decay,
   warmup_steps: warmup_steps,
};

local Schedule() = {
   steps: self.steps_per_iteration * 10,                // total number of steps to run
   steps_per_iteration: 1000,  // how many steps to loop on-device
   steps_per_checkpoint: self.steps_per_iteration, // save a checkpoint after this num of steps
};

local Experiment() = {
   infeed: infeeds.TFRecordDatasetReader(
      sources=std.extVar("dataset"),
      batch_size=32,
   ) + {
      len_sequence: 8,
      n_tokens: 18,
   },
   model: models.ToyTransformer(
      len_sequence=self.infeed.len_sequence,
      n_tokens=self.infeed.n_tokens,
      n_layers=5,
   ),
   model_path: "/tmp/checkpoints/",
   schedule: Schedule(),
   trainer: {
      device: {},
      optimizer: optimizers.Adam(),
      learning_rate: lr_schedule(),
      weight_decay: 0.1,
      gradient_clipping: 0.5,
   },
};

Experiment() // main configuration