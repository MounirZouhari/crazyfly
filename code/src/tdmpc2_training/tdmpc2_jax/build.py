"""Build an untrained TD-MPC2 agent from a Hydra config.

Training and evaluation both call this, so a checkpoint is always restored
into exactly the network shapes it was saved from.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from tdmpc2_jax.common.activations import mish
from tdmpc2_jax.networks import NormedLinear
from tdmpc2_jax.tdmpc2 import TDMPC2
from tdmpc2_jax.world_model import WorldModel


def build_agent(cfg, observation, action_dim, model_key, encoder_key):
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = dict(cfg['tdmpc2'])
  dtype = jnp.dtype(model_config.dtype)

  encoder_module = nn.Sequential(
      [
          NormedLinear(
              encoder_config.encoder_dim, activation=mish, dtype=dtype
          )
          for _ in range(encoder_config.num_encoder_layers-1)
      ] + [
          NormedLinear(
              model_config.latent_dim, activation=None, dtype=dtype
          )
      ]
  )

  if encoder_config.tabulate:
    print("Encoder")
    print("--------------")
    print(
        encoder_module.tabulate(
            jax.random.key(0), observation, compute_flops=True
        )
    )

  encoder = TrainState.create(
      apply_fn=encoder_module.apply,
      params=encoder_module.init(encoder_key, observation)['params'],
      tx=optax.chain(
          optax.zero_nans(),
          optax.clip_by_global_norm(model_config.max_grad_norm),
          optax.adam(encoder_config.learning_rate),
      )
  )

  model = WorldModel.create(
      action_dim=int(action_dim),
      encoder=encoder,
      **model_config,
      key=model_key
  )
  if model.action_dim >= 20:
    tdmpc_config['mppi_iterations'] += 2

  return TDMPC2.create(world_model=model, **tdmpc_config)
