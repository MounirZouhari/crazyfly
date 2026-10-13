import os
import time
from collections import defaultdict
from functools import partial

import gymnasium as gym
import hydra
import jax
import numpy as np
import orbax.checkpoint as ocp
import tqdm
from tensorboardX import SummaryWriter

from tdmpc2_jax.build import build_agent
from tdmpc2_jax.data import SequentialReplayBuffer

@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
  env_config = cfg['env']

  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

  ##############################
  # Environment setup
  ##############################
  def make_env(env_config, seed):
    def make_gym_env(env_id, seed):
      env = gym.make(env_id)
      env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env

    if env_config.backend == "gymnasium":
      return make_gym_env(env_config.env_id, seed)
    elif env_config.backend == "dmc":
      from tdmpc2_jax.envs.dmcontrol import make_dmc_env
      env = make_dmc_env(env_config.env_id, seed, env_config.dmc.obs_type)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env
    else:
      raise ValueError("Environment not supported:", env_config)

  if env_config.asynchronous:
    vector_env_cls = gym.vector.AsyncVectorEnv
  else:
    vector_env_cls = gym.vector.SyncVectorEnv
  env = vector_env_cls(
      [
          partial(make_env, env_config, seed)
          for seed in range(cfg.seed, cfg.seed+env_config.num_envs)
      ]
  )
  np.random.seed(cfg.seed)
  rng = jax.random.PRNGKey(cfg.seed)

  rng, model_key, encoder_key = jax.random.split(rng, 3)

  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs, _ = env.reset()
  dummy_action = env.action_space.sample()
  dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = env.step(
      dummy_action
  )
  replay_buffer = SequentialReplayBuffer(
      capacity=cfg.buffer_size,
      vectorized=True,
      num_envs=env_config.num_envs,
      seed=cfg.seed,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_term,
          truncated=dummy_trunc
      )
  )

  ##############################
  # Agent setup
  ##############################
  agent = build_agent(
      cfg,
      observation=dummy_obs,
      action_dim=np.prod(env.single_action_space.shape),
      model_key=model_key,
      encoder_key=encoder_key,
  )
  global_step = 0

  # Hover test flights during training (learning curve). Imported here because
  # the rest of this trainer is not specific to the drone.
  tester = None
  if cfg.get('eval_interval_steps'):
    from hover_eval import FlightTester, append_learning_curve
    tester = FlightTester(flight_steps=cfg.eval_flight_steps)
    curve_path = os.path.join(output_dir, 'learning_curve.csv')
    test_starts = ['home'] * int(cfg.eval_flights)

  def record_learning_curve(env_steps, train_seconds):
    flights = tester.fly(agent, test_starts, seed=cfg.seed + 1000)
    rows = append_learning_curve(
        curve_path, flights,
        env_steps=env_steps, train_seconds=round(train_seconds, 1)
    )
    late_error = np.mean([r['late_error_m'] for r in rows])
    writer.add_scalar('test/late_error_m', late_error, env_steps)
    writer.add_scalar('test/return', np.mean([r['return'] for r in rows]), env_steps)
    tqdm.tqdm.write(f'Test flights at {env_steps} steps: late error {late_error:.3f} m')

  options = ocp.CheckpointManagerOptions(
      max_to_keep=1, save_interval_steps=cfg['save_interval_steps']
  )
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  with ocp.CheckpointManager(
      checkpoint_path,
      options=options,
      item_names=('agent', 'global_step', 'buffer_state')
  ) as mngr:
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      abstract_buffer_state = jax.tree.map(
          ocp.utils.to_shape_dtype_struct, replay_buffer.get_state()
      )
      restored = mngr.restore(
          mngr.latest_step(),
          args=ocp.args.Composite(
              agent=ocp.args.StandardRestore(agent),
              global_step=ocp.args.JsonRestore(),
              buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
          )
      )
      agent, global_step = restored.agent, restored.global_step
      replay_buffer.restore(restored.buffer_state)
    else:
      print('No checkpoint folder found, starting from scratch')
      mngr.save(
          global_step,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(global_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    ##############################
    # Training loop
    ##############################
    ep_count = np.zeros(env_config.num_envs, dtype=int)
    prev_logged_step = global_step
    plan = None
    observation, _ = env.reset(seed=cfg.seed)

    seed_steps = int(cfg.get('seed_steps', 2500 * env_config.num_envs * env_config.utd_ratio))
    pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
    done = np.zeros(env_config.num_envs, dtype=bool)
    # Wall-clock time spent training, with the test flights taken out.
    loop_start = time.perf_counter()
    test_seconds = 0.0
    for global_step in range(global_step, cfg.max_steps, env_config.num_envs):
      if global_step <= seed_steps:
        action = env.action_space.sample()
      else:
        rng, action_key = jax.random.split(rng)
        action, plan = agent.act(
            observation,
            prev_plan=plan,
            deterministic=False,
            train=True,
            key=action_key
        )
        action = np.array(action)

      next_observation, reward, terminated, truncated, info = env.step(action)

      if np.any(~done):
        replay_buffer.insert(
            dict(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated
            ),
            mask=~done
        )
      observation = next_observation

      # Handle terminations/truncations
      done = np.logical_or(terminated, truncated)
      if np.any(done):
        if plan is not None:
          plan = (
              plan[0].at[done].set(0),
              plan[1].at[done].set(agent.max_plan_std)
          )
        for ienv in range(env_config.num_envs):
          if done[ienv]:
            r = info['episode']['r'][ienv]
            l = info['episode']['l'][ienv]
            print(
                f"Episode {ep_count[ienv]}: r = {r:.2f}, l = {l}"
            )
            writer.add_scalar('episode/return', r, global_step + ienv)
            writer.add_scalar('episode/length', l, global_step + ienv)
            ep_count[ienv] += 1

      if global_step >= seed_steps:
        if global_step == seed_steps:
          print('Pre-training on seed data...')
          num_updates = int(cfg.get('seed_updates', seed_steps))
        else:
          num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))

        rng, *update_keys = jax.random.split(rng, num_updates+1)
        log_this_step = global_step >= prev_logged_step + \
            cfg['log_interval_steps']
        if log_this_step:
          all_train_info = defaultdict(list)
          prev_logged_step = global_step

        for iupdate in range(num_updates):
          batch = replay_buffer.sample(agent.batch_size, agent.horizon)
          agent, train_info = agent.update(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminated=batch['terminated'],
              truncated=batch['truncated'],
              key=update_keys[iupdate]
          )

          if log_this_step:
            for k, v in train_info.items():
              all_train_info[k].append(np.array(v))

        if log_this_step:
          for k, v in all_train_info.items():
            writer.add_scalar(f'train/{k}_mean', np.mean(v), global_step)
            writer.add_scalar(f'train/{k}_std', np.std(v), global_step)

        mngr.save(
            global_step,
            args=ocp.args.Composite(
                agent=ocp.args.StandardSave(agent),
                global_step=ocp.args.JsonSave(global_step),
                buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
            ),
        )

      pbar.update(env_config.num_envs)

      env_steps = global_step + env_config.num_envs
      if tester is not None and env_steps % cfg.eval_interval_steps < env_config.num_envs:
        test_start = time.perf_counter()
        record_learning_curve(env_steps, test_start - loop_start - test_seconds)
        test_seconds += time.perf_counter() - test_start
    pbar.close()
    # The upstream interval can exceed a short experiment. Always keep the
    # final learned state so smoke runs can be resumed and inspected too.
    final_step = global_step + env_config.num_envs
    if final_step > 0 and final_step != mngr.latest_step():
      mngr.save(
          final_step, force=True,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(final_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    # Make sure the learning curve ends at the final agent.
    if tester is not None and final_step % cfg.eval_interval_steps >= env_config.num_envs:
      test_start = time.perf_counter()
      record_learning_curve(final_step, test_start - loop_start - test_seconds)
    print(f'Training took {(time.perf_counter() - loop_start) / 60:.1f} min. '
          f'Full test flights: eval_hover.py {output_dir}')
    writer.close()


if __name__ == '__main__':
  train()
