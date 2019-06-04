# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from acnetwork import AdvActorCriticNetwork

from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES
#from pysc2.lib.actions import TYPES as ACTION_TYPES
from pre_processing import is_spatial_action, stack_ndarray_dicts, Preprocessor

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import ACTION_SIZE

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
  def __init__(self,
               sess,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               network_data_format,
               value_loss_weight,
               entropy_weight,
               learning_rate,
               max_to_keep,
               envs,
               res,
               train=True,
               n_steps=8,
               discount=0.99):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step
    self.local_network = AdvActorCriticNetwork(
        sess=sess,
        action_size=ACTION_SIZE,
        thread_index=thread_index,
        device=device,
        network_data_format=network_data_format,
        value_loss_weight=value_loss_weight,
        entropy_weight=entropy_weight,
        learning_rate=learning_rate,
        max_to_keep=max_to_keep)
    self.envs = envs
    self.n_steps = n_steps
    self.discount = discount
    self.train = train

    self.episode_counter = 0
    self.cumulative_score = 0.0

    #self.preproc = Preprocessor(self.envs.observation_spec())
    #self.envs.observation_spec()
    self.preproc = Preprocessor(self.envs.observation_spec()[0])

    #self.local_network.prepare_loss(ENTROPY_BETA)
    static_shape_channels = self.preproc.get_input_channels()
    self.local_network.build(static_shape_channels, resolution=res)

    with tf.device(device):
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(
        self.local_network.loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients)
      
    self.sync = self.local_network.sync_from(global_network)

    self.reset()
    
    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0
    # variable controling log output
    self.prev_local_t = 0

  def reset(self):
    #for env in self.envs:
    obs_raw = self.envs.reset()
    self.last_obs = self.preproc.preprocess_obs(obs_raw)

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def _summarize_episode(self, timestep):
    score = timestep.observation["score_cumulative"][0]
    #if self.summary_writer is not None:
    #  summary = tf.Summary()
    #  summary.value.add(tag='sc2/episode_score', simple_value=score)
    #  self.summary_writer.add_summary(summary, self.episode_counter)

    print("episode %d: score = %f" % (self.episode_counter, score))
    self.episode_counter += 1
    return score

  def run_batch(self, train_summary=False):
    """Collect trajectories for a single batch and train (if self.train).

    Args:
      train_summary: return a Summary of the training step (losses, etc.).

    Returns:
      result: None (if not self.train) or the return value of agent.train.
    """
    shapes = (self.n_steps, self.envs.n_envs)
    values = np.zeros(shapes, dtype=np.float32)
    rewards = np.zeros(shapes, dtype=np.float32)
    dones = np.zeros(shapes, dtype=np.float32)
    all_obs = []
    all_actions = []
    all_scores = []

    last_obs = self.last_obs

    for n in range(self.n_steps):
      actions, value_estimate = self.local_network.step(last_obs)
      actions = mask_unused_argument_samples(actions)
      size = last_obs['screen'].shape[1:3]

      values[n, :] = value_estimate
      all_obs.append(last_obs)
      all_actions.append(actions)

      pysc2_actions = actions_to_pysc2(actions, size)
      #print (actions)
      #print(pysc2_actions)
      obs_raw = self.envs.step(pysc2_actions)
      last_obs = self.preproc.preprocess_obs(obs_raw)
      rewards[n, :] = [t.reward for t in obs_raw]
      dones[n, :] = [t.last() for t in obs_raw]

      for t in obs_raw:
        if t.last():
          score = self._summarize_episode(t)
          self.cumulative_score += score

    self.last_obs = last_obs

    next_values = self.local_network.get_value(last_obs)

    returns, advs = compute_returns_advantages(
        rewards, dones, values, next_values, self.discount)

    actions = stack_and_flatten_actions(all_actions)
    obs = flatten_first_dims_dict(stack_ndarray_dicts(all_obs))
    returns = flatten_first_dims(returns)
    advs = flatten_first_dims(advs)

    if self.train:
      return self.local_network.train(
          obs, actions, returns, advs,
          summary=train_summary)

    return None

  def process(self, sess, global_t, summary_writer, summary_op, score_input):
    start_local_t = self.local_t

    self.local_t += 1
    self.run_batch()

    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t

  def process1(self, sess, global_t, summary_writer, summary_op, score_input):

    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    # t_max times loop
    for i in range(LOCAL_T_MAX):
      #pi_, value_ = self.local_network.run_policy_and_value(sess, self.last_obs)
      #print (pi_, value_)
      #action = self.choose_action(pi_)
      action, value_ = self.local_network.run_policy_and_value(sess, self.last_obs)

      states.append(self.last_obs)
      actions.append(action)
      values.append(value_)

      # process game
      #actions, value_estimate = self.local_network.step(self.last_obs)
      action = mask_unused_argument_samples(action)
      size = self.last_obs['screen'].shape[1:3]
      pysc2_actions = actions_to_pysc2(action, size)

      #print(size, pysc2_actions)

      obs_raw = self.envs.step(pysc2_actions)
      last_obs = self.preproc.preprocess_obs(obs_raw)

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("pi={}".format(pysc2_actions))
        print(" V={}".format(value_))

      #rewards[n, :] = [t.reward for t in obs_raw]
      #dones[n, :] = [t.last() for t in obs_raw]

      reward = obs_raw[0].reward
      terminal = obs_raw[0].last()

      # receive game result
      #reward = self.game_state.reward
      #terminal = self.game_state.terminal

      self.episode_reward += reward

      # clip reward
      rewards.append(np.clip(reward, -1, 1))

      self.local_t += 1

      # s_t1 -> s_t
      #self.game_state.update()
      self.last_obs = last_obs

      if terminal:
        terminal_end = True
        print("score={}".format(self.episode_reward))

        self._record_score(sess, summary_writer, summary_op, score_input,
                           self.episode_reward, global_t)

        self.episode_reward = 0
        # reset state
        self.reset()
        break


    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.last_obs)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      print(a)
      print(ai)
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate = self._anneal_learning_rate(global_t)
    sess.run( self.apply_gradients,
              feed_dict = {
                self.local_network.s: batch_si,
                self.local_network.a: batch_a,
                self.local_network.td: batch_td,
                self.local_network.r: batch_R,
                self.learning_rate_input: cur_learning_rate})
#

    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t


def mask_unused_argument_samples(actions):
  """Replace sampled argument id by -1 for all arguments not used
  in a steps action (in-place).
  """
  fn_id, arg_ids = actions
  for n in range(fn_id.shape[0]):
    a_0 = fn_id[n]
    unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[a_0].args)
    for arg_type in unused_types:
      arg_ids[arg_type][n] = -1
  return (fn_id, arg_ids)

def actions_to_pysc2(actions, size):
  """Convert agent action representation to FunctionCall representation."""
  height, width = size
  fn_id, arg_ids = actions
  actions_list = []
  for n in range(fn_id.shape[0]):
    a_0 = fn_id[n]
    a_l = []
    for arg_type in FUNCTIONS._func_list[a_0].args:
      arg_id = arg_ids[arg_type][n]
      if is_spatial_action[arg_type]:
        arg = [arg_id % width, arg_id // height]
      else:
        arg = [arg_id]
      a_l.append(arg)
    action = FunctionCall(a_0, a_l)
    actions_list.append(action)
  return actions_list



def compute_returns_advantages(rewards, dones, values, next_values, discount):
  """Compute returns and advantages from received rewards and value estimates.

  Args:
    rewards: array of shape [n_steps, n_env] containing received rewards.
    dones: array of shape [n_steps, n_env] indicating whether an episode is
      finished after a time step.
    values: array of shape [n_steps, n_env] containing estimated values.
    next_values: array of shape [n_env] containing estimated values after the
      last step for each environment.
    discount: scalar discount for future rewards.

  Returns:
    returns: array of shape [n_steps, n_env]
    advs: array of shape [n_steps, n_env]
  """
  returns = np.zeros([rewards.shape[0] + 1, rewards.shape[1]])

  returns[-1, :] = next_values
  for t in reversed(range(rewards.shape[0])):
    future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
    returns[t, :] = rewards[t, :] + future_rewards

  returns = returns[:-1, :]
  advs = returns - values

  return returns, advs



def flatten_first_dims(x):
  new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
  return x.reshape(*new_shape)


def flatten_first_dims_dict(x):
  return {k: flatten_first_dims(v) for k, v in x.items()}


def stack_and_flatten_actions(lst, axis=0):
  fn_id_list, arg_dict_list = zip(*lst)
  fn_id = np.stack(fn_id_list, axis=axis)
  fn_id = flatten_first_dims(fn_id)
  arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
  arg_ids = flatten_first_dims_dict(arg_ids)
  return (fn_id, arg_ids)
