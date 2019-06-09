from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents import base_agent

from agents.network import build_net
import utils as U


class DeepQAgent(base_agent.BaseAgent):
  """An agent specifically for solving the mini-game maps."""
  def __init__(self, name='A3C/A3CAgent'):
    self.name = name
    self.training = False 
    self.summary = []
    # Minimap size, screen size and info size
    #assert msize == ssize
    self.msize = 64 #msize[0]
    self.ssize = 64 #ssize[0]
    self.isize = len(actions.FUNCTIONS)


  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer

  def setup2(self, obs_spec, action_spec):
    self.obs_spec = obs_spec
    self.action_spec = action_spec

  def setup3(self, training, msize, ssize):
    self.msize = msize[0]
    self.ssize = ssize[0]
    self.training = training

  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def reset(self):
    # Epsilon schedule
    self.epsilon = [0.05, 0.2]


  def build_model(self, reuse, dev, ntype):
    with tf.variable_scope(self.name) and tf.device(dev):
      if reuse:
        tf.get_variable_scope().reuse_variables()
        assert tf.get_variable_scope().reuse

      # Set inputs of networks
      self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
      self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

      # Build networks
      net = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS), ntype)
      self.spatial_action, self.non_spatial_action, self.value = net

      # Set targets and masks
      self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize**2], name='spatial_action_selected')
      self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')
      self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='non_spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

      # Compute log probability
      spatial_action_prob = tf.clip_by_value(tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1), 1e-10, 1.)
      non_spatial_action_prob = tf.clip_by_value(tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected * self.valid_non_spatial_action, axis=1), 1e-10, 1.)

      q_value = spatial_action_prob * self.valid_spatial_action + non_spatial_action_prob
      self.delta = self.value_target - q_value
      #self.clipped_error = tf.where(tf.abs(self.delta) < 1.0, 0.5 * tf.square(self.delta), tf.abs(self.delta) - 0.5, name='clipped_error')
      #value_loss = tf.reduce_mean(self.clipped_error, name='value_loss')
      
      value_loss = tf.reduce_mean(self.delta * self.delta)

      self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
      self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))
      self.summary.append(tf.summary.scalar('value_loss', value_loss))
      
      # Build the optimizer
      self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      grads = opt.compute_gradients(value_loss)
      cliped_grad = []
      for grad, var in grads:
        self.summary.append(tf.summary.histogram(var.op.name, var))
        grad = grad if grad is not None else tf.zeros_like(var)
        self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
        grad = tf.clip_by_norm(grad, 10.0)
        cliped_grad.append([grad, var])
      self.train_op = opt.apply_gradients(cliped_grad)
      self.summary_op = tf.summary.merge(self.summary)

      self.saver = tf.train.Saver(max_to_keep=100)



  def step(self, obs):
    minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
    minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
    screen = np.array(obs.observation.feature_screen, dtype=np.float32)
    screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
    # TODO: only use available actions
    info = np.zeros([1, self.isize], dtype=np.float32)
    #info[0, obs.observation['available_actions']] = 1
    info[0, obs.observation.available_actions] = 1

    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    non_spatial_action, spatial_action = self.sess.run(
      [self.non_spatial_action, self.spatial_action],
      feed_dict=feed)

    # Select an action and a spatial target
    non_spatial_action = non_spatial_action.ravel()
    spatial_action = spatial_action.ravel()
    valid_actions = obs.observation.available_actions #['available_actions']
    act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
    target = np.argmax(spatial_action)
    target = [int(target // self.ssize), int(target % self.ssize)]

    if False:
      print(actions.FUNCTIONS[act_id].name, target)
      #print(self.action_spec.functions[act_id].name, target)

    # Epsilon greedy exploration
    if self.training and np.random.rand() < self.epsilon[0]:
      act_id = np.random.choice(valid_actions)
    if self.training and np.random.rand() < self.epsilon[1]:
      dy = np.random.randint(-4, 5)
      target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
      dx = np.random.randint(-4, 5)
      target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    # Set act_id and act_args
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
    #for arg in self.action_spec.functions[act_id].args:
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target[1], target[0]])
      else:
        act_args.append([0])  # TODO: Be careful
    return actions.FunctionCall(act_id, act_args)


  def update(self, rbs, disc, lr, cter):
    # Compute R, which is value of the last observation
    spatial_action = None
    non_spatial_action = None

    obs = rbs[-1][-1]
    if obs.last():
      R = 0
    else:
      minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation.feature_screen, dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation.available_actions] = 1

      # first get probabilities for each action; Then greedly pick the largest to calculate q value. one hot vector softmax
      # have low confidence, just use full episode and the last observation R should be just 0.
      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      spatial_action, non_spatial_action = self.sess.run([self.spatial_action, self.non_spatial_action], feed_dict=feed)

    # Compute targets and masks
    minimaps = []
    screens = []
    infos = []
    valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
    spatial_action_selected = np.zeros([len(rbs), self.ssize**2], dtype=np.float32)
    valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
    non_spatial_action_selected = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)

    rbs.reverse()
    for i, [obs, action, next_obs] in enumerate(rbs):
      minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation.feature_screen, dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      #info[0, obs.observation['available_actions']] = 1
      info[0, obs.observation.available_actions] = 1

      minimaps.append(minimap)
      screens.append(screen)
      infos.append(info)

      act_id = action.function
      act_args = action.arguments

      #valid_actions = obs.observation["available_actions"]
      valid_actions = obs.observation.available_actions
      valid_non_spatial_action[i, valid_actions] = 1
      non_spatial_action_selected[i, act_id] = 1

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, ind] = 1


    value_target = np.zeros([len(rbs)], dtype=np.float32)
    if spatial_action is not None:
      q_spatial = np.max(spatial_action * valid_spatial_action[0], axis=1)
      q_non_spatial = np.max(non_spatial_action * valid_non_spatial_action[0], axis=1)
      q_value = q_spatial + q_non_spatial
      R = q_value[0]
      
    value_target[-1] = R

    for i, [obs, action, next_obs] in enumerate(rbs):
      reward = obs.reward
      value_target[i] = reward + disc * value_target[i-1]

    minimaps = np.concatenate(minimaps, axis=0)
    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)

    # Train
    feed = {self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected,
            self.learning_rate: lr}
    _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
    self.summary_writer.add_summary(summary, cter)


  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])
