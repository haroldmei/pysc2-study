#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import importlib
import threading

from absl import app
from absl import flags
from absl import logging
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
#from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from run_loop import run_loop

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")

flags.DEFINE_bool("training", True, "Whether to train agents.")

flags.DEFINE_bool("render", True, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 60, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 8, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("net", "fcn", "atari or fcn.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_string("device", "0", "Device for training.")
flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.mark_flag_as_required("map")

FLAGS(sys.argv)
if FLAGS.training:
  PARALLEL = FLAGS.parallel
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = 1e5
  DEVICE = ['/cpu:0']

LOG = FLAGS.log_path+FLAGS.map+'/'+FLAGS.net
SNAPSHOT = FLAGS.snapshot_path+FLAGS.map+'/'+FLAGS.net
COUNTER = 0
START_WITH = 0
LOCK = threading.Lock()
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)
  
summary_writer = tf.summary.FileWriter(LOG)
total_score = 0

def run_thread(agent, players, map_name, visualize):
  """Run one thread worth of the environment with agents."""
  with sc2_env.SC2Env(
      map_name=map_name,
      players=players,
      agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=FLAGS.feature_minimap_size,
          feature_minimap=FLAGS.feature_minimap_size,
          rgb_screen=FLAGS.rgb_screen_size,
          rgb_minimap=FLAGS.rgb_minimap_size,
          action_space=FLAGS.action_space,
          use_feature_units=FLAGS.use_feature_units),
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      disable_fog=FLAGS.disable_fog,
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    #agents = [agent_cls() for agent_cls in agent_classes]

    start_at = 0
    global total_score
    replay_buffer = []
    for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          counter = 0
          with LOCK:
            global COUNTER
            COUNTER += 1
            if start_at == 0:
              start_at = COUNTER
            counter = COUNTER
          # Learning rate schedule
          learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
          agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
          replay_buffer = []
          if counter % FLAGS.snapshot_step == 1:
            agent.save_model(SNAPSHOT, counter)
          if counter >= FLAGS.max_steps:
            break
            
          obs = recorder[-1].observation
          score = obs["score_cumulative"][0]
          total_score += score
          mean_score = total_score/(COUNTER - start_at)
          summary = tf.Summary()
          summary.value.add(tag='episode_score', simple_value=score)
          summary_writer.add_summary(summary, COUNTER)
          
          summary.value.add(tag='mean_score', simple_value=mean_score)
          summary_writer.add_summary(summary, COUNTER)

          logging.info("Your score is: %s !", str(score))
          #print('Your score is '+str(score)+'!')
      elif is_done:
        obs = recorder[-1].observation
        score = obs["score_cumulative"][0]
        print('Your score is '+str(score)+'!')
    if FLAGS.save_replay:
      env.save_replay(agent.name)


def main(unused_argv):
  """Run an agent."""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  map_inst = maps.get(FLAGS.map)

  agent_classes = []
  players = []

  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)
  agent_classes.append(agent_cls)
  players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race],
                               FLAGS.agent_name or agent_name))

  if map_inst.players >= 2:
    if FLAGS.agent2 == "Bot":
      players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                 sc2_env.Difficulty[FLAGS.difficulty]))
    else:
      agent_module, agent_name = FLAGS.agent2.rsplit(".", 1)
      agent_cls = getattr(importlib.import_module(agent_module), agent_name)
      agent_classes.append(agent_cls)
      players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent2_race],
                                   FLAGS.agent2_name or agent_name))

  agents = []
  for i in range(PARALLEL):
    agent = agent_cls()
    agent.setup3(FLAGS.training, FLAGS.feature_minimap_size, FLAGS.feature_minimap_size)
    agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
    agents.append(agent)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  for i in range(PARALLEL):
    agents[i].setup(sess, summary_writer)

  # only initialize once..
  agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  threads = []
  for i in range(FLAGS.parallel - 1):
    #print('agent name,', agents[i].name)
    t = threading.Thread(target=run_thread,
                         args=(agents[i], players, FLAGS.map, False))
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(5)

  run_thread(agents[FLAGS.parallel - 1], players, FLAGS.map, FLAGS.render)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)
