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
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.lib import point_flag
import tensorflow as tf

from run_loop import run_loop

COUNTER = 0
LOCK = threading.Lock()
FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")

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
                  
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "atari or fcn.")

flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_enum("bot_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Bot 1's race.")

flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")


flags.DEFINE_integer("max_agent_steps", 60, "Total agent steps.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 16, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

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
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)


def run_thread(agent, map_name, visualize):
  with sc2_env.SC2Env(
    map_name=map_name,
    #agent_race=FLAGS.agent_race,
    #bot_race=FLAGS.bot_race,
    difficulty=FLAGS.difficulty,
    step_mul=FLAGS.step_mul,
    agent_interface_format=sc2_env.parse_agent_interface_format(
        feature_screen=FLAGS.feature_screen_size,
        feature_minimap=FLAGS.feature_minimap_size,
        rgb_screen=FLAGS.rgb_screen_size,
        rgb_minimap=FLAGS.rgb_minimap_size,
        action_space=FLAGS.action_space,
        use_feature_units=FLAGS.use_feature_units),
    visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)

    # Only for a single player!
    replay_buffer = []
    for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          counter = 0
          with LOCK:
            global COUNTER
            COUNTER += 1
            counter = COUNTER
          # Learning rate schedule
          learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
          agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
          replay_buffer = []
          if counter % FLAGS.snapshot_step == 1:
            agent.save_model(SNAPSHOT, counter)
          if counter >= FLAGS.max_steps:
            break
      elif is_done:
        obs = recorder[-1].observation
        score = obs["score_cumulative"][0]
        print('Your score is '+str(score)+'!')
    if FLAGS.save_replay:
      env.save_replay(agent.name)


def _main(unused_argv):
  """Run agents"""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  # Setup agents
  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)

  agents = []
  for i in range(PARALLEL):
    agent = agent_cls()
    agent.setup3(FLAGS.training, FLAGS.feature_minimap_size, FLAGS.feature_minimap_size)
    agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
    agents.append(agent)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  summary_writer = tf.summary.FileWriter(LOG)
  for i in range(PARALLEL):
    agents[i].setup(sess, summary_writer)

  agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  # Run threads
  threads = []
  for i in range(PARALLEL - 1):
    t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False))
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(5)

  run_thread(agents[-1], FLAGS.map, FLAGS.render)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)


if __name__ == "__main__":
  app.run(_main)
