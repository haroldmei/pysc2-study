# -*- coding: utf-8 -*-
import tensorflow as tf
import threading

import shutil

import signal
import math
import os
import time

from a3c_training_thread import A3CTrainingThread
from acnetwork import AdvActorCriticNetwork
from rmsprop_applier import RMSPropApplier

from environment import SubprocVecEnv, make_sc2env, SingleEnv

from functools import partial

import argparse

from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import ACTION_SIZE



from absl import flags
FLAGS = flags.FLAGS
FLAGS(['run.py'])

parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
parser.add_argument('experiment_id', type=str,
                    help='identifier to store experiment results')
parser.add_argument('--eval', action='store_true',
                    help='if false, episode scores are evaluated')
parser.add_argument('--ow', action='store_true',
                    help='overwrite existing experiments (if --train=True)')
parser.add_argument('--map', type=str, default='MoveToBeacon',
                    help='name of SC2 map')
parser.add_argument('--vis', action='store_true',
                    help='render with pygame')
parser.add_argument('--max_windows', type=int, default=1,
                    help='maximum number of visualization windows to open')
parser.add_argument('--res', type=int, default=32,
                    help='screen and minimap resolution')
parser.add_argument('--envs', type=int, default=32,
                    help='number of environments simulated in parallel')
parser.add_argument('--step_mul', type=int, default=8,
                    help='number of game steps per agent step')
parser.add_argument('--steps_per_batch', type=int, default=16,
                    help='number of agent steps when collecting trajectories for a single batch')
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount for future rewards')
parser.add_argument('--iters', type=int, default=-1,
                    help='number of iterations to run (-1 to run forever)')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device id')
parser.add_argument('--nhwc', action='store_true',
                    help='train fullyConv in NCHW mode')
parser.add_argument('--summary_iters', type=int, default=10,
                    help='record training summary after this many iterations')
parser.add_argument('--save_iters', type=int, default=5000,
                    help='store checkpoint after this many iterations')
parser.add_argument('--max_to_keep', type=int, default=5,
                    help='maximum number of checkpoints to keep before discarding older ones')
parser.add_argument('--entropy_weight', type=float, default=1e-3,
                    help='weight of entropy loss')
parser.add_argument('--value_loss_weight', type=float, default=0.5,
                    help='weight of value function loss')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='initial learning rate')
parser.add_argument('--save_dir', type=str, default=os.path.join('out','models'),
                    help='root directory for checkpoint storage')
parser.add_argument('--summary_dir', type=str, default=os.path.join('out','summary'),
                    help='root directory for summary storage')

args = parser.parse_args()
# TODO write args to config file and store together with summaries (https://pypi.python.org/pypi/ConfigArgParse)
args.train = not args.eval

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

""" environment preparation """

ckpt_path = os.path.join(args.save_dir, args.experiment_id)
summary_type = 'train' if args.train else 'eval'
summary_path = os.path.join(args.summary_dir, args.experiment_id, summary_type)

if args.train and args.ow:
    shutil.rmtree(ckpt_path, ignore_errors=True)
    shutil.rmtree(summary_path, ignore_errors=True)
size_px = (args.res, args.res)
env_args = dict(
    map_name=args.map,
    step_mul=args.step_mul,
    game_steps_per_episode=0,
    screen_size_px=size_px,
    minimap_size_px=size_px)
vis_env_args = env_args.copy()
vis_env_args['visualize'] = args.vis
num_vis = min(args.envs, args.max_windows)
env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
num_no_vis = args.envs - num_vis
if num_no_vis > 0:
    env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)
envs = SubprocVecEnv(env_fns)


global_t = 0

stop_requested = False

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

network_data_format = 'NHWC' if args.nhwc else 'NCHW'
global_network = AdvActorCriticNetwork(
        sess=sess,
        action_size=ACTION_SIZE,
        thread_index=-1,
        device=device,
        network_data_format=network_data_format,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        learning_rate=args.lr,
        max_to_keep=args.max_to_keep)

training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

for i in range(PARALLEL_SIZE):
  training_thread = A3CTrainingThread(sess=sess, thread_index=i,
                                      global_network=global_network,
                                      initial_learning_rate=initial_learning_rate,
                                      learning_rate_input=learning_rate_input,
                                      grad_applier=grad_applier,
                                      max_global_time_step=MAX_TIME_STEP,
                                      device = device,
                                      network_data_format=network_data_format,
                                      value_loss_weight=args.value_loss_weight,
                                      entropy_weight=args.entropy_weight,
                                      learning_rate=args.lr,
                                      max_to_keep=args.max_to_keep,
                                      envs=envs,
                                      res=args.res,
                                      n_steps=args.steps_per_batch)

  training_threads.append(training_thread)

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.summary.scalar("score", score_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())
else:
  print("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0


def train_function(parallel_index):
  global global_t
  
  training_thread = training_threads[parallel_index]

  # set start_time
  start_time = time.time() - wall_t
  training_thread.set_start_time(start_time)

  while True:
    if stop_requested:
      break
    if global_t > MAX_TIME_STEP:
      break

    diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                            summary_op, score_input)
    global_t += diff_global_t
    
    
def signal_handler(signal, frame):
  global stop_requested
  print('You pressed Ctrl+C!')
  stop_requested = True
  
train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))
  
signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
  t.start()

print('Press Ctrl+C to stop')
signal.pause()

print('Now saving data. Please wait')
  
for t in train_threads:
  t.join()

if not os.path.exists(CHECKPOINT_DIR):
  os.mkdir(CHECKPOINT_DIR)  

# write wall time
wall_t = time.time() - start_time
wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
  f.write(str(wall_t))

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

