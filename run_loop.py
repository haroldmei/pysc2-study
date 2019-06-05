from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


def run_loop(agents, env, max_frames=0, max_episodes=0):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  total_episodes = 0
  start_time = time.time()

  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
    agent.setup2(obs_spec, act_spec)

  try:
    while not max_episodes or total_episodes < max_episodes:
      num_frames = 0
      total_episodes += 1
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        num_frames += 1
        total_frames += 1

        last_timesteps = timesteps

        actions = [agent.step(timestep)
                   for agent, timestep in zip(agents, timesteps)]
        #if max_frames and total_frames >= max_frames:
        #  return
        #if timesteps[0].last():
        #  break

        timesteps = env.step(actions)
        
        is_done = (num_frames >= max_frames) or timesteps[0].last()
        #print("-----", len(last_timesteps), actions[0], is_done)

        yield [last_timesteps[0], actions[0], timesteps[0]], is_done
        if is_done == True:
          break

  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))

'''
def run_loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  try:
    while True:
      num_frames = 0
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        timesteps = env.step(actions)
        # Only for a single player!
        is_done = (num_frames >= max_frames) or timesteps[0].last()
        yield [last_timesteps[0], actions[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)
'''