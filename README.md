## pysc2-study

Environment: Windows 10 + 8G nVidia GPU, Debian GNU/Linux 9.9 + 8G Tesla GPU   
TensorFlow 1.13
PySC2 2.02
Python 3.5+


### Experiments on:
Asynchronous Advanced Actor-Critic with Deep Convolutional Neural Network  
Asynchronous Q learning with Deep Convolutional Neural Network  

Training with A3C agent:  
python agent.py --map=MoveToBeacon --agent agents.a3c_agent.A3CAgent  

Training with DQN agent:  
python agent.py --map=MoveToBeacon --agent agents.dqn_agent.DeepQAgent  

Training with Random agent, for comparison:  
python agent.py --map=MoveToBeacon --agent agents.random_agent.RandomAgent

use --max_agent_steps to make longer trajectory if you have a powerful computer:  
python agent.py --map=FindAndDefeatZerglings --max_agent_steps=1000 

use --training=False to evaluate, will only start a single thread.:  
python agent.py --map=MoveToBeacon --training=False

--map set to other mini games as well:  
python agent.py --map=DefeatRoaches

After pysc2 installed, the training can be also run in this way  
python -m pysc2.bin.agent --map CollectMineralShards --agent agents.a3c_agent.A3CAgent  

python agent.py --map=MoveToBeacon --max_agent_steps=120 --agent=agents.dqn_agent.DeepQAgent --continuation=True

python agent.py --map=MoveToBeacon --agent=agents.dqn_agent.DeepQAgent --training=False