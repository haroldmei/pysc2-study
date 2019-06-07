"# pysc2-study" 

python -m pysc2.bin.agent --map CollectMineralShards --agent agents.a3c_agent.A3CAgent
python agent.py --map=MoveToBeacon --agent agents.a3c_agent.A3CAgent --save_replay=False --render=False

python agent.py --map=MoveToBeacon --agent agents.random_agent.RandomAgent --save_replay=False --render=False

python agent.py --map=MoveToBeacon 

python agent.py --map=FindAndDefeatZerglings --max_agent_steps=1000 

python agent.py --map=MoveToBeacon --training=False

python agent.py --map=DefeatRoaches --training=False