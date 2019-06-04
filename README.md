"# pysc2-study" 

python -m pysc2.bin.agent --map CollectMineralShards --agent agents.a3c_agent.A3CAgent
python -m agent --map=MoveToBeacon --agent agents.a3c_agent.A3CAgent --save_replay=False --render=False