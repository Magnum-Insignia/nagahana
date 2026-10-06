"""CyberWheel (Oak Ridge National Laboratory, github.com/ORNL/cyberwheel): the defense arena of CyberWorld.

env       the environment protocol the defense agents use: observation (host present, is_decoy,
          isolated, alert_now, alert_ever, topology, detector telemetry), decoy actions, reward, the four
          scripted red strategies and the network sizes; a synchronous vector environment
binding   CyberWheel itself, imported lazily from third_party/cyberwheel, behind that protocol
ppo       the PPO (MLP) and GNN-PPO defense baselines
"""
