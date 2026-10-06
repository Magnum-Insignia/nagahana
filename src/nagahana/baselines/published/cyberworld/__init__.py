"""CyberWorld (Masukawa, Yun, Hassan, Oh, Jeong and Imani, arXiv:2609.31893; citation to verify): a
DreamerV3-style world model of a defended network.

networks     symlog / two-hot, DreamerV3's normalised GRU and MLPs, graph attention, per-modality
             self-attention and cross-attention fusion
rssm         the recurrent state-space model (1,024 deterministic units, 32 x 32 categorical latent)
world        the world model: modality encoders, RSSM, decoders and prediction heads, its loss
agent        the defense agent: Dreamer actor-critic trained on imagined trajectories in CyberWheel
forecaster   the forecasting baseline on NagaHana's data (P_inf, stages and next states from rollouts)
"""
