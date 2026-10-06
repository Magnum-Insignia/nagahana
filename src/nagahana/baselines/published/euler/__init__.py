"""EULER (King and Huang, NDSS 2022): temporal link prediction for lateral movement.

graph      snapshots of an event stream: node index, edge lists and propagation weights
model      GCN / GraphSAGE snapshot encoders, the recurrent model and the inner-product decoder
baseline   the reproduction: training, the validation cutoff (Eq. 6), scoring of edges or events
"""
