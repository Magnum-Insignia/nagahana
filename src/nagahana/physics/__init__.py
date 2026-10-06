"""The shared physics boundary: network-physics and protocol laws, hard limits, and the term Phi_phys.

Importing the package registers every law in `residuals.RESIDUALS`: flow accounting (`residuals`),
network physics (`network`), protocol laws (`protocol`) and queueing (`queueing`).
"""

from nagahana.physics import network, protocol, queueing, residuals  # noqa: F401  (registration)
