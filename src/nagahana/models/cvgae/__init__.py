"""CVG-AE: Complex Variational Graph AutoEncoder.

- `model`: the registry `ENCODERS`, the reference encoder `hgnn-mean-reference` (dense, mean aggregation;
  the realisation of the architecture chapter, a baseline for tests and ablations) and its ELBO.
- `attn`: the production encoder `hgnn-attn` (`CVGAE`; typed attention on sparse incidence, AS-03, with
  the learned planes of D-04's learning options). Importing this package registers both.
"""

from nagahana.models.cvgae.attn import CVGAE, segment_softmax
from nagahana.models.cvgae.model import ENCODERS, CVGAEEncoder

__all__ = ["CVGAE", "ENCODERS", "CVGAEEncoder", "segment_softmax"]
