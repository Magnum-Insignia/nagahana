"""Input layer: value/status matrices → field states → update vectors (build-spec §2.1).

- `encoder.FieldEncoder`: the FieldEncoder (field states, attention pooling, clock switch).
- `encoder.categorical_row`: the fixed integer hash h(slot, code) mod R_hash (AS-33, AS-100).
"""

from nagahana.models.inputs.encoder import CONTRIBUTING_STATUS, FieldEncoder, categorical_row

__all__ = ["CONTRIBUTING_STATUS", "FieldEncoder", "categorical_row"]
