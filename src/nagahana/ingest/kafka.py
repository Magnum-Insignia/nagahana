"""Kafka adapter (template): live telemetry streams (datamodel.md; D-27 held).

"Since kafka can absord any upscaling of network traffic and work as a persistent queue before we
move on batches of data points to the ai model" (datamodel.md item 4).

The client library (confluent-kafka, aiokafka or kafka-python) is held (D-27). Topic naming follows
the proposed layering (datamodel/layers.py, P-04).

Operational security: queue latency is itself an attack vector ("detection-blinding via overload",
ARCH §10.1). Consumer lag and backpressure are monitored and reported, never silently absorbed.
"""

from __future__ import annotations

from collections.abc import Iterator

from nagahana.core.errors import NotBuiltYet
from nagahana.datamodel.records import StateUpdate
from nagahana.governance import decisions


class KafkaSource:
    """Kafka topic → state updates (template)."""

    name = "kafka"

    def __init__(self, topic: str) -> None:
        self.topic = topic

    def updates(self) -> Iterator[StateUpdate]:
        decisions.require("kafka-client")
        raise NotBuiltYet("Kafka consumer", waiting_on=("D-27", "P-04"))
