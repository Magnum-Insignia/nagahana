"""FlowTransformer (Manocchio, Layeghy, Lo, Kulatilleke, Sarhan and Portmann, Expert Systems with
Applications 241:122564, 2024; arXiv:2304.14746): a transformer over sequences of flow records.

preprocessing   the framework's standard pre-processing (log-scaled numeric fields, top-n categorical levels)
model           input encodings, transformer blocks and classification heads
baseline        the reproduction: windowing, class-balanced training, early stopping, scoring
"""
