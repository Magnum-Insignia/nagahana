"""Explanations: which fields, flags, ports and flow statistics drive a prediction.

The problem statement requires "an explainability output for each prediction — using SHAP values or
model attention weights — identifying which flags, ports, or flow statistics are driving the
prediction. Black-box outputs without interpretability are not acceptable." The owner's stack lists
SHAP and LIME (ai-mod-arch §6d).

NagaHana explains at four layers, from the inside out:
1. **Energy decomposition** (D-42): each TAAFT lens's share of E_total and of every descent step
   (`AnalysisOut.lens_energy`, `lens_share`).
2. **Typed attention**: spatial / temporal / causal weights of TSTCT and TAAFT (`need_weights=True`).
3. **Input-pooling weights**: which fields each state update was "about" (FieldEncoder pooling).
4. **Field attributions** (`attribution.py`): Expected Gradients (the SHAP GradientExplainer form)
   and LIME over field groups, both on the field-state embeddings of the input layer, so every
   number refers to a named column of the data model.
"""
