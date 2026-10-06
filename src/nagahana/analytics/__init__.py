"""Data analytics of the NagaHana corpora, the analysis that opens data preparation (D-22).

Every analysis reads a `corpus.Corpus` (all state updates of all sources in the canonical column layout,
with observation statuses, labels for auditing, and split assignments) or plain arrays, and returns a
`report.Report` written as JSON, CSV and Markdown. Absence is never zero (D-41): no estimator imputes an
absent cell; absence is analysed as a fact about the sensor.

Modules
-------
    corpus          the analysis view of sources, labels and split manifests
    report          reports: summary, tables, plot-ready figure data; JSON / CSV / Markdown writers
    config          typed configuration of every analysis (the single source of the conf/analytics YAML)
    robust          robust statistics and tail indices (Hill, moment estimator, Clauset choice of k)
    information     entropy and mutual information: plug-in, Miller-Madow, KSG, Ross, partially observed
    dependence      Pearson, Spearman, exact fast distance correlation, Cramer's V
    discrimination  AUROC with DeLong intervals, out-of-fold category scores
    duplicates      exact (hashing) and near (banded LSH) duplicates, label conflicts
    eda             exploratory analysis
    tda             topological data analysis with an in-house persistent-homology engine
    spatial         graph analytics of traffic: centralities, communities, motifs, multiplex planes, drift
    temporal        event times, serial dependence, spectra, stationarity, change points, long memory
    leakage         label and leakage audits and split integrity (D-23)
    drift           covariate shift, label shift, streaming concept-drift detectors
    observability   coverage, capabilities, reliability and ordering of the sensors
    cli             `python -m nagahana analyze ...`
The information audit (P-15) composes these estimators in `nagahana.lab.info_audit`.
"""
