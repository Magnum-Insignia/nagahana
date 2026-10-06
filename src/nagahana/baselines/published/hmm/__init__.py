"""Hidden Markov models of multistep attacks.

core         discrete HMM: scaled and log-space forward-backward, Baum-Welch (EM), Viterbi, k-step
             prediction, first-passage forecasts, supervised estimation, sampling
evolution    differential-evolution training of HMM parameters (Chadza et al. 2020's DE reading)
symbols      observation symbols from feature vectors (k-means codebook) when no alert stream exists
forecasters  the reproductions: Holgado et al. 2020, Chadza et al. 2020, Ghafir et al. 2019
"""
