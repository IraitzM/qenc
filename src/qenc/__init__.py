"""Reproduction code for "How Quantum Data Encoding Constrains Representation Learning".

Modules:
    data       datasets, downloads and the exact data draws of every experiment
    pooling    fixed block-averaging maps and their reconstruction loss
    models     angle, amplitude and QSM classifiers (PennyLane)
    protocols  data draws, candidates and model inputs of the five-task study and the spatial pilot
    measures   final states, Pauli and readout features, kernels and the classical fits of the ladder
    rank       the two-qubit entangler ladder of the controlled-rank regressions
    fastsim    batched torch statevector simulator used by the regime map
    utils      result paths, JSON helpers and the command line shared by the experiments
"""
