"""Competitive Baseline Schedulers for SMART SCAN Benchmark.

Exports:
- RobustPCAPSRScheduler: Robust PCA (Inexact ALM) + Predictive State Representations
- NMFScheduler: Non-Negative Matrix Factorization (Lee & Seung Multiplicative Updates)
- WhittleIndexRMABScheduler: Restless Multi-Armed Bandit with Whittle Indices
- ThompsonSamplingScheduler: Bayesian Thompson Sampling with Beta Conjugate Priors
- Exp3BanditScheduler: Adversarial Exponential-Weight Algorithm
- AdaptiveReceiverSearchScheduler: Pulse Repetition Interval (PRI) Tracking
- DoubleDQNScheduler: Value-Based Deep RL with Decoupled Target Network
"""

from scheduler.paradigms.matrix_factorization import NMFScheduler

__all__ = [
    "NMFScheduler",
]
