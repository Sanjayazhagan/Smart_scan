"""UCB-first product controller with observable Adaptive-MoE escalation.

The controller uses the inexpensive discounted UCB expert by default.  It
escalates only when the receiver's own Track-2 state or rolling observation
history indicates a situation where the Adaptive MoE has a specialised guard.
No simulator reward, emitter truth, or scenario label is used for routing.
"""

from __future__ import annotations

from scheduler.adaptive_moe import (
    AdaptiveMixtureOfExpertsScheduler,
    Regime,
    extract_complexity_features,
)
from scheduler.baselines import BaseScheduler
from scheduler.track2_core import NUM_BANDS
from scheduler.track2_runtime import (
    DEFAULT_MAX_SCAN_AGE,
    DEFAULT_MODEL_PATH,
    Track2Runtime,
)


class UCBFirstAdaptiveScheduler(BaseScheduler):
    """Fast observable UCB with selective access to Adaptive-MoE experts."""

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        *,
        adaptive: AdaptiveMixtureOfExpertsScheduler | None = None,
        runtime: Track2Runtime | None = None,
        model_path=DEFAULT_MODEL_PATH,
        max_scan_age: float = DEFAULT_MAX_SCAN_AGE,
    ):
        if num_bands != NUM_BANDS:
            raise ValueError(f"UCB-first routing requires exactly {NUM_BANDS} bands")
        super().__init__(num_bands)
        self.adaptive = adaptive or AdaptiveMixtureOfExpertsScheduler(
            num_bands,
            runtime=runtime,
            model_path=model_path,
            max_scan_age=max_scan_age,
        )
        self.runtime = self.adaptive.runtime
        # The same UCB instance is updated inside Adaptive MoE every step, so
        # its history stays warm even while another expert handles a scan.
        self.ucb_expert = self.adaptive.ucb_expert
        self.last_decision_trace: dict = {}

    @staticmethod
    def _should_escalate(regime: Regime, history_regime: str) -> tuple[bool, str]:
        if history_regime == "clean_stationary":
            return True, "history_stationary_specialist"
        # Held-out ablation showed that tree escalation for instantaneous
        # complexity, OOD, or noisy labels did not reliably beat UCB. Keep
        # those states on UCB until a sustained clean/stationary pattern is
        # established by observation history.
        return False, "ucb_primary"

    def select_band(self) -> int:
        features = extract_complexity_features(self.runtime.get_global_belief())
        regime = self.adaptive.rule_router.route(features)
        monitor = self.adaptive.regime_monitor.snapshot()
        escalate, routing_source = self._should_escalate(
            regime, str(monitor["history_regime"])
        )

        if escalate:
            selected = int(self.adaptive.select_band())
            adaptive_trace = dict(self.adaptive.last_decision_trace)
            expert = str(adaptive_trace.get("expert", "adaptive"))
            controller_mode = "adaptive_escalation"
        else:
            selected = int(self.ucb_expert.select_band())
            adaptive_trace = {}
            expert = "ucb"
            controller_mode = "ucb_primary"

        if not 0 <= selected < NUM_BANDS:
            raise RuntimeError(f"UCB-first controller returned invalid band {selected}")
        self.last_decision_trace = {
            "selected_band": selected,
            "controller_mode": controller_mode,
            "regime": regime.value,
            "expert": expert,
            "routing_source": routing_source,
            "router": dict(self.adaptive.rule_router.last_trace),
            "features": features.as_dict(),
            "regime_monitor": monitor,
            "ucb_trace": dict(self.ucb_expert.last_trace),
            "adaptive_trace": adaptive_trace,
        }
        return selected

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        # Adaptive owns the shared UCB, monitor, and Track-2 runtime. Delegating
        # once prevents double-counting observations or advancing Track 2 twice.
        self.adaptive.update(band, reward, obs_dict)

    def begin_scored_phase(self):
        return self.adaptive.begin_scored_phase()
