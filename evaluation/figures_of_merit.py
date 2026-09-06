"""Figures of Merit (FoM) Engine for Electronic Warfare Smart Scan.

Implements rigorous, traceable Electronic Support Measures (ESM) metrics:
1. Probability of Detection (Pd):
   - Burst/Window Pd: Fraction of contiguous emitter bursts intercepted.
   - Opportunity Pd: Fraction of active signal transmission opportunities intercepted.
2. Probability of False Alarm (Pfa):
   - Noise False Alarm Rate: Detections on empty bands with pure thermal noise.
   - Decoy Misclassification Rate: Detections on channels with hostile jammer/decoy interference.
   - Overall Pfa: Total false detections over all un-transmitted channel observations.
3. Average Intercept Time Error (Delta_t):
   - Latency in time slots from emitter burst start to first receiver intercept.
4. Average Intercept Rate:
   - Successful authentic intercepts per slot and per simulated second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EmitterBurst:
    """Represents a contiguous transmission window of an emitter."""
    emitter_id: int
    band: int
    start_step: int
    end_step: int
    first_intercept_step: Optional[int] = None
    intercept_count: int = 0
    snr_db: float = 0.0

    @property
    def intercepted(self) -> bool:
        return self.first_intercept_step is not None

    @property
    def time_to_intercept(self) -> Optional[int]:
        if self.first_intercept_step is None:
            return None
        return self.first_intercept_step - self.start_step


@dataclass
class StepRecord:
    """Step-by-step observation and ground truth log."""
    step: int
    action: int
    detected: bool
    quality: float
    true_signal_present: bool
    interference_present: bool
    ground_truth_active_bands: list[int]
    ground_truth_active_emitters: list[int]
    effective_snr_db: float = 0.0
    reward: float = 0.0
    switched: bool = False


@dataclass
class FiguresOfMeritResult:
    """Encapsulates all computed Figures of Merit with traceable raw counts."""
    # Probability of Detection (Pd)
    pd_window_pct: float                # % of emitter bursts intercepted (>=1 hit)
    pd_opportunity_pct: float           # % of active signal steps intercepted
    total_emitted_bursts: int           # Denominator for Burst Pd
    burst_intercept_successes: int      # Numerator for Burst Pd
    opportunity_hit_slots: int          # Numerator for Opportunity Pd
    spectrum_active_steps: int          # Denominator for Opportunity Pd

    # Probability of False Alarm (Pfa)
    pfa_overall_pct: float              # Overall false alarm rate %
    pfa_empty_noise_pct: float          # False alarm rate on pure thermal noise bands %
    pfa_decoy_jammer_pct: float         # False alarm rate on hostile decoy/jammer bands %
    total_false_alarm_events: int       # Numerator for Overall Pfa
    empty_band_opportunities: int       # Denominator for Overall Pfa
    pure_noise_false_alarms: int        # Numerator for Noise Pfa
    pure_noise_opportunities: int       # Denominator for Noise Pfa
    decoy_false_alarms: int             # Numerator for Decoy Pfa
    decoy_opportunities: int            # Denominator for Decoy Pfa

    # Time to Intercept (Delta_t)
    mean_intercept_time_error: float
    std_intercept_time_error: float
    median_intercept_time_error: float
    p90_intercept_time_error: float

    # Intercept Throughput
    intercepts_per_episode: int
    intercept_rate_per_step: float
    intercept_rate_per_sec: float       # Assuming nominal hardware dwell slots (e.g. 2.0 ms)

    # Mission Efficiency
    mean_reward: float
    mean_switches: int
    dwell_ratio_pct: float

    def get_known_limitations(self) -> list[dict]:
        """Surface explicit flagged engineering limitations."""
        limitations = []
        if self.decoy_opportunities > 0 and self.pure_noise_opportunities > 0:
            pfa_noise = self.pure_noise_false_alarms / self.pure_noise_opportunities
            pfa_jammer = self.decoy_false_alarms / self.decoy_opportunities
            if pfa_jammer > 1.5 * pfa_noise:
                inv_jammer = round(1.0 / pfa_jammer) if pfa_jammer > 0 else "N/A"
                inv_noise = round(1.0 / pfa_noise) if pfa_noise > 0 else "N/A"
                limitations.append({
                    "limitation": "elevated_false_alarm_under_jamming",
                    "pfa_pure_noise": round(pfa_noise, 4),
                    "pfa_under_jamming": round(pfa_jammer, 4),
                    "ratio_elevation": round(pfa_jammer / max(pfa_noise, 1e-6), 2),
                    "counts_pure_noise": f"{self.pure_noise_false_alarms}/{self.pure_noise_opportunities}",
                    "counts_under_jamming": f"{self.decoy_false_alarms}/{self.decoy_opportunities}",
                    "description": (
                        f"Perception layer authentication false-alarms roughly 1 in {inv_jammer} "
                        f"times ({round(pfa_jammer * 100, 2)}%, {self.decoy_false_alarms}/{self.decoy_opportunities}) when a jammer/decoy is active, "
                        f"vs ~1 in {inv_noise} ({round(pfa_noise * 100, 2)}%, {self.pure_noise_false_alarms}/{self.pure_noise_opportunities}) "
                        "on pure background noise. Root cause: high-power jammer phase jitter partially leaks through prototype cosine metric. "
                        "Mitigation TBD (requires adaptive gating thresholds or multi-scale wavelet feature representation)."
                    ),
                })
        return limitations


def extract_emitter_bursts(records: List[StepRecord]) -> List[EmitterBurst]:
    """Reconstruct contiguous emitter transmission bursts from ground truth logs."""
    bursts: List[EmitterBurst] = []
    active_bursts: Dict[Tuple[int, int], EmitterBurst] = {}

    for rec in records:
        t = rec.step
        current_emitters = rec.ground_truth_active_emitters
        current_bands = rec.ground_truth_active_bands
        current_pairs = set(zip(current_emitters, current_bands))

        # Check ended bursts
        for pair in list(active_bursts.keys()):
            if pair not in current_pairs:
                b = active_bursts.pop(pair)
                b.end_step = t - 1
                bursts.append(b)

        # Update or start new bursts
        for em_id, band in current_pairs:
            pair = (em_id, band)
            if pair not in active_bursts:
                active_bursts[pair] = EmitterBurst(
                    emitter_id=em_id,
                    band=band,
                    start_step=t,
                    end_step=t,
                    snr_db=rec.effective_snr_db if band == rec.action else 0.0
                )

            # Check if receiver is currently tuned to this burst and detected it
            b_obj = active_bursts[pair]
            if rec.action == band and rec.detected and rec.true_signal_present:
                if b_obj.first_intercept_step is None:
                    b_obj.first_intercept_step = t
                b_obj.intercept_count += 1
                if rec.effective_snr_db != 0.0:
                    b_obj.snr_db = rec.effective_snr_db

    # Close any still-active bursts at end of episode
    if records:
        final_step = records[-1].step
        for b in active_bursts.values():
            b.end_step = final_step
            bursts.append(b)

    return bursts


def compute_figures_of_merit(
    records: List[StepRecord],
    nominal_slot_seconds: float = 0.002,  # 2.0 ms nominal hardware scan slot
) -> FiguresOfMeritResult:
    """Compute all required Figures of Merit from an episode step record log.

    Every figure of merit is calculated using explicitly named numerator and
    denominator variables to ensure complete numerical traceability.
    """
    if not records:
        return FiguresOfMeritResult(
            pd_window_pct=0.0, pd_opportunity_pct=0.0,
            total_emitted_bursts=0, burst_intercept_successes=0,
            opportunity_hit_slots=0, spectrum_active_steps=0,
            pfa_overall_pct=0.0, pfa_empty_noise_pct=0.0, pfa_decoy_jammer_pct=0.0,
            total_false_alarm_events=0, empty_band_opportunities=0,
            pure_noise_false_alarms=0, pure_noise_opportunities=0,
            decoy_false_alarms=0, decoy_opportunities=0,
            mean_intercept_time_error=0.0, std_intercept_time_error=0.0,
            median_intercept_time_error=0.0, p90_intercept_time_error=0.0,
            intercepts_per_episode=0, intercept_rate_per_step=0.0,
            intercept_rate_per_sec=0.0, mean_reward=0.0, mean_switches=0,
            dwell_ratio_pct=0.0
        )

    # -------------------------------------------------------------------------
    # 1. PROBABILITY OF DETECTION (Pd)
    # -------------------------------------------------------------------------
    # A) Burst / Window Pd:
    #    Fraction of discrete emitter transmission bursts intercepted at least once.
    #    Numerator   : Number of emitter bursts with intercept_count >= 1
    #    Denominator : Total number of discrete emitter bursts in episode
    bursts = extract_emitter_bursts(records)
    total_emitted_bursts = len(bursts)
    intercepted_bursts = [b for b in bursts if b.intercepted]
    burst_intercept_successes = len(intercepted_bursts)

    pd_window_ratio = (
        burst_intercept_successes / total_emitted_bursts
        if total_emitted_bursts > 0 else 0.0
    )

    # B) Opportunity Pd:
    #    Fraction of steps where ANY emitter was active in the multi-channel spectrum
    #    that resulted in a successful authentic detection by the tuned receiver.
    #    Numerator   : Steps where tuned channel had true signal AND was detected
    #    Denominator : Steps where one or more emitters were active anywhere in spectrum
    opportunity_hit_slots = sum(1 for r in records if r.true_signal_present and r.detected)
    spectrum_active_steps = sum(1 for r in records if len(r.ground_truth_active_bands) > 0)

    pd_opportunity_ratio = (
        opportunity_hit_slots / spectrum_active_steps
        if spectrum_active_steps > 0 else 0.0
    )

    # -------------------------------------------------------------------------
    # 2. PROBABILITY OF FALSE ALARM (Pfa)
    # -------------------------------------------------------------------------
    # OPERATIONAL DEFINITION:
    # A "False Alarm" occurs when the receiver scanner is tuned to a channel where
    # NO authentic emitter is transmitting (true_signal_present == False), but the
    # detector asserts a signal detection (detected == True).
    #
    # The operational denominator is the total count of decision opportunities
    # where the tuned band was empty (true_signal_present == False).
    #
    # We report this across three distinct categories:
    # 1. Pure Noise Pfa : Receiver tuned to an empty band with only AWGN thermal noise.
    # 2. Decoy/Jammer Pfa: Receiver tuned to an empty band containing hostile DRFM / barrage jamming.
    # 3. Overall Pfa     : Pooled false alarm rate across all non-signal scan steps.
    empty_band_scans = [r for r in records if not r.true_signal_present]
    empty_band_opportunities = len(empty_band_scans)
    false_alarm_scans = [r for r in empty_band_scans if r.detected]
    total_false_alarm_events = len(false_alarm_scans)

    pfa_overall_ratio = (
        total_false_alarm_events / empty_band_opportunities
        if empty_band_opportunities > 0 else 0.0
    )

    # Pure noise breakdown
    pure_noise_scans = [r for r in empty_band_scans if not r.interference_present]
    pure_noise_opportunities = len(pure_noise_scans)
    pure_noise_false_alarms = sum(1 for r in pure_noise_scans if r.detected)
    pfa_empty_noise_ratio = (
        pure_noise_false_alarms / pure_noise_opportunities
        if pure_noise_opportunities > 0 else 0.0
    )

    # Decoy / Jammer breakdown
    decoy_scans = [r for r in empty_band_scans if r.interference_present]
    decoy_opportunities = len(decoy_scans)
    decoy_false_alarms = sum(1 for r in decoy_scans if r.detected)
    pfa_decoy_jammer_ratio = (
        decoy_false_alarms / decoy_opportunities
        if decoy_opportunities > 0 else 0.0
    )

    # -------------------------------------------------------------------------
    # 3. AVERAGE INTERCEPT TIME ERROR (Delta_t)
    # -------------------------------------------------------------------------
    # Latency in decision time slots between the onset of an emitter transmission
    # burst (start_step) and the step where the receiver first intercepts it.
    time_errors = [
        b.time_to_intercept for b in intercepted_bursts
        if b.time_to_intercept is not None
    ]
    if time_errors:
        mean_time_err = float(np.mean(time_errors))
        std_time_err = float(np.std(time_errors, ddof=1)) if len(time_errors) > 1 else 0.0
        median_time_err = float(np.median(time_errors))
        p90_time_err = float(np.percentile(time_errors, 90))
    else:
        mean_time_err = 0.0
        std_time_err = 0.0
        median_time_err = 0.0
        p90_time_err = 0.0

    # -------------------------------------------------------------------------
    # 4. INTERCEPT THROUGHPUT & CONTROL METRICS
    # -------------------------------------------------------------------------
    total_steps = len(records)
    intercept_rate_per_step = opportunity_hit_slots / total_steps if total_steps > 0 else 0.0
    simulated_seconds = total_steps * nominal_slot_seconds
    intercept_rate_per_sec = (
        opportunity_hit_slots / simulated_seconds
        if simulated_seconds > 0 else 0.0
    )

    total_reward = sum(r.reward for r in records)
    total_switches = sum(1 for r in records if r.switched)
    dwell_ratio = 1.0 - (total_switches / total_steps) if total_steps > 0 else 0.0

    return FiguresOfMeritResult(
        pd_window_pct=round(pd_window_ratio * 100.0, 2),
        pd_opportunity_pct=round(pd_opportunity_ratio * 100.0, 2),
        total_emitted_bursts=total_emitted_bursts,
        burst_intercept_successes=burst_intercept_successes,
        opportunity_hit_slots=opportunity_hit_slots,
        spectrum_active_steps=spectrum_active_steps,
        pfa_overall_pct=round(pfa_overall_ratio * 100.0, 3),
        pfa_empty_noise_pct=round(pfa_empty_noise_ratio * 100.0, 3),
        pfa_decoy_jammer_pct=round(pfa_decoy_jammer_ratio * 100.0, 3),
        total_false_alarm_events=total_false_alarm_events,
        empty_band_opportunities=empty_band_opportunities,
        pure_noise_false_alarms=pure_noise_false_alarms,
        pure_noise_opportunities=pure_noise_opportunities,
        decoy_false_alarms=decoy_false_alarms,
        decoy_opportunities=decoy_opportunities,
        mean_intercept_time_error=round(mean_time_err, 2),
        std_intercept_time_error=round(std_time_err, 2),
        median_intercept_time_error=round(median_time_err, 2),
        p90_intercept_time_error=round(p90_time_err, 2),
        intercepts_per_episode=opportunity_hit_slots,
        intercept_rate_per_step=round(intercept_rate_per_step, 3),
        intercept_rate_per_sec=round(intercept_rate_per_sec, 1),
        mean_reward=round(total_reward, 2),
        mean_switches=total_switches,
        dwell_ratio_pct=round(dwell_ratio * 100.0, 1),
    )
