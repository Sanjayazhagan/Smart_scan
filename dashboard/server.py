"""Local Python backend for the SMART SCAN dashboard.

Connects the web dashboard directly to the real SmartScanEnv and
the validated production scheduler choices.
"""

import http.server
import json
import socketserver
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.environment import SmartScanEnv
from scheduler.baselines import BaseScheduler
from scheduler.ucb_first_adaptive import UCBFirstAdaptiveScheduler

PORT = 8000
MODEL_OPTIONS = {
    "ucb_first_adaptive": "UCB-FIRST ADAPTIVE CONTROLLER",
}

class SimulationManager:
    def __init__(self):
        self.num_bands = 20
        self.episode_length = 200
        self.scenario = "stationary"
        self.seed = 42
        self.env: Optional[SmartScanEnv] = None
        self.scheduler: Optional[BaseScheduler] = None
        self.step_count = 0
        self.total_score = 0.0
        self.hits = 0
        self.misses = 0
        self.empty_scans = 0
        self.model_key = "ucb_first_adaptive"
        self.model_display_name = MODEL_OPTIONS[self.model_key]
        self.reset("stationary", 42, "ucb_first_adaptive")

    def reset(
        self,
        scenario: str = "stationary",
        seed: int = 42,
        model: str = "ucb_first_adaptive",
    ):
        self.scenario = scenario
        self.seed = seed
        self.model_key = str(model).lower()
        if self.model_key in {"moe", "adaptive", "observable_ucb", "mpp"}:
            self.model_key = "ucb_first_adaptive"
        if self.model_key not in MODEL_OPTIONS:
            raise ValueError(
                f"Unknown model {model!r}; choose from {sorted(MODEL_OPTIONS)}"
            )
        self.step_count = 0
        self.total_score = 0.0
        self.hits = 0
        self.misses = 0
        self.empty_scans = 0
        self.last_hit_band = None
        self.last_hit_emitter = None
        self.last_hit_step = None

        self.env = SmartScanEnv(
            num_bands=self.num_bands,
            episode_length=self.episode_length,
            seed=seed,
            scenario=scenario,
        )

        self.scheduler = UCBFirstAdaptiveScheduler(
            num_bands=self.num_bands,
            max_scan_age=self.episode_length,
        )
        self.model_display_name = MODEL_OPTIONS[self.model_key]

        obs, _ = self.env.reset(seed=seed)
        self.last_obs = obs

    def step(self) -> dict:
        if self.step_count >= self.episode_length:
            return {"done": True, "step": self.step_count}

        # Real scheduler action selection
        action = self.scheduler.select_band()

        # Real Environment execution
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Real scheduler update
        self.scheduler.update(action, reward, obs)
        self.last_obs = obs
        self.step_count += 1
        self.total_score += reward

        true_signal = bool(info.get("true_signal_present", False))
        detected = bool(obs.get("detected", 0) > 0.5)
        is_hit = bool((true_signal and detected) or reward > 0.5)

        # Extract real active emitters and bands from environment ground truth
        raw_bands = info.get("ground_truth_active_bands", [])
        raw_emitters = info.get("ground_truth_active_emitters", [])
        raw_snrs = info.get("ground_truth_active_snr_db", [])

        active_bands = [int(b) for b in raw_bands]
        emitter_map = {}
        for idx in range(len(raw_bands)):
            b = int(raw_bands[idx])
            e = str(raw_emitters[idx]) if idx < len(raw_emitters) else str(idx)
            snr = round(float(raw_snrs[idx]), 1) if idx < len(raw_snrs) else 15.0
            emitter_map[str(b)] = {"id": f"E{e}", "name": f"Emitter {e}", "snr": snr}

        if is_hit:
            self.hits += 1
            result_label = "HIT"
            self.last_hit_band = int(action)
            self.last_hit_emitter = emitter_map.get(str(action), {}).get("name", f"Emitter on Ch {action}")
            self.last_hit_step = self.step_count
        elif not true_signal and not detected:
            self.empty_scans += 1
            result_label = "EMPTY"
        else:
            self.misses += 1
            result_label = "MISS"

        # Extract real beliefs across paradigms
        if hasattr(self.scheduler, "runtime"):
            belief = self.scheduler.runtime.get_band_belief().tolist()
            scan_age = self.scheduler.runtime.get_scan_age().tolist()
        elif hasattr(self.scheduler, "predicted_spectrum"):
            belief = self.scheduler.predicted_spectrum.tolist()
            scan_age = self.scheduler.scan_age.tolist()
        elif hasattr(self.scheduler, "psr_state"):
            belief = self.scheduler.psr_state.tolist()
            scan_age = self.scheduler.scan_age.tolist()
        elif hasattr(self.scheduler, "belief"):
            belief = self.scheduler.belief.tolist()
            scan_age = self.scheduler.scan_age.tolist()
        elif hasattr(self.scheduler, "band_belief"):
            belief = self.scheduler.band_belief.tolist()
            scan_age = self.scheduler.scan_age.tolist()
        else:
            belief = [0.05] * self.num_bands
            scan_age = getattr(self.scheduler, "scan_age", np.zeros(self.num_bands)).tolist()

        # Extract active Track 2 confirmed tracks if present
        tracks_data = []
        if hasattr(self.scheduler, "runtime"):
            for track_id, track in self.scheduler.runtime.manager.tracks.items():
                if track.confirmed:
                    next_pred = None
                    if track.next_probabilities is not None:
                        next_pred = int(torch.argmax(track.next_probabilities).item())
                    tracks_data.append({
                        "id": str(track_id),
                        "band": int(track.last_band) if track.last_band is not None else action,
                        "hits": int(track.observations),
                        "quality": round(float(track.last_quality), 2),
                        "purity": round(float(track.association_similarity) * 100, 1),
                        "next_hop": next_pred,
                    })

        trace = getattr(self.scheduler, "last_decision_trace", {}) or getattr(
            self.scheduler, "last_trace", {}
        )
        expert_name = trace.get("expert", self.model_display_name)
        regime_name = trace.get("regime", "ONLINE")

        total_scans = max(1, self.step_count)
        correct_decisions = self.hits + self.empty_scans
        decision_accuracy = round((correct_decisions / total_scans) * 100, 1)
        target_attempts = self.hits + self.misses
        intercept_rate = round((self.hits / max(1, target_attempts)) * 100, 1)

        return {
            "done": False,
            "step": self.step_count,
            "scanned_band": int(action),
            "reward": float(reward),
            "total_score": round(self.total_score, 2),
            "hits": self.hits,
            "misses": self.misses,
            "empty_scans": self.empty_scans,
            "hit_rate": round((self.hits / total_scans) * 100, 1),
            "last_hit_band": self.last_hit_band,
            "last_hit_emitter": self.last_hit_emitter,
            "last_hit_step": self.last_hit_step,
            "decision_accuracy": decision_accuracy,
            "intercept_rate": intercept_rate,
            "result_label": result_label,
            "true_signal_present": true_signal,
            "detected": detected,
            "effective_snr": info.get("effective_snr_db"),
            "belief": [round(b, 3) for b in belief],
            "scan_age": scan_age,
            "active_bands": list(set(active_bands)),
            "emitter_map": emitter_map,
            "confirmed_tracks": tracks_data,
            "model_name": self.model_display_name,
            "expert": str(expert_name).upper(),
            "regime": str(regime_name).upper(),
        }

sim_manager = SimulationManager()

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = ROOT / "dashboard" / "index.html"
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
            return
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            data = {
                "step": sim_manager.step_count,
                "total_score": round(sim_manager.total_score, 2),
                "hits": sim_manager.hits,
                "scenario": sim_manager.scenario,
                "model_name": sim_manager.model_display_name,
                "is_real_python": True,
                "available_models": MODEL_OPTIONS,
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return
        return super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        if self.path == "/api/reset":
            scenario = payload.get("scenario", "stationary")
            seed = int(payload.get("seed", 42))
            model = payload.get("model", "ucb_first_adaptive")
            try:
                sim_manager.reset(scenario, seed, model)
            except (ValueError, FileNotFoundError) as error:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(error)}).encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "reset_ok",
                "scenario": scenario,
                "model_name": sim_manager.model_display_name,
            }).encode("utf-8"))
            return

        elif self.path == "/api/step":
            data = sim_manager.step()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

def run_server():
    print(f"[*] Starting SMART SCAN Real Python Server on http://localhost:{PORT}")
    print("[*] Running real Smart Scan schedulers; no browser-side model imitation.")
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", PORT), DashboardHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
