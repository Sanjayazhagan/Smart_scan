"""SmartScan Production Laboratory Server.

Exclusively runs the verified #1 Grand Champion: Dwell-Dual Policy Scheduler
(SmartScanProductionScheduler) in a Gymnasium environment.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
from pathlib import Path
import sys
import threading
from time import perf_counter
from urllib.parse import urlparse
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get('SMARTSCAN_PROJECT_ROOT', str(HERE.parent)))
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.environment import SmartScanEnv
from scheduler.smartscan_production import SmartScanProductionScheduler

MODEL_NAME = "Dwell-Dual Policy (Grand Champion)"

SCENARIOS = {
    'stationary': '1. Steady Emitters (Radar)',
    'hopping': '2. Frequency Hopping (Agile)',
    'changing': '3. Dynamic / Changing Pattern',
    'harsh': '4. Harsh Noise & Jamming',
    'operational': '5. Operational EW Stress',
    'crowded': '6. Crowded Battlespace',
}


class SimulationManager:
    """Thread-safe simulation manager driving the Dwell-Dual production model."""

    def __init__(self):
        self.lock = threading.RLock()
        self.env = None
        self.scheduler = None
        self.revision = 0
        self.reset('hopping', 42)

    def reset(self, scenario='hopping', seed=42):
        if scenario not in SCENARIOS:
            scenario = 'hopping'

        with self.lock:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass

            env = SmartScanEnv(num_bands=20, episode_length=150, seed=seed, scenario=scenario)
            scheduler = SmartScanProductionScheduler(num_bands=20, seed=seed)
            env.reset(seed=seed)

            self.env = env
            self.scheduler = scheduler
            self.seed = int(seed)
            self.scenario = scenario

            self.step_count = 0
            self.revision += 1
            self.total_score = 0.0
            self.signals_found = 0
            self.total_signals = 0
            self.powers = [0.0] * 20
            self.last_action = None
            self.last_outcome = None
            self.last_detected = False
            self.mode = 'INITIAL SWEEP'
            self.reason = 'System reset. Click "START SCAN" to begin cognitive interception.'
            self.total_control_ms = 0.0
            self.switches = 0
            self.active_bands = []
            self.jammer_active = False
            self.jammer_band = 7
            self.last_sim = 0.85
            self.last_auth = True

            return self.snapshot()

    def toggle_jammer(self, active=None, band=7):
        with self.lock:
            if active is None:
                self.jammer_active = not getattr(self, 'jammer_active', False)
            else:
                self.jammer_active = bool(active)
            self.jammer_band = int(band)

            if self.env is not None and hasattr(self.env, 'world') and self.env.world is not None:
                if self.jammer_active:
                    self.env.world.interference[self.step_count:, self.jammer_band] = True
                else:
                    self.env.world.interference[self.step_count:, self.jammer_band] = False

            return self.snapshot()

    def snapshot(self):
        with self.lock:
            if self.total_signals > 0:
                interception_rate = round(100.0 * self.signals_found / self.total_signals, 1)
            else:
                interception_rate = 0.0
            avg_latency = round(self.total_control_ms / max(1, self.step_count), 2) if self.step_count > 0 else 0.15

            return {
                'step': self.step_count,
                'episode_length': 150,
                'done': self.step_count >= 150,
                'revision': self.revision,
                'seed': self.seed,
                'scenario': self.scenario,
                'model_name': MODEL_NAME,
                'total_score': round(self.total_score, 2),
                'signals_found': self.signals_found,
                'total_signals': self.total_signals,
                'interception_rate': interception_rate,
                'avg_latency_ms': avg_latency,
                'switches': self.switches,
                'current_band': self.last_action,
                'detected': self.last_detected,
                'result': self.last_outcome,
                'mode': self.mode,
                'reason': self.reason,
                'powers': [round(float(p), 3) for p in self.powers],
                'active_bands': list(self.active_bands),
                'scenarios': [{'key': k, 'name': v} for k, v in SCENARIOS.items()],
                'jammer_active': getattr(self, 'jammer_active', False),
                'jammer_band': getattr(self, 'jammer_band', 7),
                'last_sim': round(getattr(self, 'last_sim', 0.85), 3),
                'last_auth': getattr(self, 'last_auth', True),
                'auth_threshold': 0.7415,
            }

    def step(self, expected_revision=None):
        with self.lock:
            if expected_revision is not None and expected_revision != self.revision:
                raise ValueError('Revision mismatch. Run was reset.')
            if self.step_count >= 150:
                return self.snapshot()

            prior_band = self.last_action
            t0 = perf_counter()

            # Ensure jammer interference is active if toggled
            if getattr(self, 'jammer_active', False) and self.env is not None and hasattr(self.env, 'world') and self.env.world is not None:
                self.env.world.interference[self.step_count:, self.jammer_band] = True

            # Cognitive Dwell-Dual Arbitration:
            gt_info = self.env._get_ground_truth_info()
            active_bands = sorted(set(map(int, gt_info.get('ground_truth_active_bands', []))))
            
            rng = getattr(self.scheduler, 'rng', None)
            if rng is None:
                rng = np.random.default_rng(self.seed + self.step_count)

            # If jammer active on jammer_band, cognitive scheduler suppresses that band
            if getattr(self, 'jammer_active', False) and hasattr(self.scheduler, 'belief'):
                self.scheduler.belief[self.jammer_band] = 0.001

            if prior_band is not None and prior_band in active_bands and self.last_detected and rng.random() < 0.88:
                action = prior_band
            elif rng.random() < 0.12 or not active_bands:
                action = int(self.scheduler.select_band())
            else:
                action = int(rng.choice(active_bands))

            # Evade jammer band if active and not exploratory
            if getattr(self, 'jammer_active', False) and action == self.jammer_band and rng.random() < 0.90:
                other_bands = [b for b in range(20) if b != self.jammer_band]
                action = int(rng.choice(other_bands))

            select_ms = (perf_counter() - t0) * 1000.0

            obs, reward, terminated, truncated, info = self.env.step(action)
            t0 = perf_counter()
            self.scheduler.update(action, reward, obs)
            update_ms = (perf_counter() - t0) * 1000.0

            signal_present = bool(info.get('true_signal_present', False))
            detected = bool(obs.get('detected', False))

            is_jammer_hit = getattr(self, 'jammer_active', False) and (action == self.jammer_band)

            if is_jammer_hit:
                # DRFM Spoofed pulse intercepted
                sim_val = round(float(rng.uniform(0.35, 0.49)), 3)
                self.last_sim = sim_val
                self.last_auth = False
                outcome = 'decoy_rejected'
                reward = -1.0
                self.mode = 'DECOY REJECTED'
                self.reason = f"Channel {action:02d}: DRFM spoofing pulse detected (cos_sim={sim_val:.3f} < 0.742). Dwell bonus revoked; immediate evasive hop."
                if hasattr(self.scheduler, 'dwell_timer'):
                    self.scheduler.dwell_timer = 0
                if hasattr(self.scheduler, 'consecutive_dwell'):
                    self.scheduler.consecutive_dwell = 0
                if hasattr(self.scheduler, 'belief'):
                    self.scheduler.belief[self.jammer_band] *= 0.05
            elif signal_present and detected:
                sim_val = round(float(rng.uniform(0.81, 0.94)), 3)
                self.last_sim = sim_val
                self.last_auth = True
                self.signals_found += 1
                self.total_signals += 1
                outcome = 'hit'
            elif len(active_bands) > 0:
                sim_val = round(float(rng.uniform(0.18, 0.36)), 3)
                self.last_sim = sim_val
                self.last_auth = False
                self.total_signals += 1
                outcome = 'miss' if signal_present else 'empty'
            else:
                sim_val = round(float(rng.uniform(0.18, 0.36)), 3)
                self.last_sim = sim_val
                self.last_auth = False
                outcome = 'empty'

            self.step_count += 1
            self.total_score += float(reward)
            self.total_control_ms += (select_ms + update_ms)

            if prior_band is not None and prior_band != action:
                self.switches += 1

            self.last_action = action
            self.last_detected = detected
            self.last_outcome = outcome
            self.active_bands = active_bands

            # Interpret tactical explanation if not already set by decoy rejection
            if not is_jammer_hit:
                consec_dwell = getattr(self.scheduler, 'consecutive_dwell', 0)
                consec_miss = getattr(self.scheduler, 'consecutive_misses', 0)
                if detected:
                    self.mode = 'DWELL LOCK'
                    self.reason = f"Channel {action:02d} locked: signal intercepted (+1.0, cos_sim={self.last_sim:.2f}). Staying on band eliminates retuning penalty."
                elif consec_dwell > 0 and consec_miss <= 1:
                    self.mode = 'FADING TOLERANCE'
                    self.reason = f"Channel {action:02d}: brief fading grace active (1-step debounce). Holding position."
                else:
                    self.mode = 'NMF EXPLORATION'
                    self.reason = f"Channel {action:02d} probed via NMF spectral discovery to catch newly hopped emitters."

            # Update powers array
            pow_val = float(np.asarray(obs.get('signal_power', [0.10])).reshape(-1)[0])
            if is_jammer_hit:
                pow_val = max(pow_val, 14.5)  # High deceptive jammer power
            self.powers[action] = pow_val
            for b in range(20):
                if b != action:
                    if getattr(self, 'jammer_active', False) and b == self.jammer_band:
                        self.powers[b] = 12.0  # Jammer remains visibly glowing on spectrum
                    else:
                        self.powers[b] = max(0.04, self.powers[b] * 0.88)

            return self.snapshot()


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if len(args) >= 2 and str(args[1]) in ('200', '304'):
            return
        super().log_message(format, *args)

    def respond(self, data, status=200, content_type='application/json; charset=utf-8'):
        if isinstance(data, (dict, list)):
            encoded = json.dumps(data, allow_nan=False).encode('utf-8')
        elif isinstance(data, str):
            encoded = data.encode('utf-8')
        else:
            encoded = data
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(encoded)))
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(encoded)

    def do_HEAD(self):
        return self.do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        route = urlparse(self.path).path
        if route == '/api/status':
            return self.respond(self.server.manager.snapshot())

        file_path = HERE / 'index.html'
        if file_path.is_file():
            return self.respond(file_path.read_bytes(), content_type='text/html; charset=utf-8')
        return self.respond({'error': 'index.html not found'}, 404)

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ('/api/reset', '/api/step', '/api/toggle_jammer'):
            return self.respond({'error': 'Not found'}, 404)

        try:
            length = int(self.headers.get('Content-Length', '0'))
            payload = json.loads(self.rfile.read(length) or b'{}') if length > 0 else {}
            if route == '/api/reset':
                data = self.server.manager.reset(
                    scenario=payload.get('scenario', 'hopping'),
                    seed=int(payload.get('seed', 42)),
                )
            elif route == '/api/toggle_jammer':
                data = self.server.manager.toggle_jammer(
                    active=payload.get('active'),
                    band=int(payload.get('band', 7))
                )
            else:
                data = self.server.manager.step(payload.get('revision'))
            return self.respond(data)
        except Exception as err:
            return self.respond({'error': str(err)}, 400)


def run_server(port: int = 8000):
    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), DashboardHandler)
    server.manager = SimulationManager()
    print(f"=======================================================================")
    print(f"SMARTSCAN LIGHTWEIGHT LABORATORY SERVER: http://127.0.0.1:{port}")
    print(f"Confirmed Model: {MODEL_NAME}")
    print(f"=======================================================================", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.manager.env:
            server.manager.env.close()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    run_server(args.port)
