"""Verification of Run Provenance Consistency Across All Evaluation Artifacts.

Scans all output JSON artifacts in results/ and evaluation/, extracts each artifact's
run_id and git commit hash from its run_metadata block, and compares them against the
reference master statistical benchmark run.

Errors out loudly (exit code 1) if any artifact was produced by a different run or commit.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"c:\Users\asus\Documents\SMART SCAN")
RESULTS_DIR = ROOT / "results"
EVALUATION_DIR = ROOT / "evaluation"

REFERENCE_ARTIFACT = RESULTS_DIR / "master_15seed_statistical_benchmark.json"

TARGET_FILES = [
    RESULTS_DIR / "master_15seed_statistical_benchmark.json",
    RESULTS_DIR / "figures_of_merit_results.json",
    RESULTS_DIR / "perception_layer_latency.json",
    RESULTS_DIR / "adversarial_decoy_results.json",
]


def check_consistency():
    print("=" * 115)
    print("  SMARTSCAN EVALUATION SUITE: RUN PROVENANCE & ARTIFACT CONSISTENCY AUDIT")
    print("=" * 115)

    if not REFERENCE_ARTIFACT.exists():
        print(f"[FATAL ERROR] Reference artifact not found:\n  {REFERENCE_ARTIFACT}")
        print("Please run scripts/statistical_rigor_benchmark.py first.")
        sys.exit(1)

    with open(REFERENCE_ARTIFACT, "r", encoding="utf-8") as f:
        try:
            ref_data = json.load(f)
        except Exception as e:
            print(f"[FATAL ERROR] Failed to parse reference artifact: {e}")
            sys.exit(1)

    ref_meta = ref_data.get("run_metadata", {})
    ref_run_id = ref_meta.get("run_id")
    ref_commit = ref_meta.get("git_commit") or ref_meta.get("git_commit_hash")

    if not ref_run_id or not ref_commit:
        print("[FATAL ERROR] Reference artifact is missing canonical run_metadata (run_id or git_commit).")
        sys.exit(1)

    print(f"Reference Benchmark Run ID : {ref_run_id}")
    print(f"Reference Git Commit Hash  : {ref_commit}")
    print("-" * 115)
    print(f"{'Artifact File':<42} | {'Status':<14} | {'Run ID':<30} | {'Commit'}")
    print("-" * 115)

    # Collect all candidate json files in results/ and evaluation/
    candidates = list(TARGET_FILES)
    for p in RESULTS_DIR.glob("*.json"):
        if p not in candidates and not p.name.startswith("."):
            candidates.append(p)
    for p in EVALUATION_DIR.glob("*results*.json"):
        if p not in candidates:
            candidates.append(p)

    mismatches = []
    missing_files = []
    verified_count = 0

    for path in candidates:
        rel_name = path.relative_to(ROOT)
        if not path.exists():
            if path in TARGET_FILES:
                missing_files.append(rel_name)
                print(f"{str(rel_name):<42} | {'[MISSING]':<14} | {'--':<30} | {'--'}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        meta = data.get("run_metadata")
        if not meta:
            # Skip non-provenance data files if not in core target files
            if path in TARGET_FILES:
                mismatches.append((rel_name, "Missing 'run_metadata' block", None, None))
                print(f"{str(rel_name):<42} | {'[NO_METADATA]':<14} | {'--':<30} | {'--'}")
            continue

        art_run_id = meta.get("run_id")
        art_commit = meta.get("git_commit") or meta.get("git_commit_hash")

        run_id_match = (art_run_id == ref_run_id)
        commit_match = (art_commit == ref_commit)

        if run_id_match and commit_match:
            status_str = "[MATCH]"
            verified_count += 1
        else:
            status_str = "[MISMATCH]"
            reasons = []
            if not run_id_match:
                reasons.append(f"run_id expected '{ref_run_id}', got '{art_run_id}'")
            if not commit_match:
                reasons.append(f"commit expected '{ref_commit[:7]}', got '{art_commit[:7] if art_commit else 'None'}'")
            mismatches.append((rel_name, "; ".join(reasons), art_run_id, art_commit))

        c_short = art_commit[:10] if art_commit else "none"
        id_short = (art_run_id[:28] + "..") if art_run_id and len(art_run_id) > 30 else (art_run_id or "none")
        print(f"{str(rel_name):<42} | {status_str:<14} | {id_short:<30} | {c_short}")

    print("=" * 115)

    if missing_files or mismatches:
        print("\n" + "!" * 115)
        print("  [ERROR] RUN PROVENANCE AUDIT FAILED - ARTIFACTS DO NOT BELONG TO THE SAME BENCHMARK SESSION")
        print("!" * 115)
        if missing_files:
            print("\nMissing Required Artifacts:")
            for mf in missing_files:
                print(f"  * {mf}")
        if mismatches:
            print("\nMismatched Artifacts:")
            for rel_name, reason, art_id, art_c in mismatches:
                print(f"  * {rel_name}: {reason}")
        print("\nRemedy: Re-run the canonical suite pipeline in order:")
        print("  1. python scripts/statistical_rigor_benchmark.py --workers 12")
        print("  2. python evaluation/run_figures_of_merit.py")
        print("  3. python evaluation/adversarial_decoy_test.py")
        print("  4. python evaluation/benchmark_perception_latency.py")
        print("  5. python results/presentation_charts.py")
        print("!" * 115)
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] AUDIT PASSED: All {verified_count} core evaluation artifacts verified!")
        print(f"Every metric, chart, and JSON artifact consistently shares:")
        print(f"  * Suite Run ID     : {ref_run_id}")
        print(f"  * Git Commit Hash  : {ref_commit}")
        print("=" * 115)
        sys.exit(0)


if __name__ == "__main__":
    check_consistency()
